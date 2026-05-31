#!/usr/bin/env python3
"""
Compare radial NUFFT adjoint reconstruction and Cartesian undersampling
from a NIfTI-derived 2D ground-truth image.

Main purposes:
    1. noise_sigma sweep
    2. radial spoke-count sweep
    3. Cartesian undersampling comparison with matched readout budget

Current simplifications:
    - 2D slice only
    - single-coil
    - zero phase object
    - no motion
    - no coil sensitivity
    - no B0/B1 effects
    - radial reconstruction is adjoint NUFFT, not optimized iterative reconstruction
    - Cartesian reconstruction is zero-filled IFFT

Interpretation:
    This is an image-derived synthetic k-space sampling prototype,
    not a scanner-realistic acquisition simulation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.ndimage import shift as ndi_shift
# ----------------------------
# Basic image utilities
# ----------------------------

def load_nifti_slice(
    nifti_path: Path,
    axis: int = 2,
    index: int | None = None,
) -> tuple[np.ndarray, int]:
    img = nib.load(str(nifti_path))
    vol = img.get_fdata().astype(np.float32)

    if vol.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {vol.shape}")

    if index is None:
        index = vol.shape[axis] // 2

    if axis == 0:
        sl = vol[index, :, :]
    elif axis == 1:
        sl = vol[:, index, :]
    elif axis == 2:
        sl = vol[:, :, index]
    else:
        raise ValueError("--axis must be 0, 1, or 2")

    return np.asarray(sl, dtype=np.float32), index


def robust_normalize(
    x: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)

    if not np.any(finite):
        raise ValueError("Input slice has no finite values.")

    vals = x[finite]
    lo, hi = np.percentile(vals, [p_low, p_high])

    if hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))

    y = (x - lo) / (hi - lo + 1e-8)
    y = np.clip(y, 0.0, 1.0)
    y[~finite] = 0.0
    return y.astype(np.float32)


def center_crop_or_pad_2d(x: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=x.dtype)

    in_h, in_w = x.shape
    out_h, out_w = shape

    crop_h = min(in_h, out_h)
    crop_w = min(in_w, out_w)

    in_y0 = (in_h - crop_h) // 2
    in_x0 = (in_w - crop_w) // 2
    out_y0 = (out_h - crop_h) // 2
    out_x0 = (out_w - crop_w) // 2

    out[out_y0:out_y0 + crop_h, out_x0:out_x0 + crop_w] = x[
        in_y0:in_y0 + crop_h,
        in_x0:in_x0 + crop_w,
    ]

    return out


def fft2c(img: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img), norm="ortho"))


def ifft2c(ksp: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ksp), norm="ortho"))


def add_complex_noise(
    kspace: np.ndarray,
    sigma: float,
    seed: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Add complex Gaussian noise to k-space.

    If mask is provided, noise is added only where mask is True.
    This is useful for Cartesian undersampled k-space.
    """
    if sigma <= 0:
        return kspace

    rng = np.random.default_rng(seed)
    noise = sigma * (
        rng.standard_normal(kspace.shape) + 1j * rng.standard_normal(kspace.shape)
    )

    noise = noise.astype(kspace.dtype, copy=False)

    if mask is not None:
        out = kspace.copy()
        out[mask] = out[mask] + noise[mask]
        return out

    return kspace + noise


def rescale_to_gt_percentile(
    recon: np.ndarray,
    gt: np.ndarray,
    percentile: float = 99.0,
) -> tuple[np.ndarray, float]:
    """
    Simple visual/metric scaling.

    This is not a physical receive gain model.
    It only removes arbitrary scaling differences between recon methods.
    """
    recon = np.asarray(recon, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    gt_ref = np.percentile(gt[np.isfinite(gt)], percentile)
    rec_ref = np.percentile(recon[np.isfinite(recon)], percentile)

    scale = float(gt_ref / (rec_ref + 1e-8))
    return recon * scale, scale


# ----------------------------
# Metrics
# ----------------------------

def compute_metrics(
    gt: np.ndarray,
    recon: np.ndarray,
    mask_threshold: float = 0.02,
) -> dict[str, float]:
    """
    Compute metrics inside a foreground mask.

    The mask prevents the large zero background from dominating metrics.
    """
    gt = np.asarray(gt, dtype=np.float32)
    recon = np.asarray(recon, dtype=np.float32)

    mask = np.isfinite(gt) & np.isfinite(recon) & (gt > mask_threshold)

    if np.count_nonzero(mask) < 10:
        mask = np.isfinite(gt) & np.isfinite(recon)

    g = gt[mask].astype(np.float64)
    r = recon[mask].astype(np.float64)

    diff = r - g

    rmse = float(np.sqrt(np.mean(diff**2)))
    nrmse_range = float(rmse / (np.max(g) - np.min(g) + 1e-8))
    nrmse_mean = float(rmse / (np.mean(g) + 1e-8))
    mae = float(np.mean(np.abs(diff)))

    if np.std(g) > 0 and np.std(r) > 0:
        corr = float(np.corrcoef(g, r)[0, 1])
    else:
        corr = float("nan")

    try:
        from skimage.metrics import structural_similarity as ssim

        gt_clip = np.clip(gt, 0.0, 1.0)
        rec_clip = np.clip(recon, 0.0, 1.0)
        ssim_val = float(ssim(gt_clip, rec_clip, data_range=1.0))
    except Exception:
        ssim_val = float("nan")

    return {
        "n_foreground_voxels": float(np.count_nonzero(mask)),
        "rmse": rmse,
        "nrmse_range": nrmse_range,
        "nrmse_mean": nrmse_mean,
        "mae": mae,
        "corr": corr,
        "ssim": ssim_val,
    }


# ----------------------------
# Cartesian undersampling
# ----------------------------

def make_cartesian_line_mask(
    shape: tuple[int, int],
    n_lines: int,
    axis: int = 0,
    mode: str = "uniform",
    center_fraction: float = 0.08,
) -> np.ndarray:
    """
    Make Cartesian undersampling mask.

    axis=0:
        sample rows in k-space.
    axis=1:
        sample columns in k-space.

    mode:
        uniform:
            uniformly spaced phase-encode lines.
        center_uniform:
            include a small fully sampled center region, then fill remaining
            lines uniformly outside the center.

    For 2D image shape (Nx, Ny), this selects n_lines phase-encode lines,
    while readout direction is fully sampled.
    """
    nx, ny = shape
    mask = np.zeros(shape, dtype=bool)

    if axis == 0:
        n_phase = nx
    elif axis == 1:
        n_phase = ny
    else:
        raise ValueError("axis must be 0 or 1")

    n_lines = int(max(1, min(n_lines, n_phase)))

    if mode == "uniform":
        idx = np.linspace(0, n_phase - 1, n_lines)
        idx = np.unique(np.round(idx).astype(int))

        # If rounding created too few unique lines, fill missing from center outward.
        if len(idx) < n_lines:
            candidates = np.argsort(np.abs(np.arange(n_phase) - n_phase // 2))
            extra = []
            existing = set(idx.tolist())
            for c in candidates:
                if c not in existing:
                    extra.append(c)
                    existing.add(c)
                if len(idx) + len(extra) >= n_lines:
                    break
            idx = np.array(sorted(list(existing)), dtype=int)

    elif mode == "center_uniform":
        n_center = int(round(center_fraction * n_phase))
        n_center = max(1, min(n_center, n_lines))

        c0 = n_phase // 2 - n_center // 2
        center_idx = np.arange(c0, c0 + n_center)
        center_idx = np.clip(center_idx, 0, n_phase - 1)

        remaining = n_lines - len(np.unique(center_idx))

        if remaining > 0:
            all_idx = np.arange(n_phase)
            outside = np.setdiff1d(all_idx, center_idx)
            if remaining >= len(outside):
                outer_idx = outside
            else:
                outer_pos = np.linspace(0, len(outside) - 1, remaining)
                outer_idx = outside[np.round(outer_pos).astype(int)]
            idx = np.unique(np.concatenate([center_idx, outer_idx]))
        else:
            idx = np.unique(center_idx)

    else:
        raise ValueError(f"Unknown Cartesian mask mode: {mode}")

    if axis == 0:
        mask[idx, :] = True
    else:
        mask[:, idx] = True

    return mask


def reconstruct_cartesian_zerofill(
    obj: np.ndarray,
    n_lines: int,
    mask_axis: int,
    mask_mode: str,
    noise_sigma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cartesian undersampling with zero-filled reconstruction.
    """
    k_full = fft2c(obj)

    mask = make_cartesian_line_mask(
        shape=obj.shape,
        n_lines=n_lines,
        axis=mask_axis,
        mode=mask_mode,
    )

    k_sampled = np.zeros_like(k_full)
    k_sampled[mask] = k_full[mask]

    k_noisy = add_complex_noise(
        k_sampled,
        sigma=noise_sigma,
        seed=seed,
        mask=mask,
    )

    recon = ifft2c(k_noisy)
    recon_mag = np.abs(recon).astype(np.float32)

    return recon_mag, k_noisy, mask


# ----------------------------
# Radial NUFFT
# ----------------------------

def reconstruct_radial_adjoint(
    obj: np.ndarray,
    n_spokes: int,
    n_samples: int,
    backend: str,
    density: str | None,
    noise_sigma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Radial synthetic k-space and adjoint NUFFT reconstruction.
    """
    import mrinufft

    samples_loc = mrinufft.initialize_2D_radial(
        Nc=n_spokes,
        Ns=n_samples,
    )

    kwargs = dict(
        samples=samples_loc,
        shape=obj.shape,
        n_coils=1,
    )

    if density is not None and density.lower() not in {"none", "null", "no"}:
        kwargs["density"] = density

    nufft = mrinufft.get_operator(
        backend,
        **kwargs,
    )

    k_radial = nufft.op(obj)
    k_radial_noisy = add_complex_noise(k_radial, sigma=noise_sigma, seed=seed)

    recon = nufft.adj_op(k_radial_noisy)
    recon_mag = np.abs(recon).astype(np.float32)

    return recon_mag, k_radial_noisy, samples_loc

