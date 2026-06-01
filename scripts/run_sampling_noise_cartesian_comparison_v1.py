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


def get_motion_shift(
    t: float,
    motion_mode: str,
    motion_shift_px: float,
    motion_start_frac: float = 0.5,
) -> float:
    if motion_mode == "none":
        return 0.0

    if motion_mode == "step":
        return motion_shift_px if t >= motion_start_frac else 0.0

    if motion_mode == "drift":
        return motion_shift_px * t

    if motion_mode == "periodic":
        return motion_shift_px * np.sin(2 * np.pi * t)

    raise ValueError(f"Unknown motion_mode: {motion_mode}")


def apply_motion_translation(
    obj: np.ndarray,
    shift_px: float,
    motion_axis: int = 0,
) -> np.ndarray:
    shift_vec = [0.0, 0.0]
    shift_vec[motion_axis] = shift_px

    real = ndi_shift(
        np.real(obj),
        shift=tuple(shift_vec),
        order=1,
        mode="constant",
        cval=0.0,
    )
    imag = ndi_shift(
        np.imag(obj),
        shift=tuple(shift_vec),
        order=1,
        mode="constant",
        cval=0.0,
    )

    return (real + 1j * imag).astype(obj.dtype, copy=False)
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


# ----------------------------
# Output
# ----------------------------

def write_metrics_csv(out_csv: Path, rows: list[dict]) -> None:
    fieldnames = [
        "method",
        "n_spokes",
        "n_cart_lines",
        "n_samples_per_readout",
        "n_total_samples",
        "noise_sigma",
        "scale",
        "cart_mask_mode",
        "cart_mask_axis",
        "n_foreground_voxels",
        "rmse",
        "nrmse_range",
        "nrmse_mean",
        "mae",
        "corr",
        "ssim",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            clean = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(clean)


def save_noise_sweep_metric_plot(
    out_png: Path,
    rows: list[dict],
    fixed_n_spokes: int,
) -> None:
    sigmas = sorted(set(float(r["noise_sigma"]) for r in rows))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metric_names = [
        ("nrmse_range", "NRMSE", False),
        ("ssim", "SSIM", True),
        ("corr", "Correlation", True),
    ]

    for ax, (metric, ylabel, high_good) in zip(axes, metric_names):
        for method in ["radial_adjoint", "cartesian_zerofill"]:
            xs = []
            ys = []
            for sigma in sigmas:
                matches = [
                    r for r in rows
                    if r["method"] == method
                    and int(r["n_spokes"]) == fixed_n_spokes
                    and float(r["noise_sigma"]) == sigma
                ]
                if not matches:
                    continue
                xs.append(sigma)
                ys.append(float(matches[0][metric]))

            if xs:
                label = "Radial adjoint" if method == "radial_adjoint" else "Cartesian zero-fill"
                ax.plot(xs, ys, marker="o", label=label)

        ax.set_xlabel("k-space noise sigma")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"Noise sigma sweep at sampling budget = {fixed_n_spokes} lines/spokes")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_sampling_comparison_panel(
    out_png: Path,
    gt: np.ndarray,
    cart_full: np.ndarray,
    rows: list[dict],
    noise_sigma: float,
    n_list: list[int],
    title: str,
) -> None:
    selected = [r for r in rows if float(r["noise_sigma"]) == float(noise_sigma)]

    n_cols = 2 + len(n_list)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.2 * n_cols, 9.0))

    vmax = float(np.percentile(gt[np.isfinite(gt)], 99.5))

    axes[0, 0].imshow(gt.T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
    axes[0, 0].set_title("GT")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cart_full.T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
    axes[0, 1].set_title("Full Cartesian\nFFT/IFFT")
    axes[0, 1].axis("off")

    for j in range(2, n_cols):
        axes[0, j].axis("off")

    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, "Radial\nadjoint", ha="center", va="center", fontsize=12)
    axes[1, 1].axis("off")

    for j, n in enumerate(n_list, start=2):
        match = [
            r for r in selected
            if r["method"] == "radial_adjoint" and int(r["n_spokes"]) == int(n)
        ]
        if match:
            r = match[0]
            axes[1, j].imshow(r["recon_scaled"].T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
            axes[1, j].set_title(
                f"{n} spokes\n"
                f"NRMSE={r['nrmse_range']:.3f}\n"
                f"SSIM={r['ssim']:.3f}"
            )
        axes[1, j].axis("off")

    axes[2, 0].axis("off")
    axes[2, 0].text(0.5, 0.5, "Cartesian\nzero-fill", ha="center", va="center", fontsize=12)
    axes[2, 1].axis("off")

    for j, n in enumerate(n_list, start=2):
        match = [
            r for r in selected
            if r["method"] == "cartesian_zerofill" and int(r["n_cart_lines"]) == int(n)
        ]
        if match:
            r = match[0]
            axes[2, j].imshow(r["recon_scaled"].T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
            axes[2, j].set_title(
                f"{n} lines\n"
                f"NRMSE={r['nrmse_range']:.3f}\n"
                f"SSIM={r['ssim']:.3f}"
            )
        axes[2, j].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_sampling_metric_plot(
    out_png: Path,
    rows: list[dict],
    noise_sigma: float,
) -> None:
    selected = [r for r in rows if float(r["noise_sigma"]) == float(noise_sigma)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metric_names = [
        ("nrmse_range", "NRMSE"),
        ("ssim", "SSIM"),
        ("corr", "Correlation"),
    ]

    for ax, (metric, ylabel) in zip(axes, metric_names):
        for method in ["radial_adjoint", "cartesian_zerofill"]:
            xs = []
            ys = []

            for r in selected:
                if r["method"] != method:
                    continue
                x = int(r["n_spokes"]) if method == "radial_adjoint" else int(r["n_cart_lines"])
                xs.append(x)
                ys.append(float(r[metric]))

            order = np.argsort(xs)
            xs = np.array(xs)[order]
            ys = np.array(ys)[order]

            label = "Radial adjoint" if method == "radial_adjoint" else "Cartesian zero-fill"
            ax.plot(xs, ys, marker="o", label=label)

        ax.set_xlabel("Number of spokes / Cartesian lines")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"Sampling comparison at noise_sigma={noise_sigma}")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def reconstruct_cartesian_zerofill_with_motion(
    obj: np.ndarray,
    n_lines: int,
    mask_axis: int,
    mask_mode: str,
    noise_sigma: float,
    seed: int,
    motion_mode: str = "none",
    motion_shift_px: float = 0.0,
    motion_axis: int = 0,
    motion_start_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_ref = fft2c(obj)

    mask = make_cartesian_line_mask(
        shape=obj.shape,
        n_lines=n_lines,
        axis=mask_axis,
        mode=mask_mode,
    )

    k_sampled = np.zeros_like(k_ref)

    if mask_axis == 0:
        sampled_lines = np.where(mask[:, 0])[0]
    else:
        sampled_lines = np.where(mask[0, :])[0]

    n_acq = len(sampled_lines)

    for acq_i, line in enumerate(sampled_lines):
        t = acq_i / max(1, n_acq - 1)

        shift_px_t = get_motion_shift(
            t=t,
            motion_mode=motion_mode,
            motion_shift_px=motion_shift_px,
            motion_start_frac=motion_start_frac,
        )

        obj_t = apply_motion_translation(
            obj=obj,
            shift_px=shift_px_t,
            motion_axis=motion_axis,
        )

        k_t = fft2c(obj_t)

        if mask_axis == 0:
            k_sampled[line, :] = k_t[line, :]
        else:
            k_sampled[:, line] = k_t[:, line]

    k_noisy = add_complex_noise(
        k_sampled,
        sigma=noise_sigma,
        seed=seed,
        mask=mask,
    )

    recon = ifft2c(k_noisy)
    recon_mag = np.abs(recon).astype(np.float32)

    return recon_mag, k_noisy, mask


def reconstruct_radial_adjoint_with_motion(
    obj: np.ndarray,
    n_spokes: int,
    n_samples: int,
    backend: str,
    density: str | None,
    noise_sigma: float,
    seed: int,
    motion_mode: str = "none",
    motion_shift_px: float = 0.0,
    motion_axis: int = 0,
    motion_start_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import mrinufft

    samples_loc = mrinufft.initialize_2D_radial(
        Nc=n_spokes,
        Ns=n_samples,
    )

    k_list = []

    for spoke_i in range(n_spokes):
        t = spoke_i / max(1, n_spokes - 1)

        shift_px_t = get_motion_shift(
            t=t,
            motion_mode=motion_mode,
            motion_shift_px=motion_shift_px,
            motion_start_frac=motion_start_frac,
        )

        obj_t = apply_motion_translation(
            obj=obj,
            shift_px=shift_px_t,
            motion_axis=motion_axis,
        )

        samples_spoke = samples_loc[spoke_i:spoke_i + 1, :, :]

        nufft_spoke = mrinufft.get_operator(
            backend,
            samples=samples_spoke,
            shape=obj.shape,
            n_coils=1,
        )

        k_spoke = nufft_spoke.op(obj_t)
        k_list.append(k_spoke)

    k_radial = np.concatenate(k_list, axis=0)

    k_radial_noisy = add_complex_noise(
        k_radial,
        sigma=noise_sigma,
        seed=seed,
    )

    kwargs_full = dict(
        samples=samples_loc,
        shape=obj.shape,
        n_coils=1,
    )

    if density is not None and density.lower() not in {"none", "null", "no"}:
        kwargs_full["density"] = density

    nufft_full = mrinufft.get_operator(
        backend,
        **kwargs_full,
    )

    recon = nufft_full.adj_op(k_radial_noisy)
    recon_mag = np.abs(recon).astype(np.float32)

    return recon_mag, k_radial_noisy, samples_loc


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--nifti", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)

    parser.add_argument("--axis", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--index", type=int, default=None)

    parser.add_argument("--matrix", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=256)

    parser.add_argument(
        "--n-list",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="Radial spokes and Cartesian phase-encode lines to compare.",
    )

    parser.add_argument(
        "--noise-sigma-list",
        type=float,
        nargs="+",
        default=[0.0, 0.001, 0.01],
    )

    parser.add_argument("--backend", default="finufft")
    parser.add_argument("--density", default="voronoi")

    parser.add_argument("--cart-mask-axis", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--cart-mask-mode",
        default="uniform",
        choices=["uniform", "center_uniform"],
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--norm-p-low", type=float, default=1.0)
    parser.add_argument("--norm-p-high", type=float, default=99.0)
    parser.add_argument("--mask-threshold", type=float, default=0.02)
    parser.add_argument("--save-npy", action="store_true")

    parser.add_argument(
        "--motion-mode",
        default="none",
        choices=["none", "step", "drift", "periodic"],
        help="Acquisition-time object motion model.",
    )
    parser.add_argument(
        "--motion-shift-px",
        type=float,
        default=0.0,
        help="Maximum translation in pixels for the motion model.",
    )
    parser.add_argument(
        "--motion-axis",
        type=int,
        default=0,
        choices=[0, 1],
        help="Image axis along which translation is applied.",
    )
    parser.add_argument(
        "--motion-start-frac",
        type=float,
        default=0.5,
        help="For step motion, fraction of acquisition after which motion occurs.",
    )

    parser.add_argument("--skip-radial", action="store_true", help="Skip radial NUFFT reconstructions (quick tests).")

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Optionally check NUFFT only when not skipping radial
    if not args.skip_radial:
        try:
            import mrinufft  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "Failed to import mrinufft. Activate kspace_nufft env and install:\n"
                '  python -m pip install "mri-nufft[finufft]"\n'
            ) from e

    raw_slice, used_index = load_nifti_slice(
        args.nifti,
        axis=args.axis,
        index=args.index,
    )

    gt = robust_normalize(
        raw_slice,
        p_low=args.norm_p_low,
        p_high=args.norm_p_high,
    )
    gt = center_crop_or_pad_2d(gt, (args.matrix, args.matrix))

    obj = gt.astype(np.complex64)

    # Full Cartesian reference.
    k_cart_full = fft2c(obj)
    cart_full = np.abs(ifft2c(k_cart_full)).astype(np.float32)

    rows: list[dict] = []

    run_id = 0

    for sigma in args.noise_sigma_list:
        for n in args.n_list:
            print(f"Running sigma={sigma}, n={n}")

            # ----------------------------
            # Radial (skip if requested)
            # ----------------------------
            radial_raw = None
            if not args.skip_radial:
                radial_raw, k_radial, samples_loc = reconstruct_radial_adjoint_with_motion(
                    obj=obj,
                    n_spokes=n,
                    n_samples=args.n_samples,
                    backend=args.backend,
                    density=args.density,
                    noise_sigma=sigma,
                    seed=args.seed + run_id,
                    motion_mode=args.motion_mode,
                    motion_shift_px=args.motion_shift_px,
                    motion_axis=args.motion_axis,
                    motion_start_frac=args.motion_start_frac,
                )
            run_id += 1

            if radial_raw is not None:
                radial_scaled, radial_scale = rescale_to_gt_percentile(
                    radial_raw,
                    gt,
                    percentile=99.0,
                )

                radial_metrics = compute_metrics(
                    gt=gt,
                    recon=radial_scaled,
                    mask_threshold=args.mask_threshold,
                )

                radial_row = {
                    "method": "radial_adjoint",
                    "n_spokes": int(n),
                    "n_cart_lines": "",
                    "n_samples_per_readout": int(args.n_samples),
                    "n_total_samples": int(n * args.n_samples),
                    "noise_sigma": float(sigma),
                    "scale": float(radial_scale),
                    "cart_mask_mode": "",
                    "cart_mask_axis": "",
                    **radial_metrics,
                    "recon_scaled": radial_scaled,
                }
                rows.append(radial_row)

            # ----------------------------
            # Cartesian undersampling
            # ----------------------------
            cart_raw, k_cart_us, cart_mask = reconstruct_cartesian_zerofill_with_motion(
                obj=obj,
                n_lines=n,
                mask_axis=args.cart_mask_axis,
                mask_mode=args.cart_mask_mode,
                noise_sigma=sigma,
                seed=args.seed + run_id,
                motion_mode=args.motion_mode,
                motion_shift_px=args.motion_shift_px,
                motion_axis=args.motion_axis,
                motion_start_frac=args.motion_start_frac,
            )
            run_id += 1

            cart_scaled, cart_scale = rescale_to_gt_percentile(
                cart_raw,
                gt,
                percentile=99.0,
            )

            cart_metrics = compute_metrics(
                gt=gt,
                recon=cart_scaled,
                mask_threshold=args.mask_threshold,
            )

            cart_row = {
                "method": "cartesian_zerofill",
                "n_spokes": int(n),
                "n_cart_lines": int(n),
                "n_samples_per_readout": int(args.n_samples),
                "n_total_samples": int(n * args.n_samples),
                "noise_sigma": float(sigma),
                "scale": float(cart_scale),
                "cart_mask_mode": args.cart_mask_mode,
                "cart_mask_axis": int(args.cart_mask_axis),
                **cart_metrics,
                "recon_scaled": cart_scaled,
            }
            rows.append(cart_row)

            if args.save_npy:
                tag = f"sigma{sigma:g}_n{n:04d}".replace(".", "p")
                if radial_raw is not None:
                    np.save(args.out_dir / f"radial_scaled_{tag}.npy", radial_scaled)
                np.save(args.out_dir / f"cartesian_scaled_{tag}.npy", cart_scaled)

    # Save CSV
    metrics_csv = args.out_dir / "sampling_noise_cartesian_comparison_metrics.csv"
    write_metrics_csv(metrics_csv, rows)

    # Save panels per sigma
    for sigma in args.noise_sigma_list:
        panel_png = args.out_dir / f"sampling_comparison_panel_sigma{sigma:g}.png".replace(".", "p")
        save_sampling_comparison_panel(
            out_png=panel_png,
            gt=gt,
            cart_full=cart_full,
            rows=rows,
            noise_sigma=sigma,
            n_list=args.n_list,
            title=(
                f"Radial vs Cartesian undersampling | "
                f"axis={args.axis}, index={used_index}, sigma={sigma}"
            ),
        )

        metric_png = args.out_dir / f"sampling_comparison_metrics_sigma{sigma:g}.png".replace(".", "p")
        save_sampling_metric_plot(
            out_png=metric_png,
            rows=rows,
            noise_sigma=sigma,
        )

    # Save noise sweep plot at the largest sampling budget by default.
    fixed_n = max(args.n_list)
    noise_plot_png = args.out_dir / f"noise_sigma_sweep_metrics_n{fixed_n}.png"
    save_noise_sweep_metric_plot(
        out_png=noise_plot_png,
        rows=rows,
        fixed_n_spokes=fixed_n,
    )

    # Save GT/reference
    if args.save_npy:
        np.save(args.out_dir / "gt_slice.npy", gt)
        np.save(args.out_dir / "cartesian_full_reference.npy", cart_full)

    summary_txt = args.out_dir / "summary.txt"
    with open(summary_txt, "w") as f:
        f.write("Sampling + noise + Cartesian comparison\n")
        f.write(f"nifti: {args.nifti}\n")
        f.write(f"axis: {args.axis}\n")
        f.write(f"index: {used_index}\n")
        f.write(f"matrix: {args.matrix}\n")
        f.write(f"n_samples: {args.n_samples}\n")
        f.write(f"n_list: {args.n_list}\n")
        f.write(f"noise_sigma_list: {args.noise_sigma_list}\n")
        f.write(f"backend: {args.backend}\n")
        f.write(f"density: {args.density}\n")
        f.write(f"cart_mask_axis: {args.cart_mask_axis}\n")
        f.write(f"cart_mask_mode: {args.cart_mask_mode}\n")
        f.write(f"seed: {args.seed}\n")
        f.write("\nOutputs:\n")
        f.write(f"metrics_csv: {metrics_csv}\n")
        f.write(f"noise_plot_png: {noise_plot_png}\n")
        f.write(f"motion_mode: {args.motion_mode}\n")
        f.write(f"motion_shift_px: {args.motion_shift_px}\n")
        f.write(f"motion_axis: {args.motion_axis}\n")
        f.write(f"motion_start_frac: {args.motion_start_frac}\n")

    print("\nSaved:")
    print(f"  {metrics_csv}")
    print(f"  {noise_plot_png}")
    print(f"  {summary_txt}")
    print("\nPanels per sigma were saved in:")
    print(f"  {args.out_dir}")


if __name__ == "__main__":
    main()

