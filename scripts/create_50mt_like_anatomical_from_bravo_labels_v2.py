#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from nibabel.processing import resample_from_to


def parse_args():
    p = argparse.ArgumentParser(
        description="Create a 50mT-like anatomical image from 3T BRAVO + label map."
    )
    p.add_argument("--bravo", required=True, help="Input 3T BRAVO NIfTI")
    p.add_argument("--labels", required=True, help="Input label NIfTI (e.g. WM=1, GM=2, CSF=3 or similar)")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--axis", type=int, default=2, help="Display axis for QC panel")
    p.add_argument("--index", type=int, default=None, help="Slice index for QC panel (default: center)")
    p.add_argument("--wm-label", type=int, default=1, help="WM label value")
    p.add_argument("--gm-label", type=int, default=2, help="GM label value")
    p.add_argument("--csf-label", type=int, default=3, help="CSF label value")

    # 50mT-like target means
    p.add_argument("--target-wm-mean", type=float, default=0.58)
    p.add_argument("--target-gm-mean", type=float, default=0.54)
    p.add_argument("--target-csf-mean", type=float, default=0.22)

    p.add_argument(
        "--texture-strength",
        type=float,
        default=0.60,
        help="How much normalized local texture from BRAVO is retained inside each tissue"
    )
    p.add_argument(
        "--post-blur-sigma",
        type=float,
        default=0.8,
        help="Gaussian blur sigma [voxels] applied after remapping"
    )
    p.add_argument(
        "--clip-min",
        type=float,
        default=0.0,
        help="Lower intensity clip"
    )
    p.add_argument(
        "--clip-max",
        type=float,
        default=1.0,
        help="Upper intensity clip"
    )

    return p.parse_args()


def normalize01(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    return (x - mn) / (mx - mn + eps)


def robust_normalize01(x, low=1.0, high=99.0, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    p1 = np.percentile(x, low)
    p2 = np.percentile(x, high)
    y = np.clip((x - p1) / (p2 - p1 + eps), 0.0, 1.0)
    return y


def extract_slice(vol, axis, index):
    if axis == 0:
        return vol[index, :, :]
    elif axis == 1:
        return vol[:, index, :]
    elif axis == 2:
        return vol[:, :, index]
    else:
        raise ValueError("axis must be 0, 1, or 2")


def build_overlay(lbl2d, wm_label, gm_label, csf_label):
    overlay = np.zeros(lbl2d.shape + (3,), dtype=np.float32)

    # Requested: R=GM, G=WM, B=CSF
    overlay[..., 0] = (lbl2d == gm_label).astype(np.float32)   # red
    overlay[..., 1] = (lbl2d == wm_label).astype(np.float32)   # green
    overlay[..., 2] = (lbl2d == csf_label).astype(np.float32)  # blue

    return overlay


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bravo_nii = nib.load(args.bravo)
    bravo = bravo_nii.get_fdata().astype(np.float32)

    label_nii = nib.load(args.labels)

    if label_nii.shape != bravo_nii.shape:
        print(f"[INFO] Resampling labels to BRAVO grid: label {label_nii.shape} -> bravo {bravo_nii.shape}")
        label_nii = resample_from_to(
            label_nii,
            bravo_nii,
            order=0,  # nearest-neighbor for labels
        )

    labels = label_nii.get_fdata().astype(np.int16)

    # Normalize 3T BRAVO robustly
    bravo_norm = robust_normalize01(bravo, low=1.0, high=99.0)

    wm_mask = labels == args.wm_label
    gm_mask = labels == args.gm_label
    csf_mask = labels == args.csf_label
    brain_mask = wm_mask | gm_mask | csf_mask

    if not np.any(brain_mask):
        raise ValueError("No brain voxels found in labels with the provided label IDs.")

    remap = np.zeros_like(bravo_norm, dtype=np.float32)

    tissue_specs = [
        ("WM", wm_mask, args.target_wm_mean),
        ("GM", gm_mask, args.target_gm_mean),
        ("CSF", csf_mask, args.target_csf_mean),
    ]

    summary_lines = []
    summary_lines.append("=== 50mT-like remapping summary ===")
    summary_lines.append(f"BRAVO: {args.bravo}")
    summary_lines.append(f"Labels: {args.labels}")
    summary_lines.append(f"WM label={args.wm_label}, GM label={args.gm_label}, CSF label={args.csf_label}")
    summary_lines.append("")
    summary_lines.append(f"Target WM mean:  {args.target_wm_mean:.4f}")
    summary_lines.append(f"Target GM mean:  {args.target_gm_mean:.4f}")
    summary_lines.append(f"Target CSF mean: {args.target_csf_mean:.4f}")
    summary_lines.append(f"Texture strength: {args.texture_strength:.4f}")
    summary_lines.append(f"Post blur sigma:  {args.post_blur_sigma:.4f}")
    summary_lines.append("")

    # Tissue-wise remapping:
    # new_signal = target_mean + texture_strength * (local_intensity - local_mean)
    # Then clip.
    for tissue_name, mask, target_mean in tissue_specs:
        if np.any(mask):
            vals = bravo_norm[mask]
            src_mean = float(vals.mean())
            centered = vals - src_mean
            mapped = target_mean + args.texture_strength * centered
            remap[mask] = mapped

            summary_lines.append(
                f"{tissue_name}: n={mask.sum()}, source_mean={src_mean:.4f}, "
                f"mapped_mean(before blur)={mapped.mean():.4f}"
            )
        else:
            summary_lines.append(f"{tissue_name}: no voxels found")

    # Outside labeled brain keep dark background
    remap[~brain_mask] = bravo_norm[~brain_mask] * 0.15

    # Optional blur to reduce blocky boundaries and make it more low-field-like
    if args.post_blur_sigma > 0:
        remap_blur = gaussian_filter(remap, sigma=args.post_blur_sigma)
        # Re-impose background attenuation gently
        remap = remap_blur

    remap = np.clip(remap, args.clip_min, args.clip_max)

    # Save NIfTI
    out_nifti = out_dir / "subj25973_50mT_like_from_BRAVO_v2.nii.gz"
    nib.save(nib.Nifti1Image(remap.astype(np.float32), bravo_nii.affine, bravo_nii.header), str(out_nifti))

    # Decide QC slice
    axis = args.axis
    if args.index is None:
        index = bravo.shape[axis] // 2
    else:
        index = args.index

    bravo2d = extract_slice(bravo_norm, axis, index)
    remap2d = extract_slice(remap, axis, index)
    label2d = extract_slice(labels, axis, index)

    overlay = build_overlay(label2d, args.wm_label, args.gm_label, args.csf_label)

    # QC figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"axis={axis}, index={index}", fontsize=18)

    axs[0].imshow(bravo2d.T, cmap="gray", origin="lower", vmin=0, vmax=1)
    axs[0].set_title("3T BRAVO normalized", fontsize=16)
    axs[0].axis("off")

    axs[1].imshow(remap2d.T, cmap="gray", origin="lower", vmin=0, vmax=1)
    axs[1].set_title("50mT-like remapped", fontsize=16)
    axs[1].axis("off")

    axs[2].imshow(np.transpose(overlay, (1, 0, 2)), origin="lower")
    axs[2].set_title("Label overlay\nR=GM, G=WM, B=CSF", fontsize=16)
    axs[2].axis("off")

    fig.tight_layout()
    out_png = out_dir / f"qc_50mT_like_axis{axis}_idx{index}.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Post stats
    summary_lines.append("")
    summary_lines.append("=== Post-blur / final means ===")
    for tissue_name, mask, _ in tissue_specs:
        if np.any(mask):
            summary_lines.append(
                f"{tissue_name}: final_mean={remap[mask].mean():.4f}, "
                f"final_std={remap[mask].std():.4f}"
            )

    summary_lines.append("")
    summary_lines.append(f"Saved NIfTI: {out_nifti}")
    summary_lines.append(f"Saved QC PNG: {out_png}")

    out_txt = out_dir / "summary_50mT_like_v2.txt"
    out_txt.write_text("\n".join(summary_lines), encoding="utf-8")

    print("Saved:")
    print(f"  {out_nifti}")
    print(f"  {out_png}")
    print(f"  {out_txt}")


if __name__ == "__main__":
    main()
