#!/usr/bin/env python3
"""
Summarize motion-shift sweep results.

Expected directory structure:

BASE/
  shift_0px/
    sampling_noise_cartesian_comparison_metrics.csv
  shift_2px/
    sampling_noise_cartesian_comparison_metrics.csv
  shift_5px/
    sampling_noise_cartesian_comparison_metrics.csv
  shift_10px/
    sampling_noise_cartesian_comparison_metrics.csv

This script collects the metrics for a fixed:
  - noise_sigma
  - n spokes / Cartesian lines

and creates:
  - motion_shift_sweep_summary.csv
  - motion_shift_sweep_metrics.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def shift_to_dirname(shift: float) -> str:
    if float(shift).is_integer():
        return f"shift_{int(shift)}px"
    return f"shift_{shift:g}px"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--shifts", nargs="+", type=float, required=True)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument(
        "--metrics-csv-name",
        default="sampling_noise_cartesian_comparison_metrics.csv",
    )

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    collected = []

    for shift in args.shifts:
        shift_dir = args.base_dir / shift_to_dirname(shift)
        csv_path = shift_dir / args.metrics_csv_name

        if not csv_path.exists():
            print(f"[WARN] Missing CSV: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # Make numeric columns robust.
        df["noise_sigma"] = df["noise_sigma"].astype(float)
        df["n_spokes"] = df["n_spokes"].astype(int)

        sub = df[
            (df["noise_sigma"] == float(args.sigma))
            & (df["n_spokes"] == int(args.n))
        ].copy()

        if sub.empty:
            print(
                f"[WARN] No matching rows in {csv_path} "
                f"for sigma={args.sigma}, n={args.n}"
            )
            continue

        sub["motion_shift_px"] = float(shift)
        sub["source_csv"] = str(csv_path)
        collected.append(sub)

    if not collected:
        raise RuntimeError("No matching rows found. Check --base-dir, --shifts, --sigma, and --n.")

    out = pd.concat(collected, ignore_index=True)

    out_csv = args.out_dir / "motion_shift_sweep_summary.csv"
    out.to_csv(out_csv, index=False)

    # Plot metrics.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    metrics = [
        ("nrmse_range", "NRMSE"),
        ("ssim", "SSIM"),
        ("corr", "Correlation"),
    ]

    method_order = [
        ("radial_adjoint", "Radial adjoint"),
        ("cartesian_zerofill", "Cartesian zero-fill"),
    ]

    for ax, (metric_col, ylabel) in zip(axes, metrics):
        for method_key, method_label in method_order:
            d = out[out["method"] == method_key].copy()

            if d.empty:
                continue

            d = d.sort_values("motion_shift_px")

            ax.plot(
                d["motion_shift_px"],
                d[metric_col],
                marker="o",
                label=method_label,
            )

        ax.set_xlabel("Step motion shift [pixels]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"Motion shift sweep | n={args.n}, noise sigma={args.sigma}")
    plt.tight_layout()

    out_png = args.out_dir / "motion_shift_sweep_metrics.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Also save one-method-per-metric CSV pivot for easier reading.
    pivot_csv = args.out_dir / "motion_shift_sweep_pivot.csv"

    pivot_rows = []
    for method_key, method_label in method_order:
        d = out[out["method"] == method_key].sort_values("motion_shift_px")
        for _, row in d.iterrows():
            pivot_rows.append(
                {
                    "method": method_label,
                    "motion_shift_px": row["motion_shift_px"],
                    "n": args.n,
                    "noise_sigma": args.sigma,
                    "nrmse_range": row["nrmse_range"],
                    "ssim": row["ssim"],
                    "corr": row["corr"],
                    "rmse": row["rmse"],
                    "mae": row["mae"],
                }
            )

    pd.DataFrame(pivot_rows).to_csv(pivot_csv, index=False)

    print("Saved:")
    print(f"  {out_csv}")
    print(f"  {pivot_csv}")
    print(f"  {out_png}")


if __name__ == "__main__":
    main()
