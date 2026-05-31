#!/usr/bin/env python3
"""
Summarize 3T-like vs 50mT-like sampling/noise comparison metrics.

Inputs:
  --condition NAME:CSV_PATH

Example:
  --condition 3T_like:/path/to/3T/sampling_noise_cartesian_comparison_metrics.csv
  --condition 50mT_like:/path/to/50mT/sampling_noise_cartesian_comparison_metrics.csv

Outputs:
  - combined_metrics.csv
  - noise_sweep_3T_vs_50mT_metrics.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_condition_arg(arg: str) -> tuple[str, Path]:
    if ":" not in arg:
        raise ValueError(
            f"Invalid --condition argument: {arg}\n"
            "Expected format: NAME:/path/to/metrics.csv"
        )
    name, path = arg.split(":", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        help="Condition specification as NAME:/path/to/metrics.csv",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--methods", nargs="+", default=["radial_adjoint", "cartesian_zerofill"])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []

    for cond_arg in args.condition:
        condition_name, csv_path = parse_condition_arg(cond_arg)

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for {condition_name}: {csv_path}")

        df = pd.read_csv(csv_path)
        df["condition"] = condition_name
        df["source_csv"] = str(csv_path)

        # Keep target n only.
        df = df[df["n_spokes"].astype(int) == int(args.n)].copy()

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No input metrics found.")

    combined = pd.concat(dfs, ignore_index=True)

    # Make numeric robust.
    for col in ["noise_sigma", "nrmse_range", "ssim", "corr", "rmse", "mae"]:
        if col in combined.columns:
            combined[col] = combined[col].astype(float)

    out_csv = args.out_dir / "combined_3T_50mT_sampling_metrics.csv"
    combined.to_csv(out_csv, index=False)

    # Long-format plotting.
    metrics = [
        ("nrmse_range", "NRMSE"),
        ("ssim", "SSIM"),
        ("corr", "Correlation"),
    ]

    method_labels = {
        "radial_adjoint": "Radial adjoint",
        "cartesian_zerofill": "Cartesian zero-fill",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax, (metric_col, ylabel) in zip(axes, metrics):
        for condition in combined["condition"].unique():
            for method in args.methods:
                d = combined[
                    (combined["condition"] == condition)
                    & (combined["method"] == method)
                ].copy()

                if d.empty:
                    continue

                d = d.sort_values("noise_sigma")

                label = f"{condition} / {method_labels.get(method, method)}"

                ax.plot(
                    d["noise_sigma"],
                    d[metric_col],
                    marker="o",
                    label=label,
                )

        ax.set_xlabel("k-space noise sigma")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=8)
    fig.suptitle(f"3T-like vs 50mT-like sampling comparison | n={args.n}")
    plt.tight_layout()

    out_png = args.out_dir / "noise_sweep_3T_vs_50mT_metrics.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Also save a compact pivot table.
    pivot_rows = []
    for _, row in combined.iterrows():
        pivot_rows.append(
            {
                "condition": row["condition"],
                "method": row["method"],
                "n": int(row["n_spokes"]),
                "noise_sigma": row["noise_sigma"],
                "nrmse_range": row["nrmse_range"],
                "ssim": row["ssim"],
                "corr": row["corr"],
                "rmse": row.get("rmse", ""),
                "mae": row.get("mae", ""),
            }
        )

    pivot = pd.DataFrame(pivot_rows)
    pivot_csv = args.out_dir / "combined_3T_50mT_sampling_metrics_compact.csv"
    pivot.to_csv(pivot_csv, index=False)

    print("Saved:")
    print(f"  {out_csv}")
    print(f"  {pivot_csv}")
    print(f"  {out_png}")


if __name__ == "__main__":
    main()
