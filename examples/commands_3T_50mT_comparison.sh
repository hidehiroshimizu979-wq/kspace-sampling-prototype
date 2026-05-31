#!/usr/bin/env bash
# Example commands for combined 3T vs 50mT metrics

# Adjust paths to your local data/ and outputs/ folders
OUT_DIR=results/3T_50mT_comparison
mkdir -p ${OUT_DIR}

python scripts/summarize_3T_50mT_sampling_metrics_v1.py \
  --condition 3T_like:/path/to/3T/sampling_noise_cartesian_comparison_metrics.csv \
  --condition 50mT_like:/path/to/50mT/sampling_noise_cartesian_comparison_metrics.csv \
  --out-dir ${OUT_DIR} \
  --n 128
