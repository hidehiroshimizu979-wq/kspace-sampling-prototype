#!/usr/bin/env bash
# Example commands for motion shift sweep summary

BASE_DIR=/path/to/motion_sweep_base
OUT_DIR=results/motion_shift_sweep
mkdir -p ${OUT_DIR}

python scripts/summarize_motion_shift_sweep_v1.py \
  --base-dir ${BASE_DIR} \
  --out-dir ${OUT_DIR} \
  --shifts 0 2 5 10 \
  --sigma 0.02 \
  --n 128
