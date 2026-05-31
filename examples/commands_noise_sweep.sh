#!/usr/bin/env bash
# Example commands for running sampling/noise sweep (single condition run example)

OUT_DIR=results/noise_sweep
mkdir -p ${OUT_DIR}

python scripts/run_sampling_noise_cartesian_comparison_v1.py \
  --input-nifti /path/to/data/subj25973/anat/subj25973_SAG_FSPGR_BRAVO.nii.gz \
  --out-dir ${OUT_DIR} \
  --n_spokes 128 \
  --noise_sigma 0.02
