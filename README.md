<<<<<<< HEAD
k-space sampling prototype

This repository contains a prototype image-derived k-space sampling simulation for comparing radial and Cartesian undersampling under synthetic noise and motion conditions.

The implementation uses 2D slices extracted from anatomical NIfTI images and supports:
- radial NUFFT adjoint reconstruction
- Cartesian zero-filled reconstruction
- k-space noise sweep
- step-motion simulation during acquisition
- contrast-remapped 50mT-like anatomical target generation

Important: DO NOT include raw DICOM or subject NIfTI data in this repository. Place input NIfTI files under a local `data/` folder as described in README_data.md.

Quick start (example):

1) Place your data:
   kspace-sampling-prototype/
     data/
       subj25973/
         anat/
         labels/
         gt_images/

2) Install dependencies:
   python -m pip install -r requirements.txt

3) Run an example (adjust paths):
   python scripts/run_sampling_noise_cartesian_comparison_v1.py --help

Limitations
- This is not a scanner-accurate simulation (no coil sens, B0/B1, relaxation, etc.)
- Single-coil, 2D-slice based, adjoint NUFFT only for radial

See `docs/summary_for_jacob.md` for a brief summary intended for collaborators.

How to run (examples)
----------------------

1) Place input data under a `data/` directory as described in `README_data.md`.

2) Install dependencies:

   python -m pip install -r requirements.txt

3) Run a noise / sampling sweep (adjust paths):

   python scripts/run_sampling_noise_cartesian_comparison_v1.py \
     --nifti /path/to/data/subj25973/anat/subj25973_SAG_FSPGR_BRAVO.nii.gz \
     --out-dir results/example_noise_sweep \
     --n-list 32 64 128 \
     --noise-sigma-list 0.0 0.01 0.05 \
     --n-samples 256 \
     --matrix 256

4) Summarize 3T vs 50mT conditions (combine two metrics CSVs):

   python scripts/summarize_3T_50mT_sampling_metrics_v1.py \
     --condition 3T_like:/path/to/3T/sampling_noise_cartesian_comparison_metrics.csv \
     --condition 50mT_like:/path/to/50mT/sampling_noise_cartesian_comparison_metrics.csv \
     --out-dir results/combined --n 128

5) Motion-shift sweep summary (collect per-shift CSVs):

   python scripts/summarize_motion_shift_sweep_v1.py \
     --base-dir /path/to/motion_sweep_base \
     --out-dir results/motion_shift_summary \
     --shifts 0 2 5 10 \
     --sigma 0.02 --n 128

Notes
- The scripts expect numeric arguments (matrix size, number of samples) and paths to NIfTI files. Edit the example commands to match your local layout.
- Do NOT place raw DICOM or subject-identifiable NIfTI files into this git repository. Use a separate data bundle and point the scripts at a local `data/` folder.
=======
# kspace-sampling-prototype
>>>>>>> origin/main
