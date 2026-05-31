I made a small prototype repository for image-derived k-space sampling simulation. The current version uses a 3T BRAVO anatomical image as a ground-truth object and also creates a contrast-remapped 50mT-like anatomical target. It compares radial adjoint and center-weighted Cartesian zero-filled reconstructions under matched sampling budgets, synthetic k-space noise, and a simple step-motion model.

The main purpose is not to claim a realistic 50mT acquisition simulation yet, but to build a working framework for testing sampling trajectories and motion/noise effects. The current results show expected behavior: increasing radial spokes or Cartesian lines improves reconstruction, k-space noise reduces SSIM, and increasing step-motion amplitude degrades both methods. The 50mT-like target should be interpreted as a contrast-remapped approximation, not a quantitative low-field image.

Next steps:
- Reduce hard-coded local paths
- Add example commands
- Prepare a data bundle shared separately (do not commit raw data to GitHub)
