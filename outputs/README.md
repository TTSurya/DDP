# Generated outputs

`bsd500_deblurring/` contains the completed BSD500 sparse-blur experiment:

- `best_model.pth`: checkpoint with the highest validation PSNR.
- `test_metrics.json`: held-out test metrics, baseline comparison, and experiment configuration.
- `training_history.json`: per-epoch training and validation history.

The current test scores are 27.0024 dB PSNR / 0.72535 SSIM for the restored model and 21.9206 dB / 0.48781 SSIM for the adjoint/backprojection baseline.