# Contractive Linear Inverse Restoration

This repository trains a wavelet-based neural network to reconstruct clean RGB image patches from linear measurements:

\[
y = A x + n,
\]

where `A` is a known linear measurement operator and `n` is additive Gaussian noise.

## Setup and run

Install the dependencies, download/extract BSD500, then train and evaluate the deblurring model:

```bash
python -m pip install -r requirements.txt
python prepare_bsds500.py
python run_bsd500_restoration.py
```

On Windows, run `setup.bat`. It creates/reuses a project-local `.venv`, installs the requirements, prepares BSD500, trains/evaluates the model, and saves outputs under `outputs/bsd500_deblurring/`.

Training uses 64x64 RGB patches, a seeded periodic sparse blur (default 15x15, 90% sparsity), noise standard deviation `5/255`, 30 layers, and 50 epochs. Reduce CPU runtime with `--epochs`, `--samples-per-epoch`, `--depth`, or `--batch-size`. Change the blur with `--blur-kernel-size` and `--blur-sparsity`.

## Latest BSD500 result

The completed run used 200 training images, 100 validation images, and 200 held-out test patches. On the test set:

| Method | PSNR | SSIM |
|---|---:|---:|
| Adjoint/backprojection | 21.92 dB | 0.4878 |
| Restored model | **27.00 dB** | **0.7253** |

The model improved the backprojection baseline by about 5.08 dB PSNR and 0.2375 SSIM. Full-precision metrics, training history, and the checkpoint are saved in `outputs/bsd500_deblurring/`.

An earlier Gaussian-blur pilot scored 25.61 dB / 0.6459 SSIM. The current result is higher, but the blur settings and training schedule differ, so this is not a controlled version-to-version comparison.

## Files

- `linear_inverse.py`: contractive reconstruction layers, singular-value bounds, and a general training helper for operators implementing `forward(x)` and `adjoint(y)`.
- `model.py`: the wavelet soft-thresholding proximal block used by those layers.
- `run_bsd500_restoration.py`: the sparse circular blur operator plus BSD500 training, validation, and test evaluation.
- `prepare_bsds500.py`: downloads and extracts the official 200/100/200 BSD500 train/validation/test split.
- `requirements.txt`: dependencies for the linear inverse workflow.
- `setup.bat`: Windows setup, dataset preparation, training, and evaluation.

The current wavelet-threshold parameterization supports 64x64 RGB patches with three decomposition levels. For a guaranteed step-size interval, provide a certified or conservative upper bound for the operator's largest singular value.