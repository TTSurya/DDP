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

On Windows, run `setup.bat`. It creates/reuses a project-local `.venv`, installs the requirements, prepares BSD500, trains/evaluates the model, and saves the checkpoint and metrics under `outputs/bsd500_deblurring/`. Reduce CPU runtime with `--epochs`, `--samples-per-epoch`, `--depth`, and `--batch-size`. Change the blur with `--blur-kernel-size` and `--blur-sparsity`.

## Files

- `linear_inverse.py`: contractive reconstruction layers, singular-value bounds, and a general training helper for operators implementing `forward(x)` and `adjoint(y)`.
- `model.py`: the wavelet soft-thresholding proximal block used by those layers.
- `run_bsd500_restoration.py`: the sparse circular blur operator plus BSD500 training, validation, and test evaluation.
- `prepare_bsds500.py`: downloads and extracts the official 200/100/200 BSD500 train/validation/test split.
- `requirements.txt`: dependencies for the linear inverse workflow.

The current wavelet-threshold parameterization supports 64x64 RGB patches with three decomposition levels. For a guaranteed step-size interval, provide a certified or conservative upper bound for the operator's largest singular value.
