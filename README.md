📌 Overview

This repository provides the implementation used to reproduce the denoising, deblurring, and super-resolution results presented in our paper.

📁 Repository Structure
.

├── denoising_test.py        # Denoising script

├── pnp_deblurring.py        # Deblurring script

├── pnp_superresolution.py      # Super-resolution script

├── model.py                # Model architecture

├── utils.py                # Utility functions

│

├── images/                 # Input images

├── models/                 # Pretrained model weights (.pth)

└── outputs/                # Saved results

⚙️ Installation

All required Python dependencies are listed in requirements.txt.

pip install --upgrade pip

pip install -r requirements.txt

🚀 Usage

🔹 Denoising

python denoising_test.py

🔹 Deblurring

python pnp_deblurring.py

🔹 Super-Resolution

python pnp_superresolution.py

📦 Outputs

All results are automatically saved in the outputs/ directory.

## General linear inverse problems

`linear_inverse.py` adds a separate, operator-driven path for measurements
`y = A x + n`. It uses the same wavelet proximal block, with the data step
replaced by `x - gamma * A.adjoint(A.forward(x) - y)`. The operator object must
provide `forward(x)` and `adjoint(y)` methods and return image-shaped output
from its adjoint.

For `f(x) = 0.5 * ||A x - y||^2`, the data-step Lipschitz factor is

```
q(gamma) = max(|1 - gamma * sigma_min^2|,
               |1 - gamma * sigma_max^2|).
```

Therefore the nonexpansive step-size interval is `0 < gamma <
2 / sigma_max^2` (the singular value is squared). If A has full column rank,
the gradient step is itself strictly contractive in that interval. If A is
rank-deficient, it is only nonexpansive; the added layer remains contractive
because its normalized outer convolution is scaled to at most `1/(q+epsilon)`.
The full layer bound is then `q/(q+epsilon) < 1`. For full rank, the gradient
step's best-conditioned choice is near `2/(sigma_min^2 + sigma_max^2)`.

Use `dense_singular_values(A)` for exact extrema of an explicitly represented
matrix. `estimate_sigma_max(operator, input_shape)` supplies a power-iteration
estimate for an implicit operator; it is not a certified upper bound, so use a
conservative bound when a formal guarantee is required. The model constructor
accepts `sigma_max` and `sigma_min`, and `train_inverse_model(...)` trains it
end-to-end from a loader of clean NCHW image batches by synthesizing `A x + n`.
The data/operator implementation and training images are problem-specific and
must be supplied by the caller. The operator must use matching inner products
so its `adjoint` is the true transpose, and the forward model must be fixed
during a training run. The inherited wavelet-threshold parameterization is
currently fixed to 64x64 RGB patches with three decomposition levels, matching
the existing project model.

### BSD500 deblurring experiment

The experiment sets A to a fixed periodic sparse-kernel blur (default 15x15,
90% sparsity), with the exact adjoint and singular values computed from its
Fourier response. This models the forward operator in a deblurring problem;
the reconstruction estimates x from y = A x + n. AWGN has standard deviation
`5/255`. It samples and corrupts 64x64 patches on-the-fly: the 200 BSD500
train images are used for optimization, 100 validation images for checkpoint
selection, and all 200 held-out test images for final evaluation. It reports
PSNR and SSIM for the adjoint/backprojection baseline and restored patches.
The archive and generated weights/metrics stay under the ignored `data/` and
`outputs/` folders.

```
python prepare_bsds500.py
python run_bsd500_restoration.py
```

Training defaults follow the paper's 30-layer, 50-epoch setup. On a CPU-only
machine, use `--depth`, `--epochs`, `--samples-per-epoch`, and `--batch-size`
to reduce the run time. Adjust the blur with `--blur-kernel-size` and
`--blur-sparsity`. The selected checkpoint is saved as
`outputs/bsd500_deblurring/best_model.pth`; final held-out metrics are written
to `test_metrics.json` in the same folder. The 64x64 patch evaluation
matches the model's training boundary convention; it is a patch restoration
benchmark rather than a full-resolution tiled-image benchmark.

### Earlier Gaussian-blur pilot

The included pilot predates the deblurring operator change and used Gaussian
blur, 30 layers, 5 epochs, and 64 generated patches per epoch.
On one deterministic 64x64 crop from each of the 200 held-out test images, the
adjoint baseline scored 24.53 dB / 0.6034 SSIM and the model scored 25.61 dB /
0.6459 SSIM. The certified product bound from the constrained layers was
0.9970. This is an initial pilot, not the full 50-epoch paper schedule. Its
checkpoint and full-precision metrics are in
`outputs/bsd500_gaussian_blur_30layer_pilot/`.
