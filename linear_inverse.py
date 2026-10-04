"""Contractive wavelet layers for general linear inverse problems y = A x + n.

The supplied operator must implement ``forward(x)`` and ``adjoint(y)``.
Provide certified (or conservatively bounded) extreme singular values when
constructing the model. All tensors use NCHW layout; operators may change the
spatial dimensions.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import proxlayer


def dense_singular_values(A: torch.Tensor) -> tuple[float, float]:
    """Return exact extreme singular values for a dense 2-D matrix A."""
    if A.ndim != 2:
        raise ValueError("A must be a 2-D matrix")
    values = torch.linalg.svdvals(A)
    sigma_max = float(values.max())
    # A maps R^n to R^m. When m<n, A^T A necessarily has a nullspace,
    # although svdvals only returns min(m,n) singular values.
    if A.shape[0] < A.shape[1]:
        return sigma_max, 0.0
    sigma_min = float(values.min())
    tolerance = max(A.shape) * torch.finfo(A.dtype).eps * sigma_max
    if sigma_min <= tolerance:
        sigma_min = 0.0
    return sigma_max, sigma_min


@torch.no_grad()
def estimate_sigma_max(operator, input_shape, *, iterations=100, device="cpu"):
    """Power-iteration estimate of ||A||₂ using only forward/adjoint calls.

    This is useful for large implicit operators, but is an estimate rather than
    a certificate. Use a safe upper bound for a guaranteed step-size interval.
    """
    x = torch.randn(input_shape, device=device)
    x = x / (torch.linalg.vector_norm(x) + 1e-12)
    for _ in range(iterations):
        z = operator.adjoint(operator.forward(x))
        x = z / (torch.linalg.vector_norm(z) + 1e-12)
    return float(torch.linalg.vector_norm(operator.forward(x)))


def gradient_step_lipschitz(sigma_max: float, sigma_min: float,
                            step_size: float) -> float:
    """Exact ||I-step*A^T A|| bound from extreme singular values."""
    if sigma_max < 0 or sigma_min < 0 or sigma_min > sigma_max:
        raise ValueError("Require 0 <= sigma_min <= sigma_max")
    if step_size < 0:
        raise ValueError("step_size must be nonnegative")
    return max(abs(1.0 - step_size * sigma_min**2),
               abs(1.0 - step_size * sigma_max**2))


class GeneralLinearInverseLayer(nn.Module):
    """One learned gradient + wavelet-prox + contractive-convolution layer."""

    def __init__(self, image_shape: tuple[int, int], sigma_max: float,
                 sigma_min: float = 0.0, prox_in: Iterable[float] =
                 (0.001, 0.001, 0.001), wave="db4", levels=3,
                 channels=3, epsilon=1e-4):
        super().__init__()
        if tuple(image_shape) != (64, 64) or levels != 3 or channels != 3:
            raise NotImplementedError(
                "The inherited wavelet-threshold parameterization currently "
                "supports 64x64 RGB patches with 3 decomposition levels."
            )
        if sigma_max <= 0:
            raise ValueError("sigma_max must be positive")
        if not 0 <= sigma_min <= sigma_max:
            raise ValueError("Require 0 <= sigma_min <= sigma_max")
        self.image_shape = tuple(image_shape)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.epsilon = float(epsilon)
        self.prox = proxlayer(list(prox_in), wave, levels)

        # sigmoid parameterization keeps 0 < step < 2/sigma_max^2.
        step_limit = 2.0 / (self.sigma_max**2)
        initial_step = 0.5 * step_limit
        self.raw_step = nn.Parameter(torch.tensor(math.log(initial_step /
                                                            (step_limit-initial_step))))
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        # Start close to an identity refinement; normalize_convolution still
        # enforces the certified spectral bound on every forward pass.
        with torch.no_grad():
            self.conv.weight.zero_()
            for channel in range(channels):
                self.conv.weight[channel, channel, 1, 1] = 1.0

    @property
    def step_size(self):
        return (2.0 / self.sigma_max**2) * torch.sigmoid(self.raw_step)

    def gradient_bound(self) -> torch.Tensor:
        step = self.step_size
        lo = torch.abs(1.0 - step * self.sigma_min**2)
        hi = torch.abs(1.0 - step * self.sigma_max**2)
        return torch.maximum(lo, hi)

    def _normalized_convolution(self, x: torch.Tensor) -> torch.Tensor:
        # Exact Fourier norm for circular convolution on this patch grid.
        h, w = self.image_shape
        fft = torch.fft.fft2(self.conv.weight, s=(h, w))
        # [out,in,h,w] -> [h,w,out,in]
        frequency_matrices = fft.permute(2, 3, 0, 1)
        norm = torch.linalg.matrix_norm(frequency_matrices, ord=2).amax()
        x = F.pad(x, (1, 1, 1, 1), mode="circular")
        return F.conv2d(x, self.conv.weight / (norm + 1e-12), bias=None)

    def forward(self, x: torch.Tensor, y: torch.Tensor, operator) -> torch.Tensor:
        # G_gamma(x) = x - gamma A^T(Ax-y).
        grad = operator.adjoint(operator.forward(x) - y)
        z = x - self.step_size * grad
        p = self.prox(z, y)
        q = self.gradient_bound()
        # ||outer|| <= 1/(q+eps), so ||layer|| <= q/(q+eps) < 1.
        return self._normalized_convolution(p) / (q + self.epsilon)


class ContractiveInverseNet(nn.Module):
    """Composition of contractive learned layers for a fixed linear operator."""

    def __init__(self, image_shape, sigma_max, sigma_min=0.0, depth=10,
                 prox_in=(0.001, 0.001, 0.001), wavefamilyset=("haar", "db4", "sym4"),
                 levels=3, channels=3, epsilon=1e-4):
        super().__init__()
        self.layers = nn.ModuleList([
            GeneralLinearInverseLayer(
                image_shape, sigma_max, sigma_min, prox_in,
                wavefamilyset[i % len(wavefamilyset)], levels, channels, epsilon
            ) for i in range(depth)
        ])

    def forward(self, x, y, operator):
        for layer in self.layers:
            x = layer(x, y, operator)
        return x


def train_inverse_model(model: nn.Module, operator, clean_loader, *, epochs=50,
                        learning_rate=1e-4, noise_std=5/255, device="cpu",
                        checkpoint_path=None):
    """Supervised training on clean images with synthetic y=A x+n observations.

    ``clean_loader`` must yield clean NCHW image batches in [0,1]. The operator
    supplies forward/adjoint and its singular-value bounds when constructing
    the model. Training optimizes reconstruction MSE end-to-end.
    """
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        total = 0.0
        count = 0
        for clean in clean_loader:
            clean = clean.to(device)
            noise = torch.randn_like(operator.forward(clean)) * noise_std
            measurement = operator.forward(clean) + noise
            # Backprojection gives a valid image-domain initialization for
            # operators whose adjoint returns the image shape.
            initial = operator.adjoint(measurement)
            estimate = model(initial, measurement, operator)
            loss = F.mse_loss(estimate, clean)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.detach().item()
            count += 1
        if (epoch + 1) in {max(1, epochs // 3), max(1, 2 * epochs // 3)}:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
        print(f"epoch {epoch + 1:03d}/{epochs}: mse={total / max(count, 1):.7g}")
    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)
    return model
