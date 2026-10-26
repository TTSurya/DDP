import torch
import torch.nn as nn
import pywt
from pytorch_wavelets import DWTForward, DWTInverse


class proxlayer(nn.Module):
    """Wavelet soft-thresholding block used as the learned image prior."""

    def __init__(self, prox_in, wavefamily="db4", levels=3):
        super().__init__()
        # One learned threshold tensor per wavelet decomposition scale.
        self.lamda1 = nn.Parameter(prox_in[0] * torch.ones((1, 3, 3, 32, 32), dtype=torch.float32))
        self.lamda2 = nn.Parameter(prox_in[1] * torch.ones((1, 3, 3, 16, 16), dtype=torch.float32))
        self.lamda3 = nn.Parameter(prox_in[2] * torch.ones((1, 3, 3, 8, 8), dtype=torch.float32))
        self.softplus = nn.Softplus(beta=20, threshold=20)
        self.xfwt = DWTForward(J=levels, wave=pywt.Wavelet(wavefamily), mode="periodization")
        self.xinvfwt = DWTInverse(wave=pywt.Wavelet(wavefamily), mode="periodization")
        self.thresh = nn.ReLU()

    def forward(self, x, y=None):
        # Keep thresholds positive, then soft-threshold each high-frequency band.
        lam1 = 1e-6 + self.softplus(self.lamda1)
        lam2 = 1e-6 + self.softplus(self.lamda2)
        lam3 = 1e-6 + self.softplus(self.lamda3)
        low, high = self.xfwt(x)
        high_new = [
            self.thresh(high[0] - lam1) - self.thresh(-high[0] - lam1),
            self.thresh(high[1] - lam2) - self.thresh(-high[1] - lam2),
            self.thresh(high[2] - lam3) - self.thresh(-high[2] - lam3),
        ]
        return self.xinvfwt((low, high_new))
