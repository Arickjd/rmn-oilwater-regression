"""PINN original com T2 via softplus e tempo em bins (célula 17)."""

import torch
from torch.nn import functional as F

from .common import PINNBase


class PINN(PINNBase):
    def decode(self, x):
        raw = self.network(x)
        amplitudes = torch.sigmoid(raw[:, :2])
        t2 = F.softplus(raw[:, 2:]).clamp_min(1e-3)
        time = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
        reconstructed = (
            amplitudes[:, None, :] * torch.exp(-time[None, :, None] / t2[:, None, :])
        ).sum(dim=-1)
        saturation = amplitudes[:, 0] / (amplitudes.sum(dim=1) + 1e-8)
        return saturation, reconstructed

