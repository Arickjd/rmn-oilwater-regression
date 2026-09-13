"""PINN com log(T2), tempo normalizado e scheduler (célula 18)."""

import torch

from .common import PINNBase


class PINNLogT2(PINNBase):
    use_scheduler = True

    def decode(self, x):
        raw = self.network(x)
        amplitudes = torch.sigmoid(raw[:, :2])
        t2 = torch.exp(raw[:, 2:].clamp(-10, 5))
        time = torch.arange(x.shape[1], device=x.device, dtype=x.dtype) / x.shape[1]
        reconstructed = (
            amplitudes[:, None, :] * torch.exp(-time[None, :, None] / t2[:, None, :])
        ).sum(dim=-1)
        saturation = amplitudes[:, 0] / (amplitudes.sum(dim=1) + 1e-8)
        return saturation, reconstructed

