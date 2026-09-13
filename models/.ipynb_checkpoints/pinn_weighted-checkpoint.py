"""PINN com Sw direto, T2 ordenados e perda temporal ponderada (célula 19)."""

import torch
from torch.nn import functional as F

from .common import PINNBase


class PINNWeighted(PINNBase):
    output_weight = 2.0
    physics_weight = 1.0
    use_scheduler = True
    time_decay = 4.0

    def decode(self, x):
        raw = self.network(x)
        saturation = torch.sigmoid(raw[:, 0])
        m0 = F.softplus(raw[:, 1])
        t2_short = torch.exp(raw[:, 2].clamp(-10, 5))
        t2_long = t2_short + torch.exp(raw[:, 3].clamp(-10, 5))
        time = torch.arange(x.shape[1], device=x.device, dtype=x.dtype) / x.shape[1]
        reconstructed = m0[:, None] * (
            saturation[:, None] * torch.exp(-time / t2_short[:, None])
            + (1 - saturation[:, None]) * torch.exp(-time / t2_long[:, None])
        )
        return saturation, reconstructed

