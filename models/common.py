"""Arquitetura compartilhada pelas três variantes físicas."""

import torch
from torch import nn
from torch.nn import functional as F


class PINNBase(nn.Module):
    output_weight = 1.0
    physics_weight = 4.0
    use_scheduler = False
    time_decay = 0.0

    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 4),
        )

    def decode(self, x):
        """Retorna (saturação, reconstrução) a partir dos parâmetros físicos."""
        raise NotImplementedError

    def forward(self, x):
        return self.decode(x)[0]

    def loss_components(self, x, y):
        saturation, reconstructed = self.decode(x)
        time = torch.arange(x.shape[1], device=x.device, dtype=x.dtype) / x.shape[1]
        weights = torch.exp(-self.time_decay * time)
        output = F.mse_loss(saturation, y.reshape(-1))
        physics = ((x - reconstructed).square() * weights).mean()
        return output, physics

