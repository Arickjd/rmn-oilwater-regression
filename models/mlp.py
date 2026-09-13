"""MLP com PCA: célula 14 de modeling.ipynb."""

from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    output_weight = 1.0
    physics_weight = 0.0
    use_scheduler = False

    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.network(x).reshape(-1)

    def loss_components(self, x, y):
        loss = F.mse_loss(self(x), y.reshape(-1))
        return loss, loss.new_zeros(())

