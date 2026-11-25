"""Delta-Net: ONE shared encoder applied to y and y' = D_test(y). Labels: 0 = spoof, 1 = bonafide."""
import torch
import torch.nn as nn

from .backbones import RawNet2Encoder
from .degradation_engine import get_perturbation
from .differential_head import DifferentialHead, build_differential_features


class DeltaNet(nn.Module):
    def __init__(self, encoder: RawNet2Encoder | None = None, perturbation: str = "mulaw8",
                 hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder or RawNet2Encoder(proj_dim=256)
        self.perturbation_name = perturbation
        self.perturb = get_perturbation(perturbation)
        self.feat_dim = 4 * self.encoder.out_dim + 1
        self.head = DifferentialHead(self.feat_dim, hidden, dropout)

    def features(self, y: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():  # perturbation is a data transform, no gradient through it
            y_prime = self.perturb(y)
        z_y = self.encoder(y)
        z_yp = self.encoder(y_prime)  # same module object -> identical weights
        return build_differential_features(z_y, z_yp)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(y))
