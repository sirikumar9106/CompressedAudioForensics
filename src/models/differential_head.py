"""Single source of truth for the differential feature vector (used by training AND inference)."""
import torch
import torch.nn as nn


def build_differential_features(z_y: torch.Tensor, z_yp: torch.Tensor) -> torch.Tensor:
    """f = [mean(z_y) | mean(z_y') | mean(dz) | std(dz) | cos_dist]  ->  [B, 4D+1].

    D=256 gives the spec's 1,025 dims. std is population std (1/T), matching Section 17.
    """
    assert z_y.shape == z_yp.shape and z_y.dim() == 3, f"expected [B,T,D], got {z_y.shape}"
    dz = z_y - z_yp
    m_y, m_yp = z_y.mean(1), z_yp.mean(1)
    cos_dist = 1.0 - torch.cosine_similarity(m_y, m_yp, dim=-1, eps=1e-8).unsqueeze(-1)
    return torch.cat([m_y, m_yp, dz.mean(1), dz.std(1, unbiased=False), cos_dist], dim=-1)


def feature_slices(d: int) -> dict:
    return {"z_y": slice(0, d), "z_yp": slice(d, 2 * d), "delta_mean": slice(2 * d, 3 * d),
            "delta_std": slice(3 * d, 4 * d), "cos_dist": slice(4 * d, 4 * d + 1)}


class DifferentialHead(nn.Module):
    def __init__(self, in_dim: int = 1025, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.LeakyReLU(0.2),
                                 nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, f):
        return self.net(f)
