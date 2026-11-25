"""Non-trainable secondary perturbations D_test(y). Pure PyTorch, shape-preserving."""
import math
import torch


def apply_mulaw_quantization(x: torch.Tensor, mu: float = 255.0) -> torch.Tensor:
    """G.711-style 8-bit mu-law companding -> 256-level quantization -> expansion."""
    x = torch.nan_to_num(x).clamp(-1.0, 1.0)
    comp = torch.sign(x) * torch.log1p(mu * x.abs()) / math.log1p(mu)
    q = torch.round((comp + 1.0) * 127.5) / 127.5 - 1.0
    out = torch.sign(q) * ((1.0 + mu) ** q.abs() - 1.0) / mu
    return out.clamp(-1.0, 1.0)


def apply_uniform_quantization(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Uniform linear PCM quantization (ablation 2)."""
    levels = 2 ** (bits - 1)
    x = torch.nan_to_num(x).clamp(-1.0, 1.0)
    return (torch.round(x * levels) / levels).clamp(-1.0, 1.0)


def apply_gaussian_noise(x: torch.Tensor, sigma: float = 0.01, seed: int | None = 0) -> torch.Tensor:
    """Baseline F: random perturbation control. Deterministic when seed is given."""
    g = None
    if seed is not None:
        g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(x.shape, generator=g).to(x.device, x.dtype)
    return (torch.nan_to_num(x) + sigma * noise).clamp(-1.0, 1.0)


PERTURBATIONS = {
    "mulaw8": apply_mulaw_quantization,
    "uniform8": apply_uniform_quantization,
    "gaussian": apply_gaussian_noise,
}


def get_perturbation(name: str):
    if name not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation '{name}'. Options: {list(PERTURBATIONS)}")
    return PERTURBATIONS[name]
