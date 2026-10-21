import torch


def rope_cache(positions: torch.Tensor, head_dim: int, base: float = 10000.0):
    inv = 1.0 / base ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim)
    angles = positions.float()[:, None] * inv[None]
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2).type_as(x)
