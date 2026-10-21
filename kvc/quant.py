import torch


def fake_quant(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 16:
        return x
    qmax = 2 ** (bits - 1) - 1
    scale = (x.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-8)
    return (x / scale).round().clamp(-qmax, qmax) * scale


def entry_bytes(layers: int, dim: int, heads: int, bits: int) -> float:
    if bits >= 16:
        return layers * 2 * dim * 2
    return layers * (2 * dim * bits / 8 + 2 * heads * 2)
