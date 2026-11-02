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


def asym_quant(x: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    qmax = 2 ** bits - 1
    lo, hi = x.amin(dim, keepdim=True), x.amax(dim, keepdim=True)
    scale = ((hi - lo) / qmax).clamp(min=1e-8)
    return ((x - lo) / scale).round().clamp(0, qmax) * scale + lo


def kivi_keys(k: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    B, H, T, D = k.shape
    pad = (-T) % group
    kp = torch.cat([k, k[:, :, -1:].expand(-1, -1, pad, -1)], 2) if pad else k
    return asym_quant(kp.view(B, H, -1, group, D), bits, 3).view(B, H, -1, D)[:, :, :T]


def quant_keys(k: torch.Tensor, bits: int, spec) -> torch.Tensor:
    if bits >= 16:
        return k
    if spec is None or spec["keys"] == "sym":
        return fake_quant(k, bits)
    return asym_quant(k, bits, -1) if spec["keys"] == "tok" else kivi_keys(k, bits, spec["group"])


def quant_vals(v: torch.Tensor, bits: int, spec) -> torch.Tensor:
    if bits >= 16:
        return v
    return fake_quant(v, bits) if spec is None or spec["vals"] == "sym" else asym_quant(v, bits, -1)
