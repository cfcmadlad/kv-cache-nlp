import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rope import apply_rope, rope_cache

BOS = 256


class WindowAttention(nn.Module):
    def __init__(self, dim: int, n_head: int, window: int):
        super().__init__()
        self.n_head, self.window = n_head, window
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin, doc):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        W, out = self.window, []
        for a in range(0, T, W):
            b, s = min(a + W, T), max(0, a - W + 1)
            delta = torch.arange(a, b, device=x.device)[:, None] - torch.arange(s, b, device=x.device)[None]
            mask = ((delta >= 0) & (delta < W))[None, None] & (doc[:, a:b, None] == doc[:, None, s:b])[:, None]
            out.append(F.scaled_dot_product_attention(q[:, :, a:b], k[:, :, s:b], v[:, :, s:b], attn_mask=mask))
        return self.proj(torch.cat(out, dim=2).transpose(1, 2).reshape(B, T, C))

    def step(self, x, cos, sin, cache):
        B, _, C = x.shape
        q, k, v = self.qkv(x).view(B, 1, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache:
            k, v = torch.cat([cache["k"], k], 2)[:, :, -self.window:], torch.cat([cache["v"], v], 2)[:, :, -self.window:]
        cache["k"], cache["v"] = k, v
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, 1, C))


class Block(nn.Module):
    def __init__(self, dim: int, n_head: int, window: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, n_head, window)
        self.fc = nn.Linear(dim, 4 * dim)
        self.out = nn.Linear(4 * dim, dim)

    def forward(self, x, cos, sin, doc):
        x = x + self.attn(self.ln1(x), cos, sin, doc)
        return x + self.out(F.gelu(self.fc(self.ln2(x))))

    def step(self, x, cos, sin, cache):
        x = x + self.attn.step(self.ln1(x), cos, sin, cache)
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class EntropyLM(nn.Module):
    def __init__(self, n_layer: int = 4, n_head: int = 4, dim: int = 256, window: int = 512):
        super().__init__()
        self.head_dim = dim // n_head
        self.emb = nn.Embedding(257, dim)
        self.blocks = nn.ModuleList(Block(dim, n_head, window) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 256, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        doc = (idx == BOS).cumsum(1)
        cos, sin = rope_cache(torch.arange(idx.size(1), device=idx.device), self.head_dim)
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x, cos, sin, doc)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
