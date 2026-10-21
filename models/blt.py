import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kvc.cache import Policy, QuantPolicy, evict_one, insertion_hi, tier_row
from kvc.quant import fake_quant
from models.entropy import BOS, Block
from models.rope import apply_rope, rope_cache


class GlobalAttention(nn.Module):
    def __init__(self, dim: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.record = None
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def _qkv(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def forward(self, x, cos, sin, doc, keep=None, tier=None):
        B, T, C = x.shape
        q, k, v = self._qkv(x, cos, sin)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))[None, None] & (doc[:, :, None] == doc[:, None, :])[:, None]
        if keep is not None:
            mask = mask & keep[:, None]
        if self.record is not None:
            w = torch.softmax((q @ k.transpose(-1, -2) * q.size(-1) ** -0.5).masked_fill(~mask, float("-inf")), -1)
            self.record.append(w.detach())
            return self.proj((w @ v).transpose(1, 2).reshape(B, T, C))
        if tier is not None:
            hi_mask, hb, lb = tier
            hi = hi_mask[:, None]
            scale = q.size(-1) ** -0.5
            s = torch.where(hi, q @ fake_quant(k, hb).transpose(-1, -2), q @ fake_quant(k, lb).transpose(-1, -2)) * scale
            w = torch.softmax(s.masked_fill(~mask, float("-inf")), -1)
            out = (w * hi) @ fake_quant(v, hb) + (w * ~hi) @ fake_quant(v, lb)
            return self.proj(out.transpose(1, 2).reshape(B, T, C))
        return self.proj(F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, T, C))

    def step(self, x, cos, sin, cache, keep=None, tier=None):
        q, k, v = self._qkv(x, cos, sin)
        if cache:
            old_k, old_v = (cache["k"], cache["v"]) if keep is None else (cache["k"][:, :, keep], cache["v"][:, :, keep])
            k, v = torch.cat([old_k, k], 2), torch.cat([old_v, v], 2)
        cache["k"], cache["v"] = k, v
        B, _, C = x.shape
        if tier is not None:
            hi_row, hb, lb = tier
            hi = hi_row[None, None, :, None]
            k, v = torch.where(hi, fake_quant(k, hb), fake_quant(k, lb)), torch.where(hi, fake_quant(v, hb), fake_quant(v, lb))
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, 1, C))


class GlobalBlock(nn.Module):
    def __init__(self, dim: int, n_head: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = GlobalAttention(dim, n_head)
        self.fc = nn.Linear(dim, 4 * dim)
        self.out = nn.Linear(4 * dim, dim)

    def forward(self, x, cos, sin, doc, keep=None, tier=None):
        x = x + self.attn(self.ln1(x), cos, sin, doc, keep, tier)
        return x + self.out(F.gelu(self.fc(self.ln2(x))))

    def step(self, x, cos, sin, cache, keep=None, tier=None):
        x = x + self.attn.step(self.ln1(x), cos, sin, cache, keep, tier)
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class BLT(nn.Module):
    def __init__(self, local_dim=256, local_heads=4, local_window=256, enc_layers=1, dec_layers=2,
                 global_dim=384, global_heads=6, global_layers=6):
        super().__init__()
        self.local_dim, self.global_dim = local_dim, global_dim
        self.local_head_dim, self.global_head_dim = local_dim // local_heads, global_dim // global_heads
        self.emb = nn.Embedding(257, local_dim)
        self.enc = nn.ModuleList(Block(local_dim, local_heads, local_window) for _ in range(enc_layers))
        self.pool = nn.Linear(local_dim, global_dim)
        self.glob = nn.ModuleList(GlobalBlock(global_dim, global_heads) for _ in range(global_layers))
        self.ln_g = nn.LayerNorm(global_dim)
        self.back = nn.Linear(global_dim, local_dim)
        self.dec = nn.ModuleList(Block(local_dim, local_heads, local_window) for _ in range(dec_layers))
        self.ln_d = nn.LayerNorm(local_dim)
        self.head = nn.Linear(local_dim, 256, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, inputs, targets=None):
        idx, nxt, *extra = inputs
        keep = extra[0] if extra else None
        tier = extra[1] if len(extra) > 1 else None
        B, N = idx.shape
        nxt = nxt.long()
        doc = (idx == BOS).cumsum(1)
        cs = nxt.cumsum(1)
        patch_in = cs - nxt
        P = int(cs[:, -1].max()) + 1
        cos, sin = rope_cache(torch.arange(N, device=idx.device), self.local_head_dim)
        e = self.emb(idx)
        for block in self.enc:
            e = block(e, cos, sin, doc)
        pooled = torch.zeros(B, P, self.local_dim, device=idx.device, dtype=e.dtype).scatter_add(
            1, patch_in[..., None].expand(-1, -1, self.local_dim), e)
        count = torch.zeros(B, P, device=idx.device, dtype=e.dtype).scatter_add(1, patch_in, torch.ones_like(patch_in, dtype=e.dtype))
        patch_doc = torch.zeros(B, P, dtype=doc.dtype, device=idx.device).scatter_reduce(1, patch_in, doc, "amax")
        g = self.pool(pooled / count.clamp(min=1)[..., None])
        gcos, gsin = rope_cache(torch.arange(P, device=idx.device), self.global_head_dim)
        for block in self.glob:
            g = block(g, gcos, gsin, patch_doc, keep, tier)
        g = self.ln_g(g)
        prev = g.gather(1, (cs - 1).clamp(min=0)[..., None].expand(-1, -1, self.global_dim))
        d = e + self.back(prev)
        for block in self.dec:
            d = block(d, cos, sin, doc)
        logits = self.head(self.ln_d(d))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_state(self):
        return {"enc": [{} for _ in self.enc], "dec": [{} for _ in self.dec], "glob": [{} for _ in self.glob],
                "sum": None, "count": 0, "pos": 0, "patches": 0, "prev": None,
                "ids": np.empty(0, dtype=np.int64), "scores": [], "h_sum": 0.0, "h_max": 0.0}

    @torch.no_grad()
    def step(self, state, token: torch.Tensor, starts_next: bool, entropy: float = 0.0, policy: Policy = None, qpolicy: QuantPolicy = None) -> torch.Tensor:
        pos = torch.tensor([state["pos"]], device=token.device)
        cos, sin = rope_cache(pos, self.local_head_dim)
        e = self.emb(token.view(-1, 1))
        for block, cache in zip(self.enc, state["enc"]):
            e = block.step(e, cos, sin, cache)
        state["sum"] = e if state["sum"] is None else state["sum"] + e
        state["count"] += 1
        state["h_sum"] += entropy
        state["h_max"] = max(state["h_max"], entropy)
        if starts_next:
            state["count_prev"] = state["count"]
            g = self.pool(state["sum"] / state["count"])
            pid = state["patches"]
            gcos, gsin = rope_cache(torch.tensor([pid], device=token.device), self.global_head_dim)
            keep = None
            ids = np.append(state["ids"], pid)
            state["scores"].append(_patch_score(state, pid, policy))
            if policy is not None and policy.mode != "full" and len(ids) > policy.budget:
                drop = evict_one(ids, np.asarray(state["scores"]), policy)
                keep = torch.tensor(np.delete(np.arange(len(ids) - 1), drop), device=token.device)
                ids = np.delete(ids, drop)
            tier = None
            if qpolicy is not None:
                state.setdefault("rng", np.random.default_rng(qpolicy.seed))
                score = state["h_max"] if qpolicy.agg == "max" else state["h_sum"] / state["count_prev"]
                flags = state.setdefault("flags", np.empty(0, dtype=bool))
                seen = np.asarray(state.setdefault("qscores", []))
                state["flags"] = np.append(flags, insertion_hi(qpolicy, pid, score, seen, state["rng"].random()))
                state["qscores"].append(score)
                row = tier_row(qpolicy, np.arange(pid + 1), state["flags"], pid)
                tier = (torch.from_numpy(row).to(token.device), qpolicy.hi_bits, qpolicy.lo_bits)
            for block, cache in zip(self.glob, state["glob"]):
                g = block.step(g, gcos, gsin, cache, keep, tier)
            state["ids"] = ids
            state["h_sum"], state["h_max"] = 0.0, 0.0
            state["prev"] = self.ln_g(g)
            state["sum"], state["count"] = None, 0
            state["patches"] += 1
        d = e + self.back(state["prev"])
        for block, cache in zip(self.dec, state["dec"]):
            d = block.step(d, cos, sin, cache)
        state["pos"] += 1
        return self.head(self.ln_d(d))[:, 0]


def _patch_score(state, pid: int, policy: Policy) -> float:
    if policy is None:
        return 0.0
    if policy.mode == "recency":
        return float(pid)
    if policy.mode == "random":
        state.setdefault("rng", np.random.default_rng(policy.seed))
        return float(state["rng"].random())
    h = state["h_max"] if policy.agg == "max" else state["h_sum"] / state["count_prev"]
    return h if policy.mode == "entropy" else -h


def expected_params(local_dim=256, enc_layers=1, dec_layers=2, global_dim=384, global_layers=6) -> int:
    block = lambda d: 12 * d * d + 9 * d
    return (257 * local_dim + enc_layers * block(local_dim) + local_dim * global_dim + global_dim
            + global_layers * block(global_dim) + 2 * global_dim + global_dim * local_dim + local_dim
            + dec_layers * block(local_dim) + 2 * local_dim + local_dim * 256)
