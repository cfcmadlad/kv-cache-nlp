import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from eval_blt import GLOBAL_CACHE_BYTES_PER_PATCH
from eval_stream import stream_nats
from kvc.cache import Policy, mean_cache_entries, patch_entropy_scores, patch_scores, visible_mask
from kvc.patching import doc_entropy_and_nats, entropy_from_logits, patch_flags
from models.blt import BLT
from models.entropy import BOS

TINY = dict(local_dim=128, local_heads=4, local_window=128, enc_layers=1, dec_layers=1, global_dim=192, global_heads=4, global_layers=3)
POLICIES = ("recency", "random", "entropy")


def byte_stream(texts: list):
    docs = [t.encode("utf-8") for t in texts]
    offsets = np.concatenate([[0], np.cumsum([len(d) for d in docs])])
    return np.frombuffer(b"".join(docs), dtype=np.uint8).astype(np.int64), offsets


def window(raw, offsets, p: int, block: int, stride: int):
    q = np.arange(p, p + block)
    rel = q - offsets[np.searchsorted(offsets, q, side="right") - 1]
    x, y = raw[p - 1:p + block - 1].copy(), raw[p:p + block]
    x[rel == 0] = BOS
    forced = rel == 0
    forced[:-1] |= rel[1:] == 0
    return x, y, patch_flags(None, rel, forced, "fixed", None, 16, stride)


def train_live(texts: list, steps: int, device: str, block: int = 512, batch: int = 16, stride: int = 4, lr: float = 2e-3, log=print):
    raw, offsets = byte_stream(texts)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = BLT(**TINY).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    log(f"{model.num_params() / 1e6:.2f}M parameters, {len(raw):,} training bytes, {steps} steps of {batch} x {block} bytes")
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr * min(step / 30, 0.5 * (1 + math.cos(math.pi * step / steps)) * 0.9 + 0.1)
        rows = [window(raw, offsets, int(p), block, stride) for p in rng.integers(1, len(raw) - block, size=batch)]
        x, y, f = (torch.from_numpy(np.stack(c)).to(device) for c in zip(*rows))
        loss = model((x, f), y)[1]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 25 == 0:
            log(f"step {step:4d}  loss {loss.item():.3f} nats  = {loss.item() / math.log(2):.3f} bits per byte  ({time.perf_counter() - t0:.0f}s)")
    return model.eval()


@torch.no_grad()
def evaluate_live(model, texts: list, budget: int, device: str, stride: int = 4, random_seeds: int = 3, max_bytes: int = 1500):
    per_patch = GLOBAL_CACHE_BYTES_PER_PATCH(TINY)
    nats = {v: 0.0 for v in ("full", *POLICIES)}
    entries = {v: [] for v in nats}
    total = 0
    for text in texts:
        doc = np.frombuffer(text.encode("utf-8")[:max_bytes], dtype=np.uint8).astype(np.int64)
        n = len(doc)
        total += n
        x = np.concatenate([[BOS], doc[:-1]])
        flags = patch_flags(None, np.arange(n), np.zeros(n, dtype=bool), "fixed", None, 16, stride)
        cs = np.cumsum(flags)
        P = int(cs[-1]) + 1
        inputs = (torch.from_numpy(x)[None].to(device), torch.from_numpy(flags)[None].to(device))
        target = torch.from_numpy(doc).to(device)
        logits = model(inputs)[0][0]
        nats["full"] += F.cross_entropy(logits, target, reduction="sum").item()
        entries["full"].append((P + 1) / 2)
        ent = patch_entropy_scores(cs - flags, entropy_from_logits(logits).cpu().numpy().astype(np.float64), P, "mean")
        for mode in POLICIES:
            pol = Policy(mode, budget)
            reps = random_seeds if mode == "random" else 1
            for r in range(reps):
                mask = visible_mask(P, patch_scores(pol, P, ent, np.random.default_rng([0, r])), pol)
                out = model((*inputs, torch.from_numpy(mask)[None].to(device)))[0][0]
                nats[mode] += F.cross_entropy(out, target, reduction="sum").item() / reps
                entries[mode].append(mean_cache_entries(mask) / reps)
    rows = [("full cache", nats["full"] / math.log(2) / total, float(np.mean(entries["full"])) * per_patch)]
    for mode in POLICIES:
        kept = np.mean(entries[mode]) * (random_seeds if mode == "random" else 1)
        rows.append((f"keep {budget} patches: {mode}", nats[mode] / math.log(2) / total, float(kept) * per_patch))
    return rows


def cache_elems(state):
    return sum(c["k"].numel() + c["v"].numel() for c in state["glob"])


@torch.no_grad()
def stream_compare(text: str, model, entropy, cfg: dict, budget: int, device: str, max_bytes: int = 6000, sink: int = 1, recent: int = 8):
    from train_blt import patch_settings
    ps = patch_settings(cfg)
    doc = np.frombuffer(text.encode("utf-8")[:max_bytes], dtype=np.uint8).astype(np.int64)
    n = len(doc)
    h = doc_entropy_and_nats(entropy, bytes(doc.astype(np.uint8)), device)[0].astype(np.float32)
    flags = patch_flags(h, np.arange(n), np.zeros(n, dtype=bool), ps["mode"], ps["theta"], ps["max_len"], ps["stride"])
    tokens = torch.from_numpy(np.concatenate([[BOS], doc[:-1]])).to(device)
    targets = torch.from_numpy(doc).to(device)
    rows = []
    for name, pol in [("full cache", None)] + [(f"keep {budget} patches: {m}", Policy(m, budget, sink, recent, "mean", 0)) for m in POLICIES]:
        t0 = time.perf_counter()
        state = {}
        total = stream_nats(model, tokens, targets, flags, h, pol, device, state)
        rows.append((name, total / math.log(2) / n, cache_elems(state) * 2, time.perf_counter() - t0))
    return {"bytes": n, "patches": int(flags.sum()) + 1, "rows": rows}
