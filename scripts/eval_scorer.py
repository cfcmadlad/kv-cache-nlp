import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import REPO_ROOT
from eval_blt import load_blt
from eval_policies import attention_matrix
from kvc.bytedata import open_split
from kvc.cache import Policy, attention_scores, mean_cache_entries, patch_entropy_scores, patch_scores, visible_mask
from kvc.patching import patch_flags
from models.entropy import BOS
from train_blt import patch_settings

ENT_COLS = (3, 4, 5, 8)
NAMES = ("recency", "entropy", "snap", "scorer_all", "scorer_noent")


def chunk_inputs(data, h, offsets_doc, a, b, ps, device):
    doc = np.asarray(data[offsets_doc[0]:offsets_doc[1]])
    x = np.empty(b - a, dtype=np.int64)
    x[0] = BOS if a == 0 else doc[a - 1]
    x[1:] = doc[a:b - 1]
    flags = patch_flags(h[a:b] if ps["mode"] == "entropy" else None, np.arange(a, b), np.zeros(b - a, dtype=bool), ps["mode"], ps["theta"], ps["max_len"], ps["stride"])
    cs = np.cumsum(flags)
    return doc, x, flags, cs - flags, int(cs[-1]) + 1


@torch.no_grad()
def kv_norms(model, inputs):
    store = []
    hooks = [blk.attn.qkv.register_forward_hook(lambda m, i, o: store.append(o.detach())) for blk in model.glob]
    model(inputs)
    for hk in hooks:
        hk.remove()
    C = store[0].size(-1) // 3
    k = torch.stack([o[0, :, C:2 * C].norm(dim=-1) for o in store]).mean(0)
    v = torch.stack([o[0, :, 2 * C:].norm(dim=-1) for o in store]).mean(0)
    return k.cpu().numpy().astype(np.float64), v.cpu().numpy().astype(np.float64)


def patch_features(model, inputs, h_chunk, patch_in, P):
    k, v = kv_norms(model, inputs)
    lengths = np.bincount(patch_in, minlength=P).astype(np.float64)
    h = h_chunk.astype(np.float64)
    return np.stack([np.log(np.maximum(lengths, 1)), patch_entropy_scores(patch_in, h, P, "mean"), patch_entropy_scores(patch_in, h, P, "max"),
                     patch_entropy_scores(patch_in, h, P, "first"), k, v], 1)


def pair_features(pf: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    age = (i - j).astype(np.float64)
    return np.column_stack([age, np.log1p(age), pf[j, 0], pf[j, 1], pf[j, 2], pf[j, 3], pf[j, 4], pf[j, 5], pf[i, 1], j / np.maximum(i, 1)])


def sample_pairs(P: int, rng, steps: int):
    rows = []
    for i in rng.choice(np.arange(24, P), size=min(steps, max(P - 24, 0)), replace=False) if P > 24 else []:
        js = np.arange(1, i - 8 + 1)
        rows.append(np.column_stack([np.full(len(js), i), js]))
    return np.concatenate(rows) if rows else np.zeros((0, 2), dtype=np.int64)


class Scorer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.enable_grad()
def fit(X: np.ndarray, y: np.ndarray, device: str, epochs: int, seed: int) -> tuple:
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.from_numpy((X - mu) / sd).float().to(device)
    yt = torch.from_numpy(y).float().to(device)
    model = Scorer(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=device)
        for a in range(0, len(Xt), 4096):
            idx = perm[a:a + 4096]
            loss = F.mse_loss(model(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval(), mu, sd


@torch.no_grad()
def predict(model, mu, sd, X: np.ndarray, device: str) -> np.ndarray:
    Xt = torch.from_numpy((X - mu) / sd).float().to(device)
    return torch.cat([model(Xt[a:a + 65536]) for a in range(0, len(Xt), 65536)]).cpu().numpy()


def target(attn: np.ndarray, window: int) -> np.ndarray:
    return np.log(attention_scores(attn, "snap", window) + 1e-6)


@torch.no_grad()
def collect(model, cfg, args, device):
    ps = patch_settings(cfg)
    base = REPO_ROOT / "data" / "bytes" / cfg["dataset"]
    data, offsets = open_split(base, "train")
    entropy = np.load(base / f"entropy_{cfg['patching']['entropy_run']}" / "train.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    Xs, ys = [], []
    for i in range(min(args.fit_docs, len(offsets) - 1)):
        s, e = offsets[i], offsets[i + 1]
        h = entropy[s:e].astype(np.float32)
        for a in range(0, e - s, cfg["block"]):
            b = min(a + cfg["block"], e - s)
            doc, x, flags, patch_in, P = chunk_inputs(data, h, (s, e), a, b, ps, device)
            if P < 40:
                continue
            inputs = (torch.from_numpy(x)[None].to(device), torch.from_numpy(flags)[None].to(device))
            pf = patch_features(model, inputs, h[a:b], patch_in, P)
            tg = target(attention_matrix(model, inputs), args.snap_window)
            pairs = sample_pairs(P, rng, args.steps_per_chunk)
            Xs.append(pair_features(pf, pairs[:, 0], pairs[:, 1]))
            ys.append(tg[pairs[:, 0], pairs[:, 1]])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    if len(X) > args.max_samples:
        keep = rng.choice(len(X), args.max_samples, replace=False)
        X, y = X[keep], y[keep]
    return X, y


@torch.no_grad()
def run(model, cfg, args, device):
    X, y = collect(model, cfg, args, device)
    keep_cols = [c for c in range(X.shape[1]) if c not in ENT_COLS]
    scorers = {"scorer_all": (fit(X, y, device, args.epochs, args.seed), list(range(X.shape[1]))),
               "scorer_noent": (fit(X[:, keep_cols], y, device, args.epochs, args.seed), keep_cols)}
    ps = patch_settings(cfg)
    dataset = cfg["dataset"]
    base = REPO_ROOT / "data" / "bytes" / dataset
    data, offsets = open_split(base, args.split)
    entropy = np.load(base / f"entropy_{cfg['patching']['entropy_run']}" / f"{args.split}.npy", mmap_mode="r")
    count = min(args.max_docs, len(offsets) - 1)
    variants = [("full", 0)] + [(n, args.budget) for n in NAMES]
    nats = {v: np.zeros(count) for v in variants}
    entries = {v: [] for v in variants}
    sizes = np.zeros(count, dtype=np.int64)
    for di in range(count):
        s, e = offsets[di], offsets[di + 1]
        h = entropy[s:e].astype(np.float32)
        sizes[di] = e - s
        for a in range(0, e - s, cfg["block"]):
            b = min(a + cfg["block"], e - s)
            doc, x, flags, patch_in, P = chunk_inputs(data, h, (s, e), a, b, ps, device)
            inputs = (torch.from_numpy(x)[None].to(device), torch.from_numpy(flags)[None].to(device))
            tgt = torch.from_numpy(doc[a:b].astype(np.int64)).to(device)
            full = F.cross_entropy(model(inputs)[0][0].float(), tgt, reduction="sum").item()
            ent = patch_entropy_scores(patch_in, h[a:b].astype(np.float64), P, "mean")
            pf = patch_features(model, inputs, h[a:b], patch_in, P)
            ii, jj = np.indices((P, P))
            Xp = pair_features(pf, ii.ravel(), jj.ravel())
            for v in variants:
                name, budget = v
                if name == "full" or budget >= P:
                    nats[v][di] += full
                    entries[v].append((P + 1) / 2)
                    continue
                if name == "snap":
                    keep = None
                    for _ in range(2):
                        sc = attention_scores(attention_matrix(model, inputs, keep), "snap", args.snap_window)
                        mask = visible_mask(P, sc, Policy("snap", budget, 1, 8))
                        keep = torch.from_numpy(mask)[None].to(device)
                elif name in scorers:
                    (net, mu, sd), cols = scorers[name]
                    sc = predict(net, mu, sd, Xp[:, cols], device).reshape(P, P)
                    mask = visible_mask(P, sc, Policy("snap", budget, 1, 8))
                else:
                    mask = visible_mask(P, patch_scores(Policy(name, budget, 1, 8), P, ent), Policy(name, budget, 1, 8))
                logits = model((*inputs, torch.from_numpy(mask)[None].to(device)))[0][0]
                nats[v][di] += F.cross_entropy(logits.float(), tgt, reduction="sum").item()
                entries[v].append(mean_cache_entries(mask))
    return {"run": args.run, "dataset": dataset, "split": args.split, "docs": count, "bytes_per_doc": sizes.tolist(), "fit_samples": int(len(X)),
            "variants": {f"{n}|{b}": {"nats": nats[(n, b)].tolist(), "mean_entries": float(np.mean(entries[(n, b)]))} for n, b in variants}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-docs", type=int, default=500)
    p.add_argument("--budget", type=int, default=32)
    p.add_argument("--fit-docs", type=int, default=300)
    p.add_argument("--steps-per-chunk", type=int, default=12)
    p.add_argument("--max-samples", type=int, default=1500000)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--snap-window", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_blt(args.run, device)
    result = run(model, cfg, args, device)
    out = REPO_ROOT / "runs" / args.run / f"scorer_{args.split}_{result['docs']}.json"
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
