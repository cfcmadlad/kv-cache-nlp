import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import REPO_ROOT
from eval_blt import load_blt
from eval_policies import attention_matrix
from kvc.bytedata import open_split
from kvc.cache import Policy, attention_scores, evict_one, mean_cache_entries, patch_entropy_scores, patch_scores, visible_mask
from kvc.patching import patch_flags
from models.entropy import BOS
from train_blt import patch_settings

FLOORS = (3, 4)
LAMBDAS = (0.25, 0.5)


def pct_rank(v: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(v)) / max(len(v) - 1, 1)


def hybrid_scores(snap: np.ndarray, ent: np.ndarray, lam: float) -> np.ndarray:
    out = np.zeros_like(snap)
    for i in range(len(snap)):
        out[i, :i + 1] = pct_rank(snap[i, :i + 1]) + lam * pct_rank(ent[:i + 1])
    return out


def dynamic_mask(num_patches: int, ent: np.ndarray, budget: int, pol: Policy) -> np.ndarray:
    floor = pol.sink + pol.recent + 1
    median = np.median(ent)
    scores = np.arange(num_patches, dtype=float)
    mask = np.zeros((num_patches, num_patches), dtype=bool)
    ids = np.empty(0, dtype=np.int64)
    for i in range(num_patches):
        ids = np.append(ids, i)
        limit = max(floor, round(budget * (1.5 if ent[i] >= median else 0.5)))
        while len(ids) > limit:
            ids = np.delete(ids, evict_one(ids, scores, pol))
        mask[i, ids] = True
    return mask


def names() -> list:
    return ["recency", "entropy", "snap"] + [f"floor{k}" for k in FLOORS] + [f"hybrid{l}" for l in LAMBDAS] + ["dynamic"]


@torch.no_grad()
def run(model, cfg, args, device):
    ps = patch_settings(cfg)
    dataset = cfg["dataset"]
    base = REPO_ROOT / "data" / "bytes" / dataset
    data, offsets = open_split(base, args.split)
    entropy = np.load(base / f"entropy_{cfg['patching']['entropy_run']}" / f"{args.split}.npy", mmap_mode="r")
    count = len(offsets) - 1 if args.max_docs is None else min(args.max_docs, len(offsets) - 1)
    docs = list(range(count))
    variants = [("full", 0)] + [(n, b) for n in names() for b in args.budgets] + [("recency", b) for b in args.extra_recency if b not in args.budgets]
    nats = {v: np.zeros(len(docs)) for v in variants}
    entries = {v: [] for v in variants}
    sizes = np.zeros(len(docs), dtype=np.int64)
    block = cfg["block"]
    for di, i in enumerate(docs):
        s, e = offsets[i], offsets[i + 1]
        doc, h = np.asarray(data[s:e]), entropy[s:e].astype(np.float32)
        sizes[di] = len(doc)
        for a in range(0, len(doc), block):
            b = min(a + block, len(doc))
            x = np.empty(b - a, dtype=np.int64)
            x[0] = BOS if a == 0 else doc[a - 1]
            x[1:] = doc[a:b - 1]
            flags = patch_flags(h[a:b] if ps["mode"] == "entropy" else None, np.arange(a, b), np.zeros(b - a, dtype=bool),
                                ps["mode"], ps["theta"], ps["max_len"], ps["stride"])
            cs = np.cumsum(flags)
            patch_in = cs - flags
            P = int(cs[-1]) + 1
            lengths = np.bincount(patch_in, minlength=P)
            inputs = (torch.from_numpy(x)[None].to(device), torch.from_numpy(flags)[None].to(device))
            target = torch.from_numpy(doc[a:b].astype(np.int64)).to(device)
            chunk_full = F.cross_entropy(model(inputs)[0][0].float(), target, reduction="sum").item()
            ent = patch_entropy_scores(patch_in, h[a:b].astype(np.float64), P, "mean")
            for v in variants:
                name, budget = v
                if name == "full" or budget >= P:
                    nats[v][di] += chunk_full
                    entries[v].append((P + 1) / 2)
                    continue
                pol = Policy(name if name in ("recency", "entropy") else "snap" if name == "snap" or name.startswith("hybrid") else "recency", budget, args.sink, args.recent)
                if name in ("snap",) or name.startswith("hybrid"):
                    keep = None
                    lam = float(name[6:]) if name.startswith("hybrid") else None
                    for _ in range(2):
                        sc = attention_scores(attention_matrix(model, inputs, keep), "snap", 16)
                        if lam is not None:
                            sc = hybrid_scores(sc, ent, lam)
                        mask = visible_mask(P, sc, pol)
                        keep = torch.from_numpy(mask)[None].to(device)
                elif name == "dynamic":
                    mask = dynamic_mask(P, ent, budget, pol)
                elif name.startswith("floor"):
                    sc = np.where(lengths >= int(name[5:]), ent, -1e9)
                    mask = visible_mask(P, sc, Policy("entropy", budget, args.sink, args.recent))
                else:
                    mask = visible_mask(P, patch_scores(pol, P, ent), pol)
                logits = model((*inputs, torch.from_numpy(mask)[None].to(device)))[0][0]
                nats[v][di] += F.cross_entropy(logits.float(), target, reduction="sum").item()
                entries[v].append(mean_cache_entries(mask))
    return {"run": args.run, "dataset": dataset, "split": args.split, "docs": len(docs), "bytes_per_doc": sizes.tolist(),
            "variants": {f"{n}|{b}": {"nats": nats[(n, b)].tolist(), "mean_entries": float(np.mean(entries[(n, b)]))} for n, b in variants}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--budgets", nargs="+", type=int, default=[32, 64])
    p.add_argument("--extra-recency", nargs="*", type=int, default=[])
    p.add_argument("--sink", type=int, default=1)
    p.add_argument("--recent", type=int, default=8)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_blt(args.run, device)
    result = run(model, cfg, args, device)
    out = REPO_ROOT / "runs" / args.run / f"variants_{args.split}_{result['docs']}.json"
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
