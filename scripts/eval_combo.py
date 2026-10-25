import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import REPO_ROOT
from eval_blt import load_blt
from eval_policies import attention_matrix
from kvc.bytedata import open_split
from kvc.cache import ATTN_MODES, Policy, attention_scores, patch_entropy_scores, patch_scores, visible_mask
from kvc.patching import patch_flags
from kvc.quant import entry_bytes
from models.entropy import BOS
from train_blt import patch_settings


def build_masks(model, inputs, P, ent, mode, budget, args, di, a, device):
    pol = Policy(mode, budget, args.sink, args.recent, "mean", args.seed)
    out = []
    for r in range(args.random_seeds if mode == "random" else 1):
        if mode in ATTN_MODES:
            keep = None
            for _ in range(2):
                mk = visible_mask(P, attention_scores(attention_matrix(model, inputs, keep), mode, args.snap_window), pol)
                keep = torch.from_numpy(mk)[None].to(device)
        else:
            mk = visible_mask(P, patch_scores(pol, P, ent, np.random.default_rng([args.seed, r, di, a])), pol)
        out.append(mk)
    return out


@torch.no_grad()
def run(model, cfg, args, device):
    ps = patch_settings(cfg)
    base = REPO_ROOT / "data" / "bytes" / cfg["dataset"]
    data, offsets = open_split(base, args.split)
    entropy = np.load(base / f"entropy_{cfg['patching']['entropy_run']}" / f"{args.split}.npy", mmap_mode="r")
    count = len(offsets) - 1 if args.max_docs is None else min(args.max_docs, len(offsets) - 1)
    docs = [i for i in range(count) if offsets[i + 1] > offsets[i]]
    variants = [("full", 0, 16)] + [(m, b, bits) for m in args.policies for b in args.budgets for bits in args.bits]
    nats = {v: np.zeros(len(docs)) for v in variants}
    entries = {v: [] for v in variants}
    sizes = np.zeros(len(docs), dtype=np.int64)
    m = cfg["model"]
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
            P = int(cs[-1]) + 1
            inputs = (torch.from_numpy(x)[None].to(device), torch.from_numpy(flags)[None].to(device))
            target = torch.from_numpy(doc[a:b].astype(np.int64)).to(device)
            ent = patch_entropy_scores(cs - flags, h[a:b].astype(np.float64), P, "mean")
            masks = {}
            for v in variants:
                mode, budget, bits = v
                if mode == "full" or budget >= P:
                    mks, count_e = [None], (P + 1) / 2
                else:
                    if (mode, budget) not in masks:
                        masks[(mode, budget)] = build_masks(model, inputs, P, ent, mode, budget, args, di, a, device)
                    mks = masks[(mode, budget)]
                    count_e = float(np.mean([mk.sum(1).mean() for mk in mks]))
                total = 0.0
                for mk in mks:
                    extra = [None if mk is None else torch.from_numpy(mk)[None].to(device)]
                    if bits < 16:
                        extra.append((torch.zeros(1, P, P, dtype=torch.bool, device=device), 16, bits))
                    logits = model((*inputs, *extra) if (mk is not None or bits < 16) else inputs)[0][0]
                    total += F.cross_entropy(logits.float(), target, reduction="sum").item()
                nats[v][di] += total / len(mks)
                entries[v].append(count_e)
    out = {"run": args.run, "split": args.split, "docs": len(docs), "bytes_per_doc": sizes.tolist(), "variants": {}}
    for v in variants:
        eb = entry_bytes(m["global_layers"], m["global_dim"], m["global_heads"], v[2])
        out["variants"]["|".join(map(str, v))] = {"nats": nats[v].tolist(), "mean_entries": float(np.mean(entries[v])),
                                                  "mean_cache_bytes": float(np.mean(entries[v])) * eb}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--policies", nargs="+", default=["recency", "snap"])
    p.add_argument("--budgets", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--bits", nargs="+", type=int, default=[16, 8, 4])
    p.add_argument("--sink", type=int, default=1)
    p.add_argument("--recent", type=int, default=8)
    p.add_argument("--snap-window", type=int, default=16)
    p.add_argument("--random-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="combo.json")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_blt(args.run, device)
    result = run(model, cfg, args, device)
    out = REPO_ROOT / "runs" / args.run / args.out
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
