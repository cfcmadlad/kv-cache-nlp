import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import REPO_ROOT
from eval_blt import load_blt
from kvc.bytedata import open_split
from kvc.cache import Policy, attention_scores, visible_mask
from kvc.patching import patch_flags
from models.entropy import BOS
from train_blt import patch_settings

SPREAD = 0.5


def schedule(budget: int, layers: int, spread: float) -> list:
    return [max(10, round(budget * (1 + spread * (1 - 2 * l / (layers - 1))))) for l in range(layers)]


@torch.no_grad()
def layer_attention(model, inputs, keep=None):
    for blk in model.glob:
        blk.attn.record = []
    model((*inputs, keep) if keep is not None else inputs)
    out = [blk.attn.record[0][0].mean(0).double().cpu().numpy() for blk in model.glob]
    for blk in model.glob:
        blk.attn.record = None
    return out


def concentration(attn: np.ndarray) -> float:
    rows = attn[10:]
    return float(np.mean(1.0 / np.maximum((rows ** 2).sum(1), 1e-12))) if len(rows) else 0.0


@torch.no_grad()
def run(model, cfg, args, device):
    ps = patch_settings(cfg)
    base = REPO_ROOT / "data" / "bytes" / cfg["dataset"]
    data, offsets = open_split(base, args.split)
    entropy = np.load(base / f"entropy_{cfg['patching']['entropy_run']}" / f"{args.split}.npy", mmap_mode="r")
    count = min(args.max_docs, len(offsets) - 1)
    L = len(model.glob)
    plans = {"snap_uniform": ("snap", [args.budget] * L), "snap_pyramid": ("snap", schedule(args.budget, L, SPREAD)), "snap_inverse": ("snap", schedule(args.budget, L, -SPREAD)),
             "recency_uniform": ("recency", [args.budget] * L), "recency_pyramid": ("recency", schedule(args.budget, L, SPREAD)), "recency_inverse": ("recency", schedule(args.budget, L, -SPREAD))}
    variants = ["full"] + list(plans)
    nats = {v: np.zeros(count) for v in variants}
    entries = {v: [] for v in variants}
    sizes = np.zeros(count, dtype=np.int64)
    conc, conc_n = np.zeros(L), 0
    for di in range(count):
        s, e = offsets[di], offsets[di + 1]
        doc, h = np.asarray(data[s:e]), entropy[s:e].astype(np.float32)
        sizes[di] = len(doc)
        for a in range(0, len(doc), cfg["block"]):
            b = min(a + cfg["block"], len(doc))
            x = np.empty(b - a, dtype=np.int64)
            x[0] = BOS if a == 0 else doc[a - 1]
            x[1:] = doc[a:b - 1]
            flags = patch_flags(h[a:b] if ps["mode"] == "entropy" else None, np.arange(a, b), np.zeros(b - a, dtype=bool), ps["mode"], ps["theta"], ps["max_len"], ps["stride"])
            cs = np.cumsum(flags)
            P = int(cs[-1]) + 1
            inputs = (torch.from_numpy(x)[None].to(device), torch.from_numpy(flags)[None].to(device))
            tgt = torch.from_numpy(doc[a:b].astype(np.int64)).to(device)
            full_attn = layer_attention(model, inputs)
            if P > 40:
                conc += np.array([concentration(m) for m in full_attn])
                conc_n += 1
            full = F.cross_entropy(model(inputs)[0][0].float(), tgt, reduction="sum").item()
            for v in variants:
                if v == "full" or P <= max(plans[v][1]):
                    nats[v][di] += full
                    entries[v].append((P + 1) / 2)
                    continue
                mode, budgets = plans[v]
                if mode == "recency":
                    masks = [visible_mask(P, np.arange(P, dtype=float), Policy("recency", bl, 1, 8)) for bl in budgets]
                else:
                    attn = full_attn
                    for _ in range(2):
                        masks = [visible_mask(P, attention_scores(attn[l], "snap", 16), Policy("snap", budgets[l], 1, 8)) for l in range(L)]
                        keep = torch.from_numpy(np.stack(masks))[None].to(device)
                        attn = layer_attention(model, inputs, keep)
                keep = torch.from_numpy(np.stack(masks))[None].to(device)
                logits = model((*inputs, keep))[0][0]
                nats[v][di] += F.cross_entropy(logits.float(), tgt, reduction="sum").item()
                entries[v].append(float(np.mean([m.sum(1).mean() for m in masks])))
    return {"run": args.run, "dataset": cfg["dataset"], "split": args.split, "docs": count, "bytes_per_doc": sizes.tolist(), "budget": args.budget,
            "schedules": {k: v[1] for k, v in plans.items()}, "attention_concentration_by_layer": (conc / max(conc_n, 1)).tolist(),
            "variants": {f"{v}|{args.budget}": {"nats": nats[v].tolist(), "mean_entries": float(np.mean(entries[v]))} for v in variants}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-docs", type=int, default=500)
    p.add_argument("--budget", type=int, default=32)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_blt(args.run, device)
    result = run(model, cfg, args, device)
    out = REPO_ROOT / "runs" / args.run / f"pyramid_{args.split}_{result['docs']}.json"
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
