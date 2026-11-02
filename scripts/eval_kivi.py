import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import REPO_ROOT
from eval_blt import load_blt
from eval_policies import attention_matrix
from kvc.bytedata import open_split
from kvc.cache import Policy, attention_scores, visible_mask
from kvc.patching import patch_flags
from kvc.quant import entry_bytes
from models.entropy import BOS
from train_blt import patch_settings

GROUP = 16
RECENT = 8
SPECS = {"sym": (None, 0, False), "sym_r8": (None, RECENT, False), "tok": ({"keys": "tok", "vals": "tok"}, 0, False),
         "kivi": ({"keys": "chan", "vals": "tok", "group": GROUP}, 0, True), "kivi_r8": ({"keys": "chan", "vals": "tok", "group": GROUP}, RECENT, True)}


def hi_mask(P: int, recent: int, grouped: bool) -> np.ndarray:
    i, j = np.indices((P, P))
    hi = np.zeros((P, P), dtype=bool)
    if recent:
        hi |= j > i - recent
    if grouped:
        hi |= ((j // GROUP + 1) * GROUP - 1) > i
    return hi


def low_bytes(m: dict, bits: int, spec) -> float:
    base = entry_bytes(m["global_layers"], m["global_dim"], m["global_heads"], bits)
    if spec is None or bits >= 16:
        return base
    d, h, layers = m["global_dim"], m["global_heads"], m["global_layers"]
    values = d * bits / 8 + 4 * h
    keys = d * bits / 8 + (4 * h if spec["keys"] == "tok" else 4 * d / spec["group"])
    return layers * (keys + values)


@torch.no_grad()
def run(model, cfg, args, device):
    ps = patch_settings(cfg)
    base = REPO_ROOT / "data" / "bytes" / cfg["dataset"]
    data, offsets = open_split(base, args.split)
    entropy = np.load(base / f"entropy_{cfg['patching']['entropy_run']}" / f"{args.split}.npy", mmap_mode="r")
    count = min(args.max_docs, len(offsets) - 1)
    m = cfg["model"]
    full_bytes = entry_bytes(m["global_layers"], m["global_dim"], m["global_heads"], 16)
    variants = [("full", 16, "none", 0)] + [(name, bits, "full", 0) for name in SPECS for bits in args.bits] + [(name, bits, "snap", args.budget) for name in ("sym", "kivi_r8") for bits in args.bits]
    nats = {v: np.zeros(count) for v in variants}
    cost = {v: [] for v in variants}
    sizes = np.zeros(count, dtype=np.int64)
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
            P = int(np.cumsum(flags)[-1]) + 1
            inputs = (torch.from_numpy(x)[None].to(device), torch.from_numpy(flags)[None].to(device))
            tgt = torch.from_numpy(doc[a:b].astype(np.int64)).to(device)
            snap_mask = None
            for v in variants:
                name, bits, evict, budget = v
                if name == "full":
                    logits = model(inputs)[0][0]
                    n_vis = (P + 1) / 2
                    byte_cost = n_vis * full_bytes
                else:
                    spec, recent, grouped = SPECS[name]
                    if evict == "snap":
                        if P <= budget:
                            keep = None
                        else:
                            if snap_mask is None:
                                kp = None
                                for _ in range(2):
                                    snap_mask = visible_mask(P, attention_scores(attention_matrix(model, inputs, kp), "snap", 16), Policy("snap", budget, 1, 8))
                                    kp = torch.from_numpy(snap_mask)[None].to(device)
                            keep = snap_mask
                    else:
                        keep = None
                    vis = np.tril(np.ones((P, P), dtype=bool)) if keep is None else keep
                    hi = hi_mask(P, recent, grouped)
                    n_hi, n_lo = (vis & hi).sum(1), (vis & ~hi).sum(1)
                    byte_cost = float((n_hi * full_bytes + n_lo * low_bytes(m, bits, spec)).mean())
                    tier = (torch.from_numpy(hi)[None].to(device), 16, bits, spec)
                    extra = (None if keep is None else torch.from_numpy(keep)[None].to(device), tier)
                    logits = model((*inputs, *extra))[0][0]
                nats[v][di] += F.cross_entropy(logits.float(), tgt, reduction="sum").item()
                cost[v].append(byte_cost)
    return {"run": args.run, "dataset": cfg["dataset"], "split": args.split, "docs": count, "bytes_per_doc": sizes.tolist(), "full_entry_bytes": full_bytes,
            "variants": {f"{n}|{bits}|{ev}|{bd}": {"nats": nats[(n, bits, ev, bd)].tolist(), "mean_cache_bytes": float(np.mean(cost[(n, bits, ev, bd)]))} for n, bits, ev, bd in variants}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-docs", type=int, default=500)
    p.add_argument("--budget", type=int, default=32)
    p.add_argument("--bits", nargs="+", type=int, default=[4, 3, 2])
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_blt(args.run, device)
    result = run(model, cfg, args, device)
    out = REPO_ROOT / "runs" / args.run / f"kivi_{args.split}_{result['docs']}.json"
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
