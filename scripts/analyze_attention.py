import argparse

import numpy as np
import torch

from data_utils import REPO_ROOT
from eval_blt import load_blt
from kvc.bytedata import open_split
from kvc.cache import patch_entropy_scores
from kvc.patching import patch_flags
from models.entropy import BOS
from train_blt import patch_settings


def ranks(x):
    return np.argsort(np.argsort(x)).astype(float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-docs", type=int, default=100)
    p.add_argument("--protect-recent", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_blt(args.run, device)
    ps = patch_settings(cfg)
    base = REPO_ROOT / "data" / "bytes" / cfg["dataset"]
    data, offsets = open_split(base, args.split)
    entropy = np.load(base / f"entropy_{cfg['patching']['entropy_run']}" / f"{args.split}.npy", mmap_mode="r")
    layers = len(model.glob)
    ent_all, att_all, age_all = [], [[] for _ in range(layers)], []
    for i in range(min(args.max_docs, len(offsets) - 1)):
        s, e = offsets[i], offsets[i + 1]
        if e - s < 64:
            continue
        doc, h = np.asarray(data[s:e]), entropy[s:e].astype(np.float32)
        for a in range(0, len(doc), cfg["block"]):
            b = min(a + cfg["block"], len(doc))
            if b - a < 256:
                continue
            x = np.empty(b - a, dtype=np.int64)
            x[0] = BOS if a == 0 else doc[a - 1]
            x[1:] = doc[a:b - 1]
            flags = patch_flags(h[a:b] if ps["mode"] == "entropy" else None, np.arange(a, b), np.zeros(b - a, dtype=bool),
                                ps["mode"], ps["theta"], ps["max_len"], ps["stride"])
            cs = np.cumsum(flags)
            P = int(cs[-1]) + 1
            ent = patch_entropy_scores(cs - flags, h[a:b].astype(np.float64), P, "mean")
            for blk in model.glob:
                blk.attn.record = []
            with torch.no_grad():
                model((torch.from_numpy(x)[None].to(device), torch.from_numpy(flags)[None].to(device)))
            q = np.arange(P)
            valid = (q[None, :] <= q[:, None] - args.protect_recent)
            far = valid.sum(0)
            for layer, blk in enumerate(model.glob):
                w = blk.attn.record[0][0].mean(0).cpu().numpy()
                att_all[layer].append(np.where(far > 0, (w * valid).sum(0) / np.maximum(far, 1), np.nan))
                blk.attn.record = None
            ent_all.append(ent)
            age_all.append(np.where(far > 0, 1.0, np.nan))
    ent = np.concatenate(ent_all)
    lines = [f"{args.run} on {args.split}: attention received by patches that are at least {args.protect_recent} patches old "
             f"(mean over queries that can see them), against patch entropy.", "",
             "| Layer | Spearman(entropy, attention) | Attention in lowest entropy third | middle third | highest third |", "|---|---|---|---|---|"]
    third = np.quantile(ent, [1 / 3, 2 / 3])
    for layer in range(layers):
        att = np.concatenate(att_all[layer])
        ok = ~np.isnan(att)
        rho = np.corrcoef(ranks(ent[ok]), ranks(att[ok]))[0, 1]
        lo, mid, hi = att[ok & (ent <= third[0])].mean(), att[ok & (ent > third[0]) & (ent <= third[1])].mean(), att[ok & (ent > third[1])].mean()
        lines.append(f"| {layer} | {rho:+.3f} | {lo:.5f} | {mid:.5f} | {hi:.5f} |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        (REPO_ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
