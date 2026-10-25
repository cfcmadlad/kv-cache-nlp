import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from cache_entropy import load_entropy_model
from data_utils import REPO_ROOT
from eval_blt import load_blt
from kvc.bytedata import open_split
from kvc.cache import ATTN_MODES, Policy, patch_entropy_scores, patch_scores, visible_mask
from kvc.patching import doc_entropy_and_nats, patch_flags
from models.entropy import BOS
from train_blt import patch_settings

CONSONANTS, VOWELS = b"bdfgklmnprstvz", b"aeiou"


def name(rng, syllables):
    return bytes(c for _ in range(syllables) for c in (CONSONANTS[rng.integers(len(CONSONANTS))], VOWELS[rng.integers(len(VOWELS))]))


def build(filler, rng, args, dist):
    word, other = name(rng, args.syllables), name(rng, args.syllables)
    start = int(rng.integers(0, len(filler) - args.lead - dist))
    seg = filler[start:start + args.lead + dist]
    first = args.lead + len(args.cue1)
    tail = args.cue2.encode() + word
    make = lambda w: seg[:args.lead] + args.cue1.encode() + w + b". " + seg[args.lead:] + tail
    return make(word), make(other), (first, first + len(word)), len(word) - args.prefix


@torch.no_grad()
def trial(model, entropy_model, doc, span, wlen, ps, variants, args, device, rng):
    span = (span[0] + args.prefix, span[1])
    n = len(doc)
    h = doc_entropy_and_nats(entropy_model, doc, device)[0].astype(np.float32)
    x = np.array([BOS, *doc[:-1]], dtype=np.int64)
    flags = patch_flags(h if ps["mode"] == "entropy" else None, np.arange(n), np.zeros(n, dtype=bool),
                        ps["mode"], ps["theta"], ps["max_len"], ps["stride"])
    cs = np.cumsum(flags)
    patch_in = cs - flags
    P = int(cs[-1]) + 1
    ent = patch_entropy_scores(patch_in, h.astype(np.float64), P, "mean")
    inputs = (torch.from_numpy(x)[None].to(device), torch.from_numpy(flags)[None].to(device))
    target = torch.from_numpy(np.frombuffer(doc, dtype=np.uint8).astype(np.int64)).to(device)
    row = int(cs[n - wlen]) - 1
    needle = np.unique(patch_in[span[0]:span[1]])
    out = {}
    for mode, budget in variants:
        if mode == "full" or budget >= P:
            logits = model(inputs)[0][0]
            seen = 1.0
        else:
            pol = Policy(mode, budget, args.sink, args.recent if mode != "hybrid" else budget // 2, "mean", 0)
            pol = Policy("entropy", budget, args.sink, budget // 2, "mean", 0) if mode == "hybrid" else pol
            if mode in ATTN_MODES:
                trace = []
                logits = model.forward_online(inputs, pol, None, args.snap_window, trace)[0][0]
                seen = float(np.isin(needle, trace[row]).mean())
            else:
                mask = visible_mask(P, patch_scores(pol, P, ent, rng), pol)
                logits = model((*inputs, torch.from_numpy(mask)[None].to(device)))[0][0]
                seen = float(mask[row, needle].mean())
        lw = logits[n - wlen:].float()
        nll = F.cross_entropy(lw, target[n - wlen:], reduction="mean").item() / np.log(2)
        acc = (lw.argmax(-1) == target[n - wlen:]).float().mean().item()
        out[f"{mode}|{budget}"] = [nll, acc, seen]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--entropy-run", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--syllables", type=int, default=4)
    p.add_argument("--prefix", type=int, default=2)
    p.add_argument("--cue1", default="There was a girl named ")
    p.add_argument("--cue2", default=" Then everybody said hello to ")
    p.add_argument("--lead", type=int, default=150)
    p.add_argument("--distances", nargs="+", type=int, default=[400, 1200, 3000])
    p.add_argument("--budgets", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--policies", nargs="+", default=["recency", "random", "entropy", "hybrid", "h2o", "snap"])
    p.add_argument("--sink", type=int, default=1)
    p.add_argument("--recent", type=int, default=8)
    p.add_argument("--snap-window", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="needle.json")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_blt(args.run, device)
    ps = patch_settings(cfg)
    dataset = args.dataset or cfg["dataset"]
    entropy_model, _ = load_entropy_model(args.entropy_run or cfg["patching"]["entropy_run"], device)
    data, offsets = open_split(REPO_ROOT / "data" / "bytes" / dataset, "val")
    filler = bytes(np.asarray(data[:min(len(data), 4_000_000)]))
    variants = [("full", 0)] + [(m, b) for m in args.policies for b in args.budgets]
    result = {"run": args.run, "dataset": dataset, "syllables": args.syllables, "prefix": args.prefix, "cue1": args.cue1, "cue2": args.cue2, "trials": args.trials, "cells": {}}
    for dist in args.distances:
        rng = np.random.default_rng([args.seed, dist])
        rows = {f"{m}|{b}": [] for m, b in variants + [("ctrl", 0)]}
        for t in range(args.trials):
            doc, ctrl, span, wlen = build(filler, rng, args, dist)
            for k, v in trial(model, entropy_model, doc, span, wlen, ps, variants, args, device, rng).items():
                rows[k].append(v)
            rows["ctrl|0"].append(trial(model, entropy_model, ctrl, span, wlen, ps, [("full", 0)], args, device, rng)["full|0"])
        result["cells"][str(dist)] = rows
        print(dist, {k: np.round(np.mean(v, 0), 3).tolist() for k, v in rows.items() if k.endswith("|32") or k == "full|0"}, flush=True)
    out = REPO_ROOT / "runs" / args.run / args.out
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
