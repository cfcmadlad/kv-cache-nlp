import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import REPO_ROOT
from kvc.bytedata import open_split
from kvc.patching import patch_flags
from models.blt import BLT
from models.entropy import BOS
from train_blt import patch_settings

GLOBAL_CACHE_BYTES_PER_PATCH = lambda m: 2 * m["global_layers"] * m["global_dim"] * 2


def load_blt(run: str, device: str):
    ckpt = torch.load(REPO_ROOT / "runs" / run / "model.pt", map_location=device, weights_only=True)
    model = BLT(**ckpt["config"]["model"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["config"]


@torch.no_grad()
def score_doc(model, doc: np.ndarray, h, ps: dict, block: int, device: str):
    n = len(doc)
    rel = np.arange(n)
    nats, patches = 0.0, 0
    for a in range(0, n, block):
        b = min(a + block, n)
        x = np.empty(b - a, dtype=np.int64)
        x[0] = BOS if a == 0 else doc[a - 1]
        x[1:] = doc[a:b - 1]
        forced = np.zeros(b - a, dtype=bool)
        flags = patch_flags(None if h is None else h[a:b], rel[a:b], forced, ps["mode"], ps["theta"], ps["max_len"], ps["stride"])
        logits = model((torch.from_numpy(x)[None].to(device), torch.from_numpy(flags)[None].to(device)))[0][0]
        nats += F.cross_entropy(logits.float(), torch.from_numpy(doc[a:b].astype(np.int64)).to(device), reduction="sum").item()
        patches += int(flags.sum())
    return nats, patches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--dataset", default=None)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_blt(args.run, device)
    ps = patch_settings(cfg)
    dataset = args.dataset or cfg["dataset"]
    base = REPO_ROOT / "data" / "bytes" / dataset
    data, offsets = open_split(base, args.split)
    entropy = None
    if ps["mode"] == "entropy":
        entropy = np.load(base / f"entropy_{cfg['patching']['entropy_run']}" / f"{args.split}.npy", mmap_mode="r")
    count = len(offsets) - 1 if args.max_docs is None else min(args.max_docs, len(offsets) - 1)
    nats = bytes_ = patches = 0
    for i in range(count):
        s, e = offsets[i], offsets[i + 1]
        if e == s:
            continue
        h = None if entropy is None else entropy[s:e].astype(np.float32)
        dn, dp = score_doc(model, np.asarray(data[s:e]), h, ps, cfg["block"], device)
        nats, patches, bytes_ = nats + dn, patches + dp, bytes_ + int(e - s)
    result = {"run": args.run, "dataset": dataset, "split": args.split, "docs": count, "bytes": bytes_,
              "bpb": round(nats / math.log(2) / bytes_, 4), "patching": ps["mode"],
              "mean_patch_len": round(bytes_ / patches, 3),
              "global_cache_bytes_per_1000": round(1000 * patches / bytes_ * GLOBAL_CACHE_BYTES_PER_PATCH(cfg["model"]))}
    (REPO_ROOT / "runs" / args.run / (args.out or f"eval_{args.split}_{count}.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
