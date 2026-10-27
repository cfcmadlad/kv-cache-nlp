import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import REPO_ROOT
from eval_blt import load_blt
from kvc.bytedata import open_split
from kvc.cache import Policy
from kvc.patching import patch_flags
from models.entropy import BOS
from train_blt import patch_settings


@torch.no_grad()
def stream_nats(model, tokens, targets, flags, h, policy, device, state=None):
    state = {} if state is None else state
    state.update(model.init_state())
    total = 0.0
    for i in range(len(tokens)):
        logits = model.step(state, tokens[i:i + 1], bool(flags[i]), float(h[i]), policy)
        total += F.cross_entropy(logits.float(), targets[i:i + 1], reduction="sum").item()
    return total


@torch.no_grad()
def reset_nats(model, doc, h, ps, block, device):
    total = 0.0
    for a in range(0, len(doc), block):
        b = min(a + block, len(doc))
        x = np.empty(b - a, dtype=np.int64)
        x[0] = BOS if a == 0 else doc[a - 1]
        x[1:] = doc[a:b - 1]
        f = patch_flags(h[a:b] if ps["mode"] == "entropy" else None, np.arange(a, b), np.zeros(b - a, dtype=bool), ps["mode"], ps["theta"], ps["max_len"], ps["stride"])
        logits = model((torch.from_numpy(x)[None].to(device), torch.from_numpy(f)[None].to(device)))[0][0]
        total += F.cross_entropy(logits.float(), torch.from_numpy(doc[a:b].astype(np.int64)).to(device), reduction="sum").item()
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--dataset", default="pg19")
    p.add_argument("--split", default="test")
    p.add_argument("--books", nargs=2, type=int, default=[0, 20])
    p.add_argument("--max-bytes", type=int, default=100_000)
    p.add_argument("--budgets", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--policies", nargs="+", default=["recency", "random", "entropy"])
    p.add_argument("--sink", type=int, default=1)
    p.add_argument("--recent", type=int, default=8)
    p.add_argument("--reset-block", type=int, default=4096)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_blt(args.run, device)
    ps = patch_settings(cfg)
    base = REPO_ROOT / "data" / "bytes" / args.dataset
    data, offsets = open_split(base, args.split)
    entropy = np.load(base / f"entropy_{cfg['patching']['entropy_run']}" / f"{args.split}.npy", mmap_mode="r")
    variants = ["full|0", f"reset{args.reset_block}|0"] + [f"{m}|{b}" for m in args.policies for b in args.budgets]
    nats = {v: [] for v in variants}
    sizes, books = [], list(range(args.books[0], min(args.books[1], len(offsets) - 1)))
    for i in books:
        s = int(offsets[i])
        e = min(int(offsets[i + 1]), s + args.max_bytes)
        doc, h = np.asarray(data[s:e]), entropy[s:e].astype(np.float32)
        n = len(doc)
        sizes.append(n)
        flags = patch_flags(h if ps["mode"] == "entropy" else None, np.arange(n), np.zeros(n, dtype=bool), ps["mode"], ps["theta"], ps["max_len"], ps["stride"])
        tokens = torch.from_numpy(np.concatenate([[BOS], doc[:-1]]).astype(np.int64)).to(device)
        targets = torch.from_numpy(doc.astype(np.int64)).to(device)
        nats[f"reset{args.reset_block}|0"].append(reset_nats(model, doc, h, ps, args.reset_block, device))
        nats["full|0"].append(stream_nats(model, tokens, targets, flags, h, None, device))
        for m in args.policies:
            for b in args.budgets:
                pol = Policy(m, b, args.sink, args.recent, "mean", 0)
                nats[f"{m}|{b}"].append(stream_nats(model, tokens, targets, flags, h, pol, device))
        print(i, n, {v: round(nats[v][-1] / np.log(2) / n, 4) for v in variants}, flush=True)
    result = {"run": args.run, "dataset": args.dataset, "split": args.split, "books": books, "bytes_per_doc": sizes,
              "settings": {"max_bytes": args.max_bytes, "sink": args.sink, "recent": args.recent, "reset_block": args.reset_block},
              "variants": {v: {"nats": nats[v]} for v in variants}}
    out = REPO_ROOT / "runs" / args.run / (args.out or f"stream_{args.dataset}_{books[0]}_{books[-1] + 1}.json")
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
