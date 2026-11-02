import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import REPO_ROOT
from eval_blt import load_blt
from eval_extras import load_entropy_model
from kvc.bytedata import open_split
from kvc.patching import doc_entropy_and_nats, patch_flags
from models.entropy import BOS
from train_blt import patch_settings


def set_windows(blt, ent, local: int, entropy: int):
    for blk in list(blt.enc) + list(blt.dec):
        blk.attn.window = local
    for blk in ent.blocks:
        blk.attn.window = entropy


@torch.no_grad()
def bpb(blt, ent, cfg, args, device):
    ps = patch_settings(cfg)
    base = REPO_ROOT / "data" / "bytes" / cfg["dataset"]
    data, offsets = open_split(base, args.split)
    nats, nbytes, patches = 0.0, 0, 0
    for di in range(min(args.max_docs, len(offsets) - 1)):
        doc = np.asarray(data[offsets[di]:offsets[di + 1]])
        h, _ = doc_entropy_and_nats(ent, bytes(doc.astype(np.uint8)), device)
        for a in range(0, len(doc), cfg["block"]):
            b = min(a + cfg["block"], len(doc))
            x = np.empty(b - a, dtype=np.int64)
            x[0] = BOS if a == 0 else doc[a - 1]
            x[1:] = doc[a:b - 1]
            flags = patch_flags(h[a:b], np.arange(a, b), np.zeros(b - a, dtype=bool), ps["mode"], ps["theta"], ps["max_len"], ps["stride"])
            logits = blt((torch.from_numpy(x)[None].to(device), torch.from_numpy(flags)[None].to(device)))[0][0]
            nats += F.cross_entropy(logits.float(), torch.from_numpy(doc[a:b].astype(np.int64)).to(device), reduction="sum").item()
            patches += int(flags.sum()) + 1
        nbytes += len(doc)
    return nats / np.log(2) / nbytes, nbytes / patches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-docs", type=int, default=100)
    p.add_argument("--local", nargs="+", type=int, default=[256, 128, 64])
    p.add_argument("--entropy", nargs="+", type=int, default=[512, 256, 128])
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blt, cfg = load_blt(args.run, device)
    ent = load_entropy_model(cfg, device)
    m = cfg["model"]
    rows = []
    for lw in args.local:
        for ew in args.entropy:
            set_windows(blt, ent, lw, ew)
            b, per_patch = bpb(blt, ent, cfg, args, device)
            local_kb = (m["enc_layers"] + m["dec_layers"]) * 2 * m["local_dim"] * lw * 2 / 1000
            ent_mb = 4 * 2 * 256 * ew * 2 / 1e6
            rows.append({"local_window": lw, "entropy_window": ew, "bpb": b, "bytes_per_patch": per_patch, "local_cache_kb": local_kb, "entropy_cache_mb": ent_mb})
            print(rows[-1], flush=True)
    out = REPO_ROOT / "runs" / args.run / f"window_{args.split}_{args.max_docs}.json"
    out.write_text(json.dumps({"run": args.run, "docs": args.max_docs, "rows": rows}), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
