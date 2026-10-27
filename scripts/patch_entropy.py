import argparse
import json
import math

import numpy as np
import torch

from data_utils import REPO_ROOT
from kvc.bytedata import open_split
from kvc.patching import doc_entropy_and_nats, starts_from_entropy, tune_threshold, word_start_stats
from models.entropy import EntropyLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--target", type=float, default=4.5)
    p.add_argument("--max-patch", type=int, default=16)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out", default="patching.json")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = REPO_ROOT / "runs" / args.run
    ckpt = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    cfg = ckpt["config"]
    m = cfg["model"]
    model = EntropyLM(m["n_layer"], m["n_head"], m["dim"], m["window"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    data, offsets = open_split(REPO_ROOT / "data" / "bytes" / cfg["dataset"], args.split)
    count = len(offsets) - 1 if args.max_docs is None else min(args.max_docs, len(offsets) - 1)
    docs = [bytes(data[offsets[i]:offsets[i + 1]]) for i in range(count)]
    docs = [d for d in docs if d]
    entropies, nats = [], 0.0
    for d in docs:
        h, n = doc_entropy_and_nats(model, d, device)
        entropies.append(h)
        nats += float(n.sum())
    total = sum(len(d) for d in docs)
    bpb = nats / math.log(2) / total
    threshold = args.threshold if args.threshold is not None else tune_threshold(entropies, args.target, args.max_patch)
    starts = [starts_from_entropy(h, threshold, args.max_patch) for h in entropies]
    lengths = np.concatenate([np.diff(np.append(s, len(h))) for s, h in zip(starts, entropies)])
    result = {"run": args.run, "dataset": cfg["dataset"], "split": args.split, "docs": len(docs), "bytes": total,
              "bpb": round(bpb, 4), "threshold": threshold, "max_patch": args.max_patch,
              "mean_patch_len": round(float(lengths.mean()), 3), "median_patch_len": float(np.median(lengths)),
              "mean_entropy_nats": round(float(np.concatenate(entropies).mean()), 4),
              **{k: round(v, 4) if isinstance(v, float) else v for k, v in word_start_stats(docs, starts).items()}}
    (run_dir / args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
