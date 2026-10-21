import argparse

import numpy as np
import torch

from data_utils import REPO_ROOT
from kvc.bytedata import open_split
from kvc.patching import batched_entropies
from models.entropy import EntropyLM


def load_entropy_model(run: str, device: str):
    ckpt = torch.load(REPO_ROOT / "runs" / run / "model.pt", map_location=device, weights_only=True)
    m = ckpt["config"]["model"]
    model = EntropyLM(m["n_layer"], m["n_head"], m["dim"], m["window"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["config"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--chunk-docs", type=int, default=2000)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_entropy_model(args.run, device)
    dataset = args.dataset or cfg["dataset"]
    base = REPO_ROOT / "data" / "bytes" / dataset
    out_dir = base / f"entropy_{args.run}"
    out_dir.mkdir(exist_ok=True)
    for split in args.splits:
        data, offsets = open_split(base, split)
        cache = np.lib.format.open_memmap(out_dir / f"{split}.npy", mode="w+", dtype=np.float16, shape=(len(data),))
        for a in range(0, len(offsets) - 1, args.chunk_docs):
            b = min(a + args.chunk_docs, len(offsets) - 1)
            docs = [bytes(data[offsets[i]:offsets[i + 1]]) for i in range(a, b)]
            for i, h in zip(range(a, b), batched_entropies(model, [d or b"\0" for d in docs], device)):
                cache[offsets[i]:offsets[i + 1]] = h[:offsets[i + 1] - offsets[i]]
        cache.flush()
        print(f"{dataset}/{split}: {len(data):,} bytes cached")


if __name__ == "__main__":
    main()
