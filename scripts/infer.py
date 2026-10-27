import argparse
import sys
from pathlib import Path

import torch

from cache_demo import analyse, load_pair
from data_utils import safe_print
from demo import MIN_BUDGET, ensure_part3_assets
from demo_part3 import stream_compare


def main():
    ap = argparse.ArgumentParser(description="Score a text with a trained byte-level model under each KV cache policy.")
    ap.add_argument("--dataset", choices=["tinystories", "wikitext103"], default="tinystories", help="which trained model to load")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--file", type=Path)
    ap.add_argument("--budget", type=int, default=32, help=f"global cache size in patches, at least {MIN_BUDGET}")
    ap.add_argument("--stream", action="store_true", help="decode one byte at a time with real eviction instead of one masked forward pass")
    ap.add_argument("--max-bytes", type=int, default=4000, help="bytes to decode with --stream (a plain run scores the first 4,096)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.budget < MIN_BUDGET:
        sys.exit(f"--budget must be at least {MIN_BUDGET}: one sink patch and the eight newest are always kept.")
    if not ensure_part3_assets():
        sys.exit("Could not get the trained models; see the message above.")
    text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    model, entropy, cfg = load_pair(args.dataset, args.device)
    if args.stream:
        result = stream_compare(text, model, entropy, cfg, args.budget, args.device, args.max_bytes)
        print(f"{result['bytes']} bytes in {result['patches']} patches, decoded one byte at a time with real cache eviction.\n")
        print(f"{'Cache setting':<36}{'BPB':>8}{'Final KV cache':>16}{'Seconds':>10}")
        for name, bpb, size, seconds in result["rows"]:
            print(f"{name:<36}{bpb:>8.3f}{size / 1000:>13.1f} KB{seconds:>9.1f}")
        return
    result = analyse(text, model, entropy, cfg, args.budget, args.device)
    safe_print(f"{result['bytes']} bytes in {result['patches']} patches ({result['mean_patch_len']:.1f} bytes per patch). Patch boundaries:\n{result['marked'][:600]}\n")
    print(f"{'Cache setting':<36}{'BPB':>8}{'Mean KV cache':>16}")
    for name, bpb, size in result["rows"]:
        print(f"{name:<36}{bpb:>8.3f}{size / 1000:>13.1f} KB")


if __name__ == "__main__":
    main()
