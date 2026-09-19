"""Train + evaluate the byte-level Kneser-Ney n-gram baseline.

Usage:
    python scripts/train_ngram.py --dataset tinystories --order 4 --max-train-docs 5000
"""

import argparse
import json
import time
from pathlib import Path

from data_utils import read_jsonl_texts  # also puts repo root on sys.path, see data_utils.py
from models.ngram import KneserNeyByteLM

TRAIN_SAMPLE_SIZE = 200  # docs used for the reported train-set BPB (train set itself can be huge)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["tinystories", "wikitext103"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--max-train-docs", type=int, default=5000,
                         help="Cap training docs -- pure-Python counting doesn't scale to the full corpus.")
    parser.add_argument("--eval-max-docs", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    if args.order < 2:
        parser.error("--order must be >= 2 (Kneser-Ney needs bigram continuation counts)")

    train_path = args.data_dir / args.dataset / "train.jsonl"
    val_path = args.data_dir / args.dataset / "val.jsonl"

    print(f"Loading train docs from {train_path} (max {args.max_train_docs})...")
    train_docs = [t.encode("utf-8") for t in read_jsonl_texts(train_path, args.max_train_docs)]
    print(f"Loading val docs from {val_path} (max {args.eval_max_docs})...")
    val_docs = [t.encode("utf-8") for t in read_jsonl_texts(val_path, args.eval_max_docs)]

    print(f"Training order-{args.order} Kneser-Ney byte LM on {len(train_docs)} docs...")
    t0 = time.time()
    model = KneserNeyByteLM(order=args.order)
    model.fit(train_docs)
    fit_time = time.time() - t0

    sample_size = min(TRAIN_SAMPLE_SIZE, len(train_docs))
    train_bpb = model.bits_per_byte(train_docs[:sample_size])
    val_bpb = model.bits_per_byte(val_docs)

    print(f"Fit time: {fit_time:.1f}s")
    print(f"Discounts per order: {[round(d, 3) for d in model.discounts]}")
    print(f"Train BPB ({sample_size}-doc sample): {train_bpb:.3f}")
    print(f"Val BPB: {val_bpb:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / f"ngram_{args.dataset}_order{args.order}.pkl"
    model.save(model_path)
    print(f"Saved model to {model_path}")

    results = {
        "model": "kneser_ney_ngram",
        "dataset": args.dataset,
        "order": args.order,
        "train_docs": len(train_docs),
        "val_docs": len(val_docs),
        "discounts": [round(d, 4) for d in model.discounts],
        "train_time_sec": round(fit_time, 1),
        "train_bpb_sample": round(train_bpb, 4),
        "val_bpb": round(val_bpb, 4),
    }
    results_path = args.out / f"ngram_{args.dataset}_order{args.order}_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
