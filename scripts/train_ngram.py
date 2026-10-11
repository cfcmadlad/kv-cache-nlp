"""Step 2a: train and evaluate the byte-level Kneser-Ney n-gram baseline.

Reads data/processed/<dataset>/{train,val}.jsonl, fits models/ngram.py on the first
--max-train-docs training documents, scores the first --eval-max-docs validation
documents in bits per byte, and writes checkpoints/ngram_<dataset>_order<K>.pkl plus a
_results.json that bpb_harness.py reads.

Usage (the reported runs):
    python scripts/train_ngram.py --dataset tinystories --order 4 --max-train-docs 25000 --eval-max-docs 2000
    python scripts/train_ngram.py --dataset wikitext103 --order 4 --max-train-docs 4500 --eval-max-docs 60
"""

import argparse
import json
import time
from pathlib import Path

from data_utils import check_not_truncated, read_jsonl_texts  # also puts repo root on sys.path
from models.ngram import KneserNeyByteLM

TRAIN_SAMPLE_SIZE = 200  # train-set BPB is reported on this many docs only, to keep it fast


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["tinystories", "wikitext103"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="processed data folder")
    parser.add_argument("--order", type=int, default=4, help="n-gram order (4 = 3 bytes of context)")
    parser.add_argument("--max-train-docs", type=int, default=5000,
                         help="training documents to count (pure-Python counting does not scale to the full corpus)")
    parser.add_argument("--eval-max-docs", type=int, default=1000, help="validation documents to score")
    parser.add_argument("--out", type=Path, default=Path("checkpoints"), help="where to write the model and results")
    args = parser.parse_args()
    if args.order < 2:
        parser.error("--order must be >= 2 (Kneser-Ney needs bigram continuation counts)")
    return args


def save_outputs(model: KneserNeyByteLM, args, metrics: dict) -> None:
    """Write checkpoints/ngram_<dataset>_order<K>.pkl and the matching _results.json."""
    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / f"ngram_{args.dataset}_order{args.order}.pkl"
    model.save(model_path)
    print(f"Saved model to {model_path}")

    # Flat keys, shared with the transformer results file, so bpb_harness.py reads both the same way.
    results = {
        "model": "kneser_ney_ngram",
        "dataset": args.dataset,
        "order": args.order,
        **metrics,
    }
    results_path = args.out / f"ngram_{args.dataset}_order{args.order}_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {results_path}")


def main():
    args = parse_args()
    train_path = args.data_dir / args.dataset / "train.jsonl"
    val_path = args.data_dir / args.dataset / "val.jsonl"

    print(f"Loading train docs from {train_path} (max {args.max_train_docs})...")
    train_docs = [t.encode("utf-8") for t in read_jsonl_texts(train_path, args.max_train_docs)]
    check_not_truncated(train_path, len(train_docs), args.max_train_docs)
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

    save_outputs(model, args, {
        "train_docs": len(train_docs),
        "val_docs": len(val_docs),
        "discounts": [round(d, 4) for d in model.discounts],
        "train_time_sec": round(fit_time, 1),
        "train_bpb_sample": round(train_bpb, 4),
        "val_bpb": round(val_bpb, 4),
    })


if __name__ == "__main__":
    main()
