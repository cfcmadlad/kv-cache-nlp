import argparse
import json
import time
from pathlib import Path

from data_utils import check_not_truncated, read_jsonl_texts
from models.ngram import KneserNeyByteLM

TRAIN_SAMPLE_SIZE = 200


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["tinystories", "wikitext103"], required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--max-train-docs", type=int, default=5000)
    p.add_argument("--eval-max-docs", type=int, default=1000)
    p.add_argument("--out", type=Path, default=Path("checkpoints"))
    args = p.parse_args()
    if args.order < 2:
        p.error("--order must be >= 2")
    return args


def main():
    args = parse_args()
    train_path = args.data_dir / args.dataset / "train.jsonl"
    train_docs = [t.encode("utf-8") for t in read_jsonl_texts(train_path, args.max_train_docs)]
    check_not_truncated(train_path, len(train_docs), args.max_train_docs)
    val_docs = [t.encode("utf-8") for t in read_jsonl_texts(args.data_dir / args.dataset / "val.jsonl", args.eval_max_docs)]

    print(f"Training order-{args.order} Kneser-Ney byte LM on {len(train_docs)} docs...")
    t0 = time.time()
    model = KneserNeyByteLM(order=args.order)
    model.fit(train_docs)
    fit_time = time.time() - t0
    sample = train_docs[:TRAIN_SAMPLE_SIZE]
    train_bpb = model.bits_per_byte(sample)
    val_bpb = model.bits_per_byte(val_docs)
    print(f"Fit {fit_time:.1f}s | discounts {[round(d, 3) for d in model.discounts]} | "
          f"train BPB ({len(sample)} docs) {train_bpb:.3f} | val BPB {val_bpb:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"ngram_{args.dataset}_order{args.order}"
    model.save(args.out / f"{stem}.pkl")
    results = {"model": "kneser_ney_ngram", "dataset": args.dataset, "order": args.order,
               "train_docs": len(train_docs), "val_docs": len(val_docs),
               "discounts": [round(d, 4) for d in model.discounts], "train_time_sec": round(fit_time, 1),
               "train_bpb_sample": round(train_bpb, 4), "val_bpb": round(val_bpb, 4)}
    (args.out / f"{stem}_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {args.out / stem}.pkl and _results.json")


if __name__ == "__main__":
    main()
