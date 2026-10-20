import argparse
import math
import statistics
from pathlib import Path

from data_utils import load_ngram, load_transformer, read_jsonl_texts, safe_print
from train_bpe_transformer import per_document_bpb

PREVIEW_CHARS = 200


def ngram_per_doc_bpb(model, texts: list) -> list:
    return [-model.log2_prob_doc(d) / len(d) if d else math.nan for d in (t.encode("utf-8") for t in texts)]


def distribution_stats(scores: list) -> dict:
    clean = [s for s in scores if not math.isnan(s)]
    if not clean:
        return dict.fromkeys(["mean", "median", "stdev", "min", "max"], math.nan)
    return {"mean": statistics.mean(clean), "median": statistics.median(clean),
            "stdev": statistics.stdev(clean) if len(clean) > 1 else 0.0,
            "min": min(clean), "max": max(clean)}


def correlation(a: list, b: list) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    try:
        return statistics.correlation(*zip(*pairs))
    except (statistics.StatisticsError, TypeError, ValueError):
        return math.nan


def length_buckets(texts: list, scores: list, num_buckets: int = 3) -> list:
    rows = sorted((len(t.encode("utf-8")), s) for t, s in zip(texts, scores) if not math.isnan(s))
    bounds = [round(i * len(rows) / num_buckets) for i in range(num_buckets + 1)]
    return [(rows[a][0], rows[b - 1][0], b - a, statistics.mean(s for _, s in rows[a:b]))
            for a, b in zip(bounds, bounds[1:]) if b > a]


def preview(text: str) -> str:
    text = text.replace("\n", " ").strip()
    return text[:PREVIEW_CHARS] + ("..." if len(text) > PREVIEW_CHARS else "")


def format_report(dataset: str, ngram: list, tf: list, texts: list, k: int) -> str:
    lines = [f"# Error analysis: {dataset}\n", f"Docs scored: {len(texts)}\n",
             "| | n-gram | transformer |", "|---|---|---|"]
    ns, ts = distribution_stats(ngram), distribution_stats(tf)
    lines += [f"| {key} | {ns[key]:.4f} | {ts[key]:.4f} |" for key in ns]
    lines.append(f"\nPer-doc BPB correlation between models: {correlation(ngram, tf):.4f}\n")
    for name, scores in [("n-gram", ngram), ("transformer", tf)]:
        lines.append(f"## {name}: BPB by document length")
        lines += [f"- {lo}-{hi} bytes ({n} docs): {m:.4f}" for lo, hi, n, m in length_buckets(texts, scores)]
        ranked = sorted((s, i) for i, s in enumerate(scores) if not math.isnan(s))
        n = min(k, len(ranked) // 2)
        lines.append(f"\n## {name}: worst docs")
        lines += [f"- {s:.4f}: {preview(texts[i])}" for s, i in ranked[::-1][:n]]
        lines.append(f"\n## {name}: best docs")
        lines += [f"- {s:.4f}: {preview(texts[i])}" for s, i in ranked[:n]]
        lines.append("")
    gains = sorted(((a - b, i) for i, (a, b) in enumerate(zip(ngram, tf))
                    if not (math.isnan(a) or math.isnan(b))), reverse=True)
    lines.append("## Largest transformer gain over n-gram")
    lines += [f"- {g:.4f}: {preview(texts[i])}" for g, i in gains[:k]]
    lines.append("\n## Smallest transformer gain over n-gram")
    lines += [f"- {g:.4f}: {preview(texts[i])}" for g, i in gains[-k:]]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["tinystories", "wikitext103"], required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--out", type=Path, default=Path("results/part2"))
    p.add_argument("--eval-max-docs", type=int, default=300)
    p.add_argument("--num-examples", type=int, default=5)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    texts = read_jsonl_texts(args.data_dir / args.dataset / "val.jsonl", args.eval_max_docs)
    ngram = load_ngram(args.checkpoints_dir, args.dataset)
    model, tokenizer, config = load_transformer(args.checkpoints_dir, args.dataset, args.device)
    print(f"Scoring {len(texts)} docs with both models...")
    report = format_report(args.dataset, ngram_per_doc_bpb(ngram, texts),
                           per_document_bpb(model, texts, tokenizer, config["block_size"], args.device),
                           texts, args.num_examples)
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"error_analysis_{args.dataset}.md"
    out_path.write_text(report, encoding="utf-8")
    safe_print(report)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
