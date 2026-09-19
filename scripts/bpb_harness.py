"""Build a bits-per-byte comparison table across both baselines.

Reads every checkpoints/*_results.json (written by train_ngram.py / train_bpe_transformer.py),
groups by dataset, and prints + saves a comparison table. Both baselines write a flat schema
sharing model/dataset/train_docs/val_docs/train_time_sec/train_bpb_sample/val_bpb -- see the
"Baseline results schema" section of the README.

Usage:
    python scripts/bpb_harness.py
    python scripts/bpb_harness.py --checkpoints-dir checkpoints --out eval
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

SHARED_COLUMNS = ["model", "train_docs", "val_docs", "train_time_sec", "train_bpb_sample", "val_bpb"]

# Extra, model-specific context worth showing alongside the shared columns.
EXTRA_COLUMNS = {
    "kneser_ney_ngram": ["order", "discounts"],
    "bpe_transformer": ["num_params", "n_layer", "n_embd", "block_size"],
}


def load_results(checkpoints_dir: Path) -> list:
    """One result dict per (model, dataset) pair. If a sweep left multiple results.json
    for the same pair, keep the most recently written one and say so -- silently picking
    an arbitrary file would make the comparison table nondeterministic. A single bad file
    (e.g. left half-written by an interrupted run) is skipped with a warning rather than
    taking down the whole comparison."""
    by_key = defaultdict(list)
    for path in sorted(checkpoints_dir.glob("*_results.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["_path"] = path
            by_key[(data["model"], data["dataset"])].append(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping {path}: {e}")

    results = []
    for (model, dataset), entries in by_key.items():
        if len(entries) > 1:
            entries.sort(key=lambda d: d["_path"].stat().st_mtime)
            print(f"Note: {len(entries)} results.json found for {model}/{dataset}, "
                  f"using the most recent: {entries[-1]['_path'].name}")
        results.append(entries[-1])
    return results


def _format_cell(value) -> str:
    """Render one table cell -- comma-join lists (e.g. n-gram's discounts) instead of
    Python's bracketed repr, and escape '|' so it can't be mistaken for a column separator."""
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    return str(value).replace("|", "\\|")


def format_table(results: list) -> str:
    datasets = sorted({r["dataset"] for r in results})
    lines = []
    for dataset in datasets:
        rows = [r for r in results if r["dataset"] == dataset]
        columns = list(SHARED_COLUMNS)
        for row in rows:
            for col in EXTRA_COLUMNS.get(row["model"], []):
                if col not in columns:
                    columns.append(col)

        lines.append(f"### {dataset}\n")
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join(["---"] * len(columns)) + "|")
        for row in rows:
            cells = [_format_cell(row.get(col, "—")) for col in columns]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--out", type=Path, default=Path("eval"))
    args = parser.parse_args()

    results = load_results(args.checkpoints_dir)
    if not results:
        print(f"No *_results.json found under {args.checkpoints_dir} -- train both baselines first.")
        return

    table = format_table(results)
    print(table)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "bpb_comparison.md"
    out_path.write_text(f"# BPB comparison\n\n{table}", encoding="utf-8")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
