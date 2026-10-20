import argparse
import json
from collections import defaultdict
from pathlib import Path

SHARED_COLUMNS = ["model", "train_docs", "val_docs", "train_time_sec", "train_bpb_sample", "val_bpb"]
EXTRA_COLUMNS = {
    "kneser_ney_ngram": ["order", "discounts"],
    "bpe_transformer": ["num_params", "n_layer", "n_embd", "block_size"],
}


def load_results(checkpoints_dir: Path) -> list:
    latest = {}
    for path in sorted(checkpoints_dir.glob("*_results.json"), key=lambda p: p.stat().st_mtime):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            latest[(data["model"], data["dataset"])] = data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping {path}: {e}")
    return list(latest.values())


def _cell(value) -> str:
    if isinstance(value, list):
        value = ", ".join(map(str, value))
    return str(value).replace("|", "\\|")


def format_table(results: list) -> str:
    by_dataset = defaultdict(list)
    for r in results:
        by_dataset[r["dataset"]].append(r)
    lines = []
    for dataset in sorted(by_dataset):
        rows = by_dataset[dataset]
        columns = SHARED_COLUMNS + [c for r in rows for c in EXTRA_COLUMNS.get(r["model"], [])]
        columns = list(dict.fromkeys(columns))
        lines += [f"### {dataset}\n", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        lines += ["| " + " | ".join(_cell(r.get(c, "-")) for c in columns) + " |" for r in rows]
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--out", type=Path, help="optional markdown file to write the table to")
    args = p.parse_args()
    results = load_results(args.checkpoints_dir)
    if not results:
        print(f"No *_results.json in {args.checkpoints_dir}. Run scripts/fetch_assets.py or train a model.")
        return
    table = format_table(results)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(f"# BPB comparison\n\n{table}", encoding="utf-8")


if __name__ == "__main__":
    main()
