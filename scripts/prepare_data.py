"""Step 1 of the pipeline: download, clean, deduplicate and split TinyStories and
WikiText-103 into data/processed/<dataset>/<split>.jsonl plus a stats.json per dataset.

TinyStories: one story per row. --max-docs caps every split (the reported data used
--max-docs 30000: 30,000 train stories, and all 21,990 validation stories since that
split is smaller than the cap).
WikiText-103: articles are rebuilt from the raw line export (split on "= Title =" headers)
and the @-@ / @,@ / @.@ placeholders are turned back into real punctuation.
All text is Unicode NFC normalised and exact duplicates (SHA-256) are dropped per split.

Usage:
    python scripts/prepare_data.py --dataset tinystories --max-docs 30000
    python scripts/prepare_data.py --dataset wikitext103
"""

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset

WIKITEXT_HEADER_RE = re.compile(r"^\s*=\s([^=].*?)\s=\s*$")
WIKITEXT_ARTIFACT_RE = re.compile(r" @([-,.])@ ")

TINYSTORIES_SPLITS = {"train": "train", "val": "validation"}
WIKITEXT_SPLITS = {"train": "train", "val": "validation", "test": "test"}


def normalize(text: str) -> str:
    """Unicode NFC normalisation plus whitespace trimming."""
    return unicodedata.normalize("NFC", text).strip()


def clean_wikitext_artifacts(text: str) -> str:
    """Undo WikiText's `@-@`/`@,@`/`@.@` placeholder tokens (an artifact of the
    original extraction tooling, not natural text)."""
    return WIKITEXT_ARTIFACT_RE.sub(r"\1", text)


def dedup(docs: list) -> tuple:
    """Drop exact duplicates (by SHA-256), keeping first occurrences; return (kept, removed count)."""
    seen = set()
    kept = []
    for doc in docs:
        key = hashlib.sha256(doc.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        kept.append(doc)
    return kept, len(docs) - len(kept)


def load_tinystories_split(split: str, max_docs) -> list:
    """One document per story, the first max_docs of the split (all if None)."""
    hf_split = TINYSTORIES_SPLITS[split]
    slice_expr = f"{hf_split}[:{max_docs}]" if max_docs is not None else hf_split
    ds = load_dataset("roneneldan/TinyStories", split=slice_expr)
    docs = [normalize(row["text"]) for row in ds]
    return [d for d in docs if d]


def load_wikitext103_split(split: str, max_docs) -> list:
    """Reconstruct per-article documents from WikiText's raw lines, split on
    `= Title =` headers. Streams when max_docs is set so a small sample doesn't
    require materializing the entire (multi-GB) split first."""
    hf_split = WIKITEXT_SPLITS[split]
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=hf_split,
                       streaming=max_docs is not None)

    docs = []
    current_lines = []

    def flush():
        # Require at least one body line: a header with nothing under it
        # (e.g. two headers in a row) isn't a real document.
        if len(current_lines) > 1:
            doc = normalize("\n".join(current_lines))
            if doc:
                docs.append(doc)

    for row in ds:
        line = clean_wikitext_artifacts(row["text"])
        if WIKITEXT_HEADER_RE.match(line):
            flush()
            current_lines = [line.strip()]
        elif line.strip():
            current_lines.append(line.strip())
        if max_docs is not None and len(docs) >= max_docs:
            break
    flush()
    return docs[:max_docs] if max_docs is not None else docs


LOADERS = {
    "tinystories": load_tinystories_split,
    "wikitext103": load_wikitext103_split,
}

SPLITS = {
    "tinystories": ["train", "val"],
    "wikitext103": ["train", "val", "test"],
}


def write_split(name: str, split: str, docs: list, dup_removed: int, out_dir: Path) -> dict:
    """Write <out_dir>/<name>/<split>.jsonl and return that split's statistics."""
    dataset_dir = out_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = dataset_dir / f"{split}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for i, doc in enumerate(docs):
            f.write(json.dumps({"id": f"{name}-{split}-{i}", "text": doc}) + "\n")

    num_bytes = sum(len(d.encode("utf-8")) for d in docs)
    stats = {
        "dataset": name,
        "split": split,
        "num_docs": len(docs),
        "num_bytes": num_bytes,
        "avg_doc_bytes": round(num_bytes / len(docs), 1) if docs else 0,
        "duplicates_removed": dup_removed,
    }
    print(f"[{name}/{split}] docs={stats['num_docs']} bytes={stats['num_bytes']} "
          f"dedup_removed={dup_removed} -> {jsonl_path}")
    return stats


def build(name: str, out_dir: Path, max_docs) -> None:
    """Load, deduplicate and write every split of one dataset, then its stats.json."""
    all_stats = {}
    for split in SPLITS[name]:
        raw_docs = LOADERS[name](split, max_docs)
        docs, dup_removed = dedup(raw_docs)
        all_stats[split] = write_split(name, split, docs, dup_removed, out_dir)

    stats_path = out_dir / name / "stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["tinystories", "wikitext103", "all"], required=True)
    parser.add_argument("--out", type=Path, default=Path("data/processed"), help="output folder")
    parser.add_argument("--max-docs", type=int, default=None,
                         help="cap on documents per split (default: no cap)")
    args = parser.parse_args()

    names = ["tinystories", "wikitext103"] if args.dataset == "all" else [args.dataset]
    for name in names:
        build(name, args.out, args.max_docs)


if __name__ == "__main__":
    main()
