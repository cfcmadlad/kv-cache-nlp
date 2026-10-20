import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("prepare_data.py needs the datasets package: pip install datasets")

WIKITEXT_HEADER_RE = re.compile(r"^\s*=\s([^=].*?)\s=\s*$")
WIKITEXT_ARTIFACT_RE = re.compile(r" @([-,.])@ ")
HF_SPLITS = {
    "tinystories": {"train": "train", "val": "validation"},
    "wikitext103": {"train": "train", "val": "validation", "test": "test"},
}


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def dedup(docs: list) -> tuple:
    seen, kept = set(), []
    for doc in docs:
        key = hashlib.sha256(doc.encode("utf-8")).digest()
        if key not in seen:
            seen.add(key)
            kept.append(doc)
    return kept, len(docs) - len(kept)


def load_tinystories(hf_split: str, max_docs) -> list:
    ds = load_dataset("roneneldan/TinyStories", split=f"{hf_split}[:{max_docs}]" if max_docs is not None else hf_split)
    return [d for d in (normalize(row["text"]) for row in ds) if d]


def load_wikitext103(hf_split: str, max_docs) -> list:
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=hf_split, streaming=max_docs is not None)
    docs, lines = [], []

    def flush():
        if len(lines) > 1 and (doc := normalize("\n".join(lines))):
            docs.append(doc)

    for row in ds:
        line = WIKITEXT_ARTIFACT_RE.sub(r"\1", row["text"])
        if WIKITEXT_HEADER_RE.match(line):
            flush()
            lines = [line.strip()]
        elif line.strip():
            lines.append(line.strip())
        if max_docs is not None and len(docs) >= max_docs:
            break
    flush()
    return docs[:max_docs] if max_docs is not None else docs


LOADERS = {"tinystories": load_tinystories, "wikitext103": load_wikitext103}


def build(name: str, out_dir: Path, max_docs) -> None:
    dataset_dir = out_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    for split, hf_split in HF_SPLITS[name].items():
        docs, removed = dedup(LOADERS[name](hf_split, max_docs))
        path = dataset_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for i, doc in enumerate(docs):
                f.write(json.dumps({"id": f"{name}-{split}-{i}", "text": doc}) + "\n")
        num_bytes = sum(len(d.encode("utf-8")) for d in docs)
        all_stats[split] = {"dataset": name, "split": split, "num_docs": len(docs), "num_bytes": num_bytes,
                            "avg_doc_bytes": round(num_bytes / len(docs), 1) if docs else 0,
                            "duplicates_removed": removed}
        print(f"[{name}/{split}] docs={len(docs)} bytes={num_bytes} dedup_removed={removed} -> {path}")
    (dataset_dir / "stats.json").write_text(json.dumps(all_stats, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["tinystories", "wikitext103", "all"], required=True)
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--max-docs", type=int, default=None)
    args = p.parse_args()
    for name in (LOADERS if args.dataset == "all" else [args.dataset]):
        build(name, args.out, args.max_docs)


if __name__ == "__main__":
    main()
