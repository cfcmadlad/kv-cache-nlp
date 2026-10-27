import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

from data_utils import REPO_ROOT
from kvc.bytedata import open_split, read_doc, write_split


def load_books(repo: str, split: str, limit):
    from datasets import load_dataset
    for i, row in enumerate(load_dataset(repo, split=split, streaming=True)):
        if limit is not None and i >= limit:
            break
        yield row["short_book_title"], row["text"]


def build(books, out: Path, split: str) -> None:
    titles, texts = [], []
    for title, text in books:
        titles.append(title)
        texts.append(unicodedata.normalize("NFC", text))
    stats = {split: write_split(texts, out, split)}
    stats[split]["titles"] = titles
    stats[split]["sha256"] = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
    (out / f"stats_{split}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[pg19/{split}] books={stats[split]['num_docs']:,} bytes={stats[split]['num_bytes']:,}")


def verify(out: Path, split: str) -> None:
    stats = json.loads((out / f"stats_{split}.json").read_text(encoding="utf-8"))[split]
    data, offsets = open_split(out, split)
    assert (len(offsets) - 1, len(data)) == (stats["num_docs"], stats["num_bytes"]), "counts differ from stats"
    assert len(set(stats["sha256"])) == len(stats["sha256"]), "duplicate books"
    for i in range(len(offsets) - 1):
        assert hashlib.sha256(read_doc(data, offsets, i).encode("utf-8")).hexdigest() == stats["sha256"][i], f"book {i}: hash mismatch"
    print(f"[pg19/{split}] {len(offsets) - 1} books: counts match, hashes match, no duplicates")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="emozilla/pg19")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "bytes" / "pg19")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.verify:
        verify(args.out, args.split)
    else:
        build(load_books(args.repo, args.split, args.limit), args.out, args.split)


if __name__ == "__main__":
    main()
