import argparse
import json
from pathlib import Path

from data_utils import REPO_ROOT
from kvc.bytedata import open_split, read_jsonl, sample_roundtrip, write_split

SPLITS = {"tinystories": ["train", "val"], "wikitext103": ["train", "val", "test"]}


def build(dataset: str, src: Path, out: Path) -> None:
    for split in SPLITS[dataset]:
        stats = write_split(read_jsonl(src / f"{split}.jsonl"), out, split)
        print(f"[{dataset}/{split}] docs={stats['num_docs']:,} bytes={stats['num_bytes']:,}")


def verify(dataset: str, src: Path, out: Path, n: int, seed: int) -> None:
    expected = json.loads((src / "stats.json").read_text(encoding="utf-8"))
    for split in SPLITS[dataset]:
        data, offsets = open_split(out, split)
        want = expected[split]
        assert (len(offsets) - 1, len(data)) == (want["num_docs"], want["num_bytes"]), \
            f"{dataset}/{split}: got {len(offsets) - 1} docs, {len(data)} bytes; stats.json has {want['num_docs']}, {want['num_bytes']}"
        checked = sample_roundtrip(out, split, list(read_jsonl(src / f"{split}.jsonl")), n, seed)
        print(f"[{dataset}/{split}] bytes match stats.json ({len(data):,}); {checked} docs round-trip")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=[*SPLITS, "all"], default="all")
    p.add_argument("--src", type=Path, default=REPO_ROOT / "data" / "processed")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "bytes")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    for name in SPLITS if args.dataset == "all" else [args.dataset]:
        if args.verify:
            verify(name, args.src / name, args.out / name, args.samples, args.seed)
        else:
            build(name, args.src / name, args.out / name)


if __name__ == "__main__":
    main()
