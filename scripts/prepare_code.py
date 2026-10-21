import argparse
import hashlib
import json
from pathlib import Path

from data_utils import REPO_ROOT
from kvc.bytedata import open_split, read_doc, write_split

LANGS = ["python", "java", "javascript"]
SPLITS = ["train", "val", "test"]


def file_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def assign_split(digest: str, val_pct: int, test_pct: int) -> str:
    bucket = int(digest[:8], 16) % 100
    if bucket < test_pct:
        return "test"
    return "val" if bucket < test_pct + val_pct else "train"


def split_files(records, min_bytes: int, max_bytes: int, val_pct: int, test_pct: int) -> dict:
    seen, splits, dropped = set(), {s: [] for s in SPLITS}, {"size": 0, "duplicate": 0, "null": 0}
    for lang, content in records:
        size = len(content.encode("utf-8"))
        if not min_bytes <= size <= max_bytes:
            dropped["size"] += 1
        elif "\x00" in content:
            dropped["null"] += 1
        elif (digest := file_hash(content)) in seen:
            dropped["duplicate"] += 1
        else:
            seen.add(digest)
            splits[assign_split(digest, val_pct, test_pct)].append((lang, digest, content))
    return {"splits": splits, "dropped": dropped}


def load_stack(repo: str, langs: list, per_lang: int):
    from datasets import load_dataset
    for lang in langs:
        ds = load_dataset(repo, data_dir=f"data/{lang}", split="train", streaming=per_lang is not None)
        for i, row in enumerate(ds):
            if per_lang is not None and i >= per_lang:
                break
            yield lang, row["content"]


def build(records, out: Path, args) -> None:
    result = split_files(records, args.min_bytes, args.max_bytes, args.val_pct, args.test_pct)
    stats, manifest = {}, []
    for split, files in result["splits"].items():
        stats[split] = write_split((c for _, _, c in files), out, split)
        stats[split]["languages"] = {l: sum(1 for x, _, _ in files if x == l) for l in LANGS}
        manifest += [{"split": split, "lang": l, "sha256": h} for l, h, _ in files]
        print(f"[code/{split}] docs={stats[split]['num_docs']:,} bytes={stats[split]['num_bytes']:,} {stats[split]['languages']}")
    stats["dropped"] = result["dropped"]
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n", encoding="utf-8")
    print("dropped", result["dropped"])


def verify(out: Path, val_pct: int, test_pct: int) -> None:
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    manifest = [json.loads(l) for l in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    hashes = [m["sha256"] for m in manifest]
    assert len(set(hashes)) == len(hashes), f"{len(hashes) - len(set(hashes))} duplicate file hashes across splits"
    pos = 0
    for split in SPLITS:
        data, offsets = open_split(out, split)
        assert (len(offsets) - 1, len(data)) == (stats[split]["num_docs"], stats[split]["num_bytes"]), f"{split}: counts differ from stats.json"
        for i in range(len(offsets) - 1):
            m = manifest[pos + i]
            text = read_doc(data, offsets, i)
            assert m["split"] == split and file_hash(text) == m["sha256"], f"{split} doc {i}: hash mismatch"
            assert assign_split(m["sha256"], val_pct, test_pct) == split, f"{split} doc {i}: wrong split for its hash"
        pos += len(offsets) - 1
        print(f"[code/{split}] {len(offsets) - 1:,} files: counts match stats.json, hashes match, split rule holds")
    print(f"no duplicate hashes across splits ({len(hashes):,} files)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="bigcode/the-stack-smol")
    p.add_argument("--langs", nargs="+", default=LANGS)
    p.add_argument("--per-lang", type=int, default=None)
    p.add_argument("--min-bytes", type=int, default=200)
    p.add_argument("--max-bytes", type=int, default=100_000)
    p.add_argument("--val-pct", type=int, default=5)
    p.add_argument("--test-pct", type=int, default=5)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "bytes" / "code")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    if args.verify:
        verify(args.out, args.val_pct, args.test_pct)
    else:
        build(load_stack(args.repo, args.langs, args.per_lang), args.out, args)


if __name__ == "__main__":
    main()
