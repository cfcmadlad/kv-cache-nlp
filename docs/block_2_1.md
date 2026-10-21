# Block 2.1: byte datasets

Rubric: Results and error analysis (2), Demo and viva (20).

## What exists

| Path | Purpose |
|---|---|
| `kvc/bytedata.py` | Writes a split as one uint8 `.bin` plus an int64 offsets array; reads documents back; round-trip sampling |
| `scripts/prepare_bytes.py` | TinyStories and WikiText-103 from `data/processed/*.jsonl` to `data/bytes/<dataset>/` |
| `scripts/prepare_code.py` | Python, Java, JavaScript from `bigcode/the-stack-smol` to `data/bytes/code/` |
| `tests/test_bytedata.py` | Round trip with multi-byte text and empty documents, CLI build and verify, code split rules |

Layout per dataset: `train.bin`, `val.bin`, `test.bin` (WikiText and code; TinyStories has no test split), and `<split>_offsets.npy` with N+1 entries so document i is bytes `offsets[i]:offsets[i+1]`. The spec said a single `offsets.npy`; one file per split was needed because splits have different document counts. Documents are stored without separators; the offsets are the boundaries.

## Code split

Files are filtered to 200 to 100,000 bytes and no null bytes, deduplicated by SHA-256 of the content, then assigned to a split by the first 32 bits of that hash modulo 100 (below 5 test, below 10 val, else train). Identical content always lands in the same split, and exact duplicates are dropped before splitting, so no hash appears in two splits. `manifest.jsonl` records split, language and hash for every file, and `--verify` recomputes the hashes from the binary files. The size limits and percentages are command-line arguments. Near-duplicates (copied files with small edits) are not removed; this could make validation slightly easier than true held-out code.

## Run

```bash
python scripts/prepare_bytes.py
python scripts/prepare_bytes.py --verify
python scripts/prepare_code.py
python scripts/prepare_code.py --verify
```

`prepare_code.py` needs `pip install datasets` and a Hugging Face login with the dataset terms accepted, because the-stack-smol is gated. `--per-lang N` streams only N files per language for a quick trial.

## Viva notes

Why bytes in one file? Memory-mapping gives random access to any window without loading 526 MB of WikiText, and the offsets let evaluation score whole documents.
Why split code by hash, not randomly? The split is reproducible from the content alone, and a file and its duplicate cannot be on both sides.
Why not remove near-duplicates? It needs fuzzy hashing and was out of the block's scope; stated as a limitation.
