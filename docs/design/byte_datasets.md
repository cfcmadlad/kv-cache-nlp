# Byte datasets

## What exists

| Path | Purpose |
|---|---|
| `kvc/bytedata.py` | Writes a split as one uint8 `.bin` plus an int64 offsets array; reads documents back; round-trip sampling |
| `scripts/prepare_bytes.py` | TinyStories and WikiText-103 from `data/processed/*.jsonl` to `data/bytes/<dataset>/` |
| `scripts/prepare_code.py` | Python, Java, JavaScript from `bigcode/the-stack-smol` to `data/bytes/code/` |
| `tests/test_bytedata.py` | Round trip with multi-byte text and empty documents, CLI build and verify, code split rules |

Each dataset keeps `train.bin`, `val.bin` and `test.bin` (WikiText and code; TinyStories has no test split), together with a `<split>_offsets.npy` of N+1 entries, so that document i is the stretch of bytes `offsets[i]:offsets[i+1]`. The specification had asked for a single `offsets.npy`; one file per split proved necessary, because the splits hold different numbers of documents. Documents are stored without separators: the offsets are the frontiers.

## Code split

Files are admitted if they measure 200 to 100,000 bytes and harbour no null byte; they are then deduplicated by the SHA-256 of their content and assigned to a split by the first 32 bits of that hash modulo 100 (below 5 test, below 10 validation, the rest training). Identical content therefore always falls into the same split, and exact duplicates are discarded before the division, so that no hash appears in two splits. `manifest.jsonl` records the split, the language and the hash of every file, and `--verify` recomputes the hashes from the binary files. The size limits and the percentages are command-line arguments. Near-duplicates (copied files with small emendations) are not removed, which may render the validation code a little easier than truly unseen code.

## Run

```bash
python scripts/prepare_bytes.py
python scripts/prepare_bytes.py --verify
python scripts/prepare_code.py
python scripts/prepare_code.py --verify
```

`prepare_code.py` requires `pip install datasets` and a Hugging Face login with the dataset terms accepted, the-stack-smol being gated. `--per-lang N` streams only N files per language, for a quick trial.
