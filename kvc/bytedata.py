import json
from pathlib import Path

import numpy as np


def write_split(docs, out_dir: Path, split: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    with open(out_dir / f"{split}.bin", "wb") as f:
        for doc in docs:
            offsets.append(offsets[-1] + f.write(doc.encode("utf-8")))
    np.save(out_dir / f"{split}_offsets.npy", np.asarray(offsets, dtype=np.int64))
    return {"split": split, "num_docs": len(offsets) - 1, "num_bytes": offsets[-1]}


def open_split(out_dir: Path, split: str):
    data = np.memmap(out_dir / f"{split}.bin", dtype=np.uint8, mode="r")
    return data, np.load(out_dir / f"{split}_offsets.npy")


def read_doc(data, offsets, i: int) -> str:
    return bytes(data[offsets[i]:offsets[i + 1]]).decode("utf-8")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)["text"]


def sample_roundtrip(out_dir: Path, split: str, source_docs: list, n: int, seed: int) -> int:
    data, offsets = open_split(out_dir, split)
    assert len(offsets) - 1 == len(source_docs), f"{split}: {len(offsets) - 1} docs vs {len(source_docs)}"
    idx = np.random.default_rng(seed).choice(len(source_docs), size=min(n, len(source_docs)), replace=False)
    bad = [int(i) for i in idx if read_doc(data, offsets, int(i)) != source_docs[int(i)]]
    assert not bad, f"{split}: round-trip mismatch at docs {bad[:5]}"
    return len(idx)
