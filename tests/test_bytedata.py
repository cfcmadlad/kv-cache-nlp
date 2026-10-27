import json
import subprocess
import sys
from types import SimpleNamespace

from kvc.bytedata import open_split, read_doc, sample_roundtrip, write_split
from prepare_code import build, split_files, verify

DOCS = ["plain ascii", "café 中文 \U0001f600", "", "line\nbreak\r\n", "x" * 1000]

def test_roundtrip_and_offsets(tmp_path):
    stats = write_split(DOCS, tmp_path, "val")
    data, offsets = open_split(tmp_path, "val")
    assert stats["num_bytes"] == len(data) == sum(len(d.encode()) for d in DOCS)
    assert [read_doc(data, offsets, i) for i in range(len(DOCS))] == DOCS
    assert sample_roundtrip(tmp_path, "val", DOCS, 100, 0) == len(DOCS)

def test_prepare_bytes_cli(tmp_path):
    src, out = tmp_path / "src" / "tinystories", tmp_path / "out"
    src.mkdir(parents=True)
    stats = {}
    for split in ("train", "val"):
        (src / f"{split}.jsonl").write_text("\n".join(json.dumps({"text": d}) for d in DOCS), encoding="utf-8")
        stats[split] = {"num_docs": len(DOCS), "num_bytes": sum(len(d.encode()) for d in DOCS)}
    (src / "stats.json").write_text(json.dumps(stats))
    for extra in ([], ["--verify"]):
        r = subprocess.run([sys.executable, "scripts/prepare_bytes.py", "--dataset", "tinystories",
                            "--src", str(tmp_path / "src"), "--out", str(out), *extra], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

def fake_records(n):
    langs = ["python", "java", "javascript"]
    return [(langs[i % 3], f"// file {i}\n" + "code line;\n" * 40) for i in range(n)]

def test_code_split_no_cross_split_duplicates(tmp_path):
    records = fake_records(600) + fake_records(100) + [("python", "tiny")]
    args = SimpleNamespace(min_bytes=200, max_bytes=100_000, val_pct=5, test_pct=5)
    build(iter(records), tmp_path, args)
    verify(tmp_path, 5, 5)
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["dropped"] == {"size": 1, "duplicate": 100, "null": 0}
    assert sum(stats[s]["num_docs"] for s in ("train", "val", "test")) == 600
    assert stats["val"]["num_docs"] > 0 and stats["test"]["num_docs"] > 0

def test_split_is_a_function_of_content():
    a = split_files([("python", "same content " * 30)], 10, 10**6, 5, 5)
    b = split_files([("java", "same content " * 30)], 10, 10**6, 5, 5)
    assert [len(v) for v in a["splits"].values()] == [len(v) for v in b["splits"].values()]
