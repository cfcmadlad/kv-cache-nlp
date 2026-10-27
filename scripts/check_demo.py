import importlib
import sys
import time

from data_utils import REPO_ROOT

MODULES = ("torch", "numpy", "tokenizers", "tqdm")


def report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return ok


def main():
    results = [report("Python 3.10 or newer", sys.version_info >= (3, 10), sys.version.split()[0])]
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            results.append(report(f"import {name}", True, getattr(module, "__version__", "")))
        except ImportError as e:
            results.append(report(f"import {name}", False, str(e)))
    if not all(results):
        sys.exit("Install the requirements first: pip install -r requirements.txt (PyTorch from https://pytorch.org)")
    import torch
    from demo import PART3_FILES, REQUIRED_FILES

    results.append(report("device", True, "cuda" if torch.cuda.is_available() else "cpu"))
    missing = [f for f in REQUIRED_FILES + PART3_FILES if not (REPO_ROOT / f).exists()]
    results.append(report("assets present", not missing, ", ".join(missing[:3]) + (" ..." if len(missing) > 3 else "") if missing else f"{len(REQUIRED_FILES) + len(PART3_FILES)} files"))
    if missing:
        print("Run python scripts/demo.py once with internet access (it downloads the assets), or place the two release zips in this folder.")
        sys.exit(1)
    from cache_demo import analyse, load_pair
    from demo_part3 import evaluate_live, train_live

    t0 = time.perf_counter()
    model, entropy, cfg = load_pair("tinystories", "cpu")
    text = "Once upon a time, there was a little robot who wanted to find its friend."
    out = analyse(text, model, entropy, cfg, 12, "cpu")
    results.append(report("option 4 path (compress a text)", len(out["rows"]) == 7, f"{out['patches']} patches, {time.perf_counter() - t0:.1f}s"))
    t0 = time.perf_counter()
    tiny = train_live([text * 40], 3, "cpu", block=64, batch=2, log=lambda *_: None)
    rows = evaluate_live(tiny, [text], 12, "cpu", max_bytes=64)
    results.append(report("option 6 path (live training)", len(rows) == 4, f"{time.perf_counter() - t0:.1f}s"))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
