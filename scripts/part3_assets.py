import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import fetch_assets
from fetch_assets import REPO_ROOT, download, plan_extraction, sha256_of

URL = "https://github.com/cfcmadlad/kv-cache-nlp/releases/download/v0.3-part3/kv-cache-nlp-part3-assets.zip"
SHA256 = "95b3c9e212bd342f5226eb734d3faaa9ae3fb790b5f869e92eed36f40dba14ce"
FILES = [f"runs/{run}/{name}" for run, name in [
    ("entropy_tinystories", "model.pt"), ("entropy_tinystories", "patching.json"), ("blt_tinystories_entropy", "model.pt"),
    ("entropy_wikitext103", "model.pt"), ("entropy_wikitext103", "patching.json"), ("blt_wikitext103_entropy", "model.pt")]]


def build(out: Path):
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(REPO_ROOT / f, f)
    with out.open("rb") as f:
        print(f"{out} sha256 {sha256_of(f)}")


def fetch(zip_path: Path = None, force: bool = False):
    fetch_assets.ALLOWED_PREFIXES = ("runs/",)
    with tempfile.TemporaryDirectory() as tmp:
        if zip_path is None:
            zip_path = Path(tmp) / "assets.zip"
            try:
                download(URL, zip_path)
            except OSError as e:
                sys.exit(f"Download failed ({e}). Get the zip from {URL} and run with --zip <path>.")
        with zip_path.open("rb") as f:
            if SHA256 and (actual := sha256_of(f)) != SHA256:
                sys.exit(f"SHA-256 mismatch: got {actual}, expected {SHA256}.")
        with zipfile.ZipFile(zip_path) as archive:
            for member in plan_extraction(archive, force):
                target = REPO_ROOT / member.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
                print(f"  placed {member.filename}")
    print("Part 3 assets ready.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["build", "fetch"])
    p.add_argument("--zip", type=Path)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    build(args.zip or REPO_ROOT / "runs" / "kv-cache-nlp-part3-assets.zip") if args.command == "build" else fetch(args.zip, args.force)


if __name__ == "__main__":
    main()
