"""Download the trained models and processed data from the GitHub release.

Fetches kv-cache-nlp-part2-assets.zip, checks its SHA-256, and unpacks it into
checkpoints/ and data/processed/ so the demo and check_setup.py run without any
training. Uses only the Python standard library.

A file that already exists with identical content is left alone. A file that exists
with different content (for example your own retrained checkpoint, or the full
WikiText-103 train.jsonl, which the release truncates to its first 3,000 documents)
is never overwritten unless you pass --force.

Usage:
    python scripts/fetch_assets.py
    python scripts/fetch_assets.py --zip path/to/kv-cache-nlp-part2-assets.zip
    python scripts/fetch_assets.py --force
"""

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSET_URL = ("https://github.com/cfcmadlad/kv-cache-nlp/releases/download/"
             "v0.2-part2/kv-cache-nlp-part2-assets.zip")
ASSET_SHA256 = "21bfdd3e7d445c7f0cdece72dc11942ca409084ecd2de2d04a71c9f3483366dd"
ALLOWED_PREFIXES = ("checkpoints/", "data/processed/")


def sha256_of(fileobj) -> str:
    """SHA-256 of an open binary file, read in 1 MB blocks."""
    digest = hashlib.sha256()
    for block in iter(lambda: fileobj.read(1 << 20), b""):
        digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream url to dest, printing progress in MB."""
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        for block in iter(lambda: response.read(1 << 20), b""):
            out.write(block)
            done += len(block)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()


def plan_extraction(archive: zipfile.ZipFile, force: bool) -> list:
    """Return the members to write, or exit if any would overwrite a different file."""
    to_write, conflicts = [], []
    for member in archive.infolist():
        if member.is_dir():
            continue
        name = member.filename
        if not name.startswith(ALLOWED_PREFIXES) or ".." in Path(name).parts:
            sys.exit(f"Refusing to extract unexpected path from the zip: {name}")
        target = REPO_ROOT / name
        if target.exists():
            with target.open("rb") as on_disk, archive.open(member) as in_zip:
                if sha256_of(on_disk) == sha256_of(in_zip):
                    print(f"  already present  {name}")
                    continue
            if not force:
                conflicts.append(name)
                continue
        to_write.append(member)
    if conflicts:
        sys.exit("These files already exist with different content and were NOT overwritten:\n  "
                 + "\n  ".join(conflicts)
                 + "\nRe-run with --force to replace them with the released versions.")
    return to_write


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=ASSET_URL, help="where to download the zip from")
    parser.add_argument("--zip", type=Path, help="use an already downloaded zip instead of downloading")
    parser.add_argument("--force", action="store_true", help="overwrite existing files that differ")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = args.zip
        if zip_path is not None and not zip_path.is_file():
            sys.exit(f"No such file: {zip_path}")
        if zip_path is None:
            zip_path = Path(tmp) / "assets.zip"
            try:
                download(args.url, zip_path)
            except OSError as e:
                sys.exit(f"Download failed ({e}). Download the zip in a browser from\n  {args.url}\n"
                         "and run: python scripts/fetch_assets.py --zip <path to the zip>")

        with zip_path.open("rb") as f:
            actual = sha256_of(f)
        if actual != ASSET_SHA256:
            sys.exit(f"SHA-256 mismatch: got {actual}, expected {ASSET_SHA256}. The file is "
                     "incomplete or not the Part 2 release zip. Download it again and re-run.")
        print("SHA-256 verified.")

        with zipfile.ZipFile(zip_path) as archive:
            for member in plan_extraction(archive, args.force):
                target = REPO_ROOT / member.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                # Write to a temporary name first, so an interrupted run never leaves a
                # half-written file under the real name.
                partial = target.with_name(target.name + ".part")
                with archive.open(member) as src, partial.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                partial.replace(target)
                print(f"  placed  {member.filename}  ({member.file_size / 1e6:.1f} MB)")

    print("Done. Next: python scripts/check_setup.py")


if __name__ == "__main__":
    main()
