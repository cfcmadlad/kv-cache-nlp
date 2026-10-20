import argparse
import hashlib
import http.client
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSET_URL = "https://github.com/cfcmadlad/kv-cache-nlp/releases/download/v0.2-part2/kv-cache-nlp-part2-assets.zip"
ASSET_SHA256 = "21bfdd3e7d445c7f0cdece72dc11942ca409084ecd2de2d04a71c9f3483366dd"
ALLOWED_PREFIXES = ("checkpoints/", "data/processed/")


def sha256_of(fileobj) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: fileobj.read(1 << 20), b""):
        digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path, attempts: int = 3) -> None:
    print(f"Downloading {url}")
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as response, dest.open("wb") as out:
                total, done = int(response.headers.get("Content-Length", 0)), 0
                for block in iter(lambda: response.read(1 << 20), b""):
                    out.write(block)
                    done += len(block)
                    if total:
                        print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            print()
            if total and done != total:
                raise OSError(f"connection closed after {done:,} of {total:,} bytes")
            return
        except (OSError, http.client.HTTPException) as e:
            if attempt == attempts or isinstance(e, urllib.error.HTTPError):
                raise OSError(e) from e
            print(f"\n  attempt {attempt} failed ({e}), retrying...")
            time.sleep(3)


def plan_extraction(archive: zipfile.ZipFile, force: bool) -> list:
    to_write, conflicts = [], []
    for member in archive.infolist():
        if member.is_dir():
            continue
        name = member.filename
        if not name.startswith(ALLOWED_PREFIXES) or ".." in Path(name).parts:
            sys.exit(f"Refusing to extract unexpected path: {name}")
        target = REPO_ROOT / name
        if target.exists():
            with target.open("rb") as on_disk, archive.open(member) as in_zip:
                if sha256_of(on_disk) == sha256_of(in_zip):
                    continue
            if not force:
                conflicts.append(name)
                continue
        to_write.append(member)
    if conflicts:
        sys.exit("These files exist with different content and were not overwritten (use --force):\n  "
                 + "\n  ".join(conflicts))
    return to_write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=ASSET_URL)
    p.add_argument("--zip", type=Path, help="use an already downloaded zip")
    p.add_argument("--force", action="store_true", help="overwrite existing files that differ")
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = args.zip
        if zip_path is not None and not zip_path.is_file():
            sys.exit(f"No such file: {zip_path}")
        if zip_path is None:
            zip_path = Path(tmp) / "assets.zip"
            try:
                download(args.url, zip_path)
            except OSError as e:
                sys.exit(f"Download failed ({e}). Get the zip from {args.url} and run with --zip <path>.")
        with zip_path.open("rb") as f:
            if (actual := sha256_of(f)) != ASSET_SHA256:
                sys.exit(f"SHA-256 mismatch: got {actual}, expected {ASSET_SHA256}.")
        with zipfile.ZipFile(zip_path) as archive:
            for member in plan_extraction(archive, args.force):
                target = REPO_ROOT / member.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".part")
                with archive.open(member) as src, partial.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                partial.replace(target)
                print(f"  placed {member.filename} ({member.file_size / 1e6:.1f} MB)")
    print("Assets ready.")


if __name__ == "__main__":
    main()
