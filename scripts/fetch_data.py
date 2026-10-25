"""Download and verify the Ott Deceptive Opinion Spam Corpus v1.4.

The corpus is CC BY-NC-SA 3.0, so it is fetched at run time and never committed.

    python scripts/fetch_data.py                 # downloads to data/op_spam_v1.4
    python scripts/fetch_data.py --from-zip PATH  # use a local copy of the zip
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://myleott.com/op_spam_v1.4.zip"
SHA256 = "e3a61b3709ccaca842aeec7f525e273988cf3b1bab35151712cf89759a6724fa"
REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "data" / "op_spam_v1.4"
EXPECTED_FILES = 1600


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_reviews(root: Path) -> int:
    return len(list(root.glob("*_polarity/*/fold*/*.txt")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-zip", type=Path, help="local op_spam_v1.4.zip")
    parser.add_argument("--dest", type=Path, default=DEST)
    args = parser.parse_args()

    if args.dest.exists() and count_reviews(args.dest) == EXPECTED_FILES:
        print(f"Already present: {args.dest} ({EXPECTED_FILES} reviews)")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "op_spam_v1.4.zip"
        if args.from_zip:
            shutil.copy(args.from_zip, zip_path)
        else:
            print(f"Downloading {URL}")
            urllib.request.urlretrieve(URL, zip_path)
        got = sha256(zip_path)
        if got != SHA256:
            print(f"Checksum mismatch: {got}", file=sys.stderr)
            return 1
        args.dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(args.dest.parent)

    n = count_reviews(args.dest)
    if n != EXPECTED_FILES:
        print(f"Expected {EXPECTED_FILES} reviews, found {n}", file=sys.stderr)
        return 1
    print(f"Extracted {n} reviews to {args.dest} (sha256 ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
