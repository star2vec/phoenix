#!/usr/bin/env python3
"""Compare the sha256 of the relabeled data files in data/relabel/ with a
manifest (by default the one next to them; on the laptop, the manifest in
the transfer folder). Exit 0 on a full match.

    python scripts/verify_relabel.py [--manifest PATH] [--data-dir PATH]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(ROOT / "data" / "relabel"))
    p.add_argument("--manifest", default=None, help="default: <data-dir>/manifest.json")
    args = p.parse_args()
    data_dir = Path(args.data_dir)
    manifest = json.load(open(args.manifest or data_dir / "manifest.json"))
    ok = True
    for split, info in manifest["splits"].items():
        f = data_dir / Path(info["output"]).name
        if not f.exists():
            print(f"  missing   {split}: {f}")
            ok = False
            continue
        h = sha256(f)
        match = h == info["output_sha256"]
        ok &= match
        print(f"  {'match   ' if match else 'MISMATCH'} {split}: {f.name}  {h[:16]}...")
    print(f"seed in manifest: {manifest['seed']}")
    print("RELABEL DATA: MATCH" if ok else "RELABEL DATA: MISMATCH (copy the transfer folder's files, see LAPTOP.md step 5)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
