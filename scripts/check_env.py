#!/usr/bin/env python3
"""Check that the active Python has what phoenix needs, and print what is
missing or off-version. Exit code 0 when everything matches, 1 otherwise.

    python scripts/check_env.py

phoenix imports torch, transformers, numpy and (through the vendor dataset
builder) datasets. tqdm and pyyaml are pinned by the vendor setup but no code
path phoenix runs imports them; they are reported as optional.
"""

import importlib
import sys

REQUIRED = {"torch": "2.5.1", "transformers": "4.46.2", "numpy": "2.1.3", "datasets": "3.1.0"}
OPTIONAL = {"tqdm": "4.67.0", "yaml": "6.0"}


def version_of(mod):
    return getattr(mod, "__version__", "?")


def main():
    print(f"python {sys.version.split()[0]}  ({sys.executable})")
    ok = True
    for name, pin in REQUIRED.items():
        try:
            mod = importlib.import_module(name)
        except ImportError:
            print(f"  MISSING   {name}  (need {pin})")
            ok = False
            continue
        v = version_of(mod)
        flag = "ok       " if v.startswith(pin) else "VERSION  "
        if not v.startswith(pin):
            ok = False
        print(f"  {flag} {name} {v}  (pinned {pin})")
    for name, pin in OPTIONAL.items():
        try:
            v = version_of(importlib.import_module(name))
            print(f"  optional  {name} {v}")
        except ImportError:
            print(f"  optional  {name} not installed (not needed by phoenix)")
    try:
        import torch
        print(f"  cuda available: {torch.cuda.is_available()}"
              + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    except Exception as e:  # noqa: BLE001
        print(f"  cuda check failed: {e}")
    print("ENV: OK" if ok else "ENV: FIX THE LINES ABOVE (see LAPTOP.md, offline install)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
