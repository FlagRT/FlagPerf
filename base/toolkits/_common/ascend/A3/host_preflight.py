#!/usr/bin/env python3
"""Compatibility entrypoint for the shared Ascend host preflight."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from base.vendors.ascend.preflight import *  # noqa: F401,F403

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
