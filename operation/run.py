#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Vendor-neutral operation CLI. Help and dry-run do not import device libraries."""
from runtime.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
