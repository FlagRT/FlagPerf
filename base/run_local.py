#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Deprecated compatibility entry for the Ascend Base Toolkit runner."""

from __future__ import annotations

from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from executors.toolkit import (  # noqa: E402,F401
    REPORT_SCHEMA_VERSION,
    add_cli_arguments,
    container_namespace_args,
    docker_inspect,
    fail,
    generate_report_safely,
    main,
    parse_args,
    run_host_preflight,
    validate_static_args,
    write_summary,
)


if __name__ == "__main__":
    print(
        "DEPRECATED: base/run_local.py; use base/run.py toolkit run "
        "(or base/run_toolkit.py during migration)",
        file=sys.stderr,
    )
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
