#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Deprecated container entry shim for Benchmark and legacy Toolkit runs."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from benchmark_worker import (  # noqa: E402,F401
    build_parser,
    build_torchrun_command,
    main as benchmark_main,
    parse_args,
    resolve_benchmark_entrypoint,
    write_pid_file,
)


def requested_domain(argv: list[str]) -> str:
    try:
        index = argv.index("--bench_or_tool")
    except ValueError:
        return "BENCHMARK"
    if index + 1 >= len(argv):
        raise RuntimeError("--bench_or_tool requires a value")
    return argv[index + 1].upper()


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    domain = requested_domain(arguments)
    if domain == "BENCHMARK":
        return benchmark_main(arguments)
    if domain != "TOOLKIT":
        raise RuntimeError(f"unsupported container domain: {domain}")

    print(
        "DEPRECATED: container_main.py Toolkit mode; use the host ToolkitExecutor",
        file=sys.stderr,
    )
    legacy = BASE_DIR / "legacy" / "container_main.py"
    previous = sys.argv
    sys.argv = [str(legacy), *arguments]
    try:
        try:
            runpy.run_path(str(legacy), run_name="__main__")
        except SystemExit as exc:
            return int(exc.code or 0)
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
