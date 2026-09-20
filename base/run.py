#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Unified host facade for FlagPerf Base Benchmark and Toolkit runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from executors.benchmark import (  # noqa: E402
    BenchmarkExecutor,
    BenchmarkRunRequest,
    add_cli_arguments as add_benchmark_arguments,
    generate_benchmark_report,
)
from executors.common import ConfigurationError  # noqa: E402
from executors.toolkit import (  # noqa: E402
    ToolkitExecutor,
    ToolkitRunRequest,
    add_cli_arguments as add_toolkit_arguments,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flagperf-base",
        description=(
            "Unified host control plane for FlagPerf Base. Benchmark and "
            "Toolkit keep separate requests, permissions, and result semantics."
        ),
    )
    domains = parser.add_subparsers(dest="domain", metavar="DOMAIN")

    benchmark = domains.add_parser(
        "benchmark", help="PyTorch/Torch-FL Base Benchmark operations",
    )
    benchmark_actions = benchmark.add_subparsers(dest="action", metavar="ACTION")
    benchmark_run = benchmark_actions.add_parser(
        "run", help="plan or execute a Benchmark from the host",
    )
    add_benchmark_arguments(benchmark_run)
    from executors.preflight import add_cli_arguments as add_preflight_arguments
    preflight = benchmark_actions.add_parser("preflight", help="bounded vendor identity qualification, without a performance case")
    add_preflight_arguments(preflight)

    toolkit = domains.add_parser(
        "toolkit", help="vendor measurement and diagnosis Toolkit operations",
    )
    toolkit_actions = toolkit.add_subparsers(dest="action", metavar="ACTION")
    toolkit_run = toolkit_actions.add_parser(
        "run", help="plan or execute the evidence-first Ascend Toolkit",
    )
    add_toolkit_arguments(toolkit_run, include_compat_suite=False)

    report = domains.add_parser(
        "report", help="regenerate a report from stored run evidence",
    )
    report.add_argument("--run-id", required=True)
    report.add_argument(
        "--result-root", type=Path, default=BASE_DIR / "result",
    )
    return parser


def generate_report(run_id: str, result_root: Path) -> int:
    root = (result_root.expanduser().resolve() / run_id).resolve()
    expected_parent = result_root.expanduser().resolve()
    if root.parent != expected_parent:
        raise ConfigurationError("--run-id must name one direct result directory")
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise ConfigurationError(f"run summary does not exist: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    kind = summary.get("kind")
    if kind == "benchmark-preflight":
        from executors.preflight import render_report
        render_report(root)
        return 0
    # Results produced before the unified facade did not carry ``kind``.
    # Prefer the explicit discriminator, then fall back to durable artifacts so
    # report regeneration remains backward compatible without rewriting the
    # stored experiment status.
    if kind == "benchmark" or (
        kind is None and (root / "benchmark-result.json").is_file()
    ):
        metadata = generate_benchmark_report(root)
    elif kind == "toolkit" or summary.get("suite") == "ascend-toolkit":
        from generate_toolkit_report import generate_and_record

        metadata = generate_and_record(root)
    else:
        raise ConfigurationError(
            f"cannot determine report generator for run kind: {kind!r}"
        )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Report directory: {root}")
    return 0 if metadata.get("status") == "passed" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help(sys.stderr)
        print(
            "\nThe previous no-argument cluster runner moved to "
            "base/legacy/cluster_run.py.",
            file=sys.stderr,
        )
        return 2
    args = parser.parse_args(arguments)
    try:
        if args.domain == "benchmark" and args.action == "preflight":
            from executors.preflight import execute
            return execute(args)
        if args.domain == "benchmark" and args.action == "run":
            request = BenchmarkRunRequest.from_namespace(args)
            return BenchmarkExecutor().execute(request)
        if args.domain == "toolkit" and args.action == "run":
            request = ToolkitRunRequest.from_namespace(args)
            return ToolkitExecutor(require_selection=True).execute(request)
        if args.domain == "report":
            return generate_report(args.run_id, args.result_root)
        parser.error("a DOMAIN and ACTION are required")
    except (ConfigurationError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
