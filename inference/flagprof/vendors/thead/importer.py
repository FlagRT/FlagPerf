# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Thin recorded CUDA import; launch/correlation data remains raw evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from flagprof.store.writer import Writer
from flagprof.vendors.format import SourceFormat
from flagprof.vendors.import_common import import_fixture as import_recorded
from .rules import build_rules
from flagprof.vendors.disposition import UnrepresentableError


def device_fields(device: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    args = device["ext"]["chrome.args"]
    missing = (["dev"] if type(args.get("device")) is not int else []) + (["stream"] if args.get("stream") is None else [])
    if missing:
        raise UnrepresentableError(missing)
    result = {"dev": args["device"], "stream": args["stream"]}
    for key in ("grid", "block", "regs", "smem"):
        if key in args:
            result[key] = args[key]
    if device["category"] == "memcpy" and type(args.get("bytes")) is int and args["bytes"] >= 0:
        directions = {"Memcpy HtoD": "h2d", "Memcpy DtoH": "d2h", "Memcpy DtoD": "d2d"}
        matches = [kind for prefix, kind in directions.items() if device["name"].startswith(prefix)]
        if len(matches) == 1:
            result.update(kind=matches[0], bytes=args["bytes"])
    return result


def import_fixture(source: str | Path, writer: Writer, *, registry: Mapping[str, Any] | None = None,
                   kind: str | None = None, calls: Sequence[dict[str, Any]] | None = None,
                   marker_prefix: str | None = None, identities: Mapping[str, Any] | None = None,
                   scope_config: Any = None, taxonomy_version: str | None = None) -> dict[str, Any]:
    return import_recorded(source, writer, vendor="thead", build_rules=build_rules,
                           device_fields=device_fields, registry=registry or {"entries": []},
                           kind=kind, calls=calls, marker_prefix=marker_prefix, identities=identities, scope_config=scope_config, taxonomy_version=taxonomy_version)


def device_taxonomy(version: str | None = None) -> SourceFormat:
    from .format import Taxonomy, VERSION
    return Taxonomy(VERSION if version is None else version)


def default_registry() -> dict[str, Any]:
    return {"entries": []}


def field_context(events: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    return events


def memory_record(event, loc):
    from .memory import record
    return record(event, loc)
