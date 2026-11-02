# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Thin recorded CANN import; physical mappings are never inferred from pid."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import re

from flagprof.store.writer import Writer
from flagprof.vendors.format import SourceFormat
from flagprof.vendors.import_common import import_fixture as import_recorded
from .rules import build_rules
from .rules import VERSION as RULES_VERSION
from flagprof.vendors.disposition import UnrepresentableError


def device_fields(device: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    args = device["ext"]["chrome.args"]
    labels = {event.get("pid"): event.get("args", {}).get("labels", "")
              for event in events if event.get("name") == "process_labels"}
    matched = re.fullmatch(r"NPU (\d+)", labels.get(device["pid"], ""))
    stream = args.get("Physic Stream Id", device["tid"])
    missing = ([] if matched is not None else ['dev']) + ([] if stream is not None else ['stream'])
    if missing:
        raise UnrepresentableError(missing)
    # This is the logical profiler label, not a physical board/socket index.
    return {"dev": int(matched.group(1)), "stream": stream}


def import_fixture(source: str | Path, writer: Writer, *, registry: Mapping[str, Any] | None = None,
                   kind: str | None = None, calls: Sequence[dict[str, Any]] | None = None,
                   marker_prefix: str | None = None, identities: Mapping[str, Any] | None = None,
                   scope_config: Any = None, taxonomy_version: str | None = None) -> dict[str, Any]:
    if registry is None:
        registry = json.loads(Path(__file__).with_name("registry.json").read_text())
    return import_recorded(source, writer, vendor="ascend", build_rules=build_rules,
                           device_fields=device_fields, registry=registry, kind=kind,
                           calls=calls, marker_prefix=marker_prefix, identities=identities, scope_config=scope_config, taxonomy_version=taxonomy_version)


def device_taxonomy(version: str | None = None) -> SourceFormat:
    from .format import Taxonomy, VERSION
    return Taxonomy(VERSION if version is None else version)


def default_registry() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("registry.json").read_bytes())


def field_context(events: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    return [e for e in events if e.get('name') == 'process_labels']


def rules_identity() -> dict[str,str]:
    from . import rules,native_rules
    from flagprof.vendors.fingerprint import format_identity
    return format_identity(RULES_VERSION,(rules.__file__,native_rules.__file__))
