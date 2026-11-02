# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Derived merge only: no mutation of writer fragments or original events."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import uuid

from .manifest import atomic_json, json_bytes
from .reader import Reader


def compact(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    # Compact is a lossless storage operation, including future schema fields.
    with Reader(root, preserve_unknown=True) as reader:
        records = reader.read()
        report = reader.report
    target = root / "compact"
    target.mkdir(exist_ok=True)
    temporary = target / (".records." + uuid.uuid4().hex + ".jsonl.tmp")
    try:
        with temporary.open("xb") as handle:
            for stored in records:
                handle.write(json_bytes(stored.as_dict()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target / "records.jsonl")
    finally:
        temporary.unlink(missing_ok=True)
    index = {"records": len(records), "produced": sum("seq" in x.record for x in records),
             "signatures": sum(x.record["t"] == "sig" for x in records), "source_report": report,
             "identity_preserved": True, "pruned": False}
    atomic_json(target / "index.json", index)
    return index
