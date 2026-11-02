# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Independent raw-device inventory. Does not import adapter or linker."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Sequence
from .format import SourceFormat


def enumerate_raw(events: Sequence[Dict[str, Any]], trace_uid: str, *, taxonomy: SourceFormat) -> List[Dict[str, Any]]:
    """Enumerate R, including every unknown device task rather than dropping it."""
    context = taxonomy.context(events)
    output = []
    for i, event in enumerate(events):
        if not taxonomy.is_device(event, context):
            continue
        category = taxonomy.category(event, context)
        task = taxonomy.task_type(event)
        output.append({"trace_uid": trace_uid, "loc": event.get("_linker_loc", i),
                       "category": category, "name": event.get("name", ""), "task_type": task,
                       "dur": int(Decimal(str(event["dur"])) * 1000) if event.get("dur") is not None else None})
    return sorted(output, key=lambda r: (str(r["trace_uid"]), str(r["loc"])))
