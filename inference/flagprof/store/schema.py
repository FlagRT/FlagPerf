# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Declarative PLAN/03 field contracts, shared by writing and reading.

Conditional requirements come from the text immediately following the tables:
route(hit), span(e), counter(src), tensor(unscoped), sig(trunc), and link(N/A).
Unknown fields/types are counted and ignored, never interpreted as known facts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Protocol
import re

SCHEMA_VERSION = "1.2"
REQUIRED_FIELDS = {
    "meta": ("kind",),
    "sig": ("sid", "role", "name", "overload", "tree"),
    "op": ("sid", "corr", "dur"),
    "opsum": ("sid", "n", "dur_sum", "win"),
    "route": ("corr", "lib", "decision", "reason"),
    "compile": ("corr", "backend", "key", "cache", "dur"),
    "comm": ("op", "group", "group_seq", "bytes", "dur"),
    "span": ("ph", "cat", "name", "corr"),
    "kernel": ("name", "dev", "stream", "ts", "dur", "link"),
    # PLAN/03 §4.8 specifies memop inline instead of in a bold field table.
    "memop": ("kind", "bytes", "dev", "stream", "ts", "dur", "link"),
    "tensor": ("call_key", "role", "boundary", "shape", "dtype", "stats", "chunks", "probe", "probe_alg"),
    "counter": ("src", "dev", "values"),
    "health": ("kind", "sev"),
}
OPTIONAL_FIELDS = {
    "meta": {"data"},
    "sig": {"trunc"},
    "op": {"osid", "dev", "scope", "seq_in_scope", "fw", "parent_corr", "async", "err"},
    "opsum": {"scope", "route_decision", "dur_min", "dur_max", "dur_p50", "dur_p99", "dev", "nested"},
    "route": {"impl", "depth", "detail"},
    "compile": {"kname", "stage_dur", "ver", "err"},
    "comm": {"dtype", "count", "lib", "algo", "phase", "corr", "link", "dev", "stream"},
    "span": {"parent", "attrs", "dur"},
    "kernel": {"ktype", "grid", "block", "regs", "smem", "metrics"},
    "memop": set(),
    "tensor": {"corr", "digest", "blob", "mask", "align", "sid", "src"},
    "counter": {"blocked"},
    "health": {"detail", "evidence"},
}
ENVELOPE = {"v", "t", "ts", "clk", "seq", "pid", "tid", "rank", "p", "corr", "ext"}
CONTROL_META = {"shard_header", "defaults", "clock", "drop_report"}
META_KINDS = CONTROL_META | {"run_start", "env", "model", "run_end"}
LINK_REASONS = {"no_bridge", "no_device_key", "conflict", "source_missing", "excluded", "graph_replay"}
ENUM_FIELDS = {
    "meta": {"kind": META_KINDS},
    "sig": {"role": {"in", "out"}},
    "route": {"decision": {"hit", "fallback", "excluded", "error"},
              "reason": {"not_registered", "unsupported_signature", "device_unsupported", "policy_excluded", "compile_failed", "runtime_error", "numerical_guard"}},
    "compile": {"cache": {"hit", "miss", "invalid"}},
    "comm": {"op": {"allreduce", "allgather", "broadcast", "reducescatter", "alltoall", "send", "recv", "barrier"}, "phase": {"enqueue", "device"}},
    "span": {"ph": {"b", "e", "x"}, "cat": {"phase", "module", "user", "engine", "batch", "step", "lifecycle"}},
    "kernel": {"ktype": {"compute", "comm", "memcpy", "unknown"}},
    "memop": {"kind": {"h2d", "d2h", "d2d", "alloc", "free"}},
    "tensor": {"role": {"in", "out", "grad"}},
    "counter": {"src": {"device", "framework", "host"}},
    "health": {"kind": {"heartbeat", "hang", "oom", "comm_timeout", "device_error", "process_exit", "ecc", "link_down"},
               "sev": {"info", "warn", "error", "fatal"}},
}
U64_MAX = (1 << 64) - 1
SID_RE = re.compile(r"[0-9a-f]{16}\Z")


class RecordValidator(Protocol):
    def decode(self, record: Mapping[str, Any], defaults: Mapping[str, Any] | None = None) -> dict[str, Any] | None: ...


class SchemaError(ValueError):
    """A known record violates the schema."""


class UnsupportedVersion(SchemaError):
    """A major schema change cannot be interpreted as current semantics."""


def is_control(record: Mapping[str, Any]) -> bool:
    return record.get("t") == "sig" or (record.get("t") == "meta" and
        (record.get("kind") in CONTROL_META or (record.get("kind") not in META_KINDS and "seq" not in record)))


def _integer(value: Any, *, minimum: int = 0, maximum: int | None = None) -> bool:
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def _require(ok: bool, text: str) -> None:
    if not ok:
        raise SchemaError(text)


def validate_link(link: Mapping[str, Any]) -> None:
    _require(isinstance(link, dict), "link must be an object")
    status = link.get("status")
    _require(status in {"linked", "ambiguous", "unlinked", "not_applicable"}, "invalid link.status")
    via = link.get("via")
    _require(isinstance(via, list) and all(isinstance(x, str) for x in via), "link.via must be an array of rule ids")
    _require(via == sorted(set(via)), "link.via must be sorted and unique")
    evidence = link.get("evidence")
    _require(isinstance(evidence, list) and bool(evidence), "link.evidence must include the device source")
    for ref in evidence:
        _require(isinstance(ref, dict), "invalid evidence reference")
        _require(("trace_uid" in ref and "loc" in ref) or ("w" in ref and "seq" in ref), "evidence requires a stable identity")
    if status == "linked":
        _require(_integer(link.get("corr"), minimum=1, maximum=U64_MAX), "linked requires a corr")
        _require("candidates" not in link, "linked must not have candidates")
        _require("reason" not in link, "linked must not have reason")
    else:
        _require(isinstance(link.get("reason"), str) and bool(link["reason"]), "non-linked requires reason")
        _require("corr" not in link, "only linked may carry corr")
    if status == "ambiguous":
        candidates = link.get("candidates")
        _require(isinstance(candidates, list) and len(candidates) >= 2, "ambiguous requires multiple candidates")
        real = [x for x in candidates if x is not None]
        _require(all(_integer(x, minimum=1, maximum=U64_MAX) for x in real), "invalid candidate")
        _require(candidates == sorted(set(real)) + ([None] if None in candidates else []), "candidates must be sorted, null last")
    else:
        _require("candidates" not in link, "only ambiguous may carry candidates")
    if status == "not_applicable":
        # §4.8 permits null; T4 emits the narrower form (field absent).
        _require(link.get("segments") is None and "level" not in link, "N/A has no segments or level")
        _require(link["reason"] == "excluded", "N/A requires registry exclusion")
        return
    segments = link.get("segments")
    _require(isinstance(segments, dict) and set(segments) == {"host_attr", "device_link"}, "link requires both segments")
    _require(all(x in {"ok", "broken"} for x in segments.values()), "invalid segment state")
    host, device = segments["host_attr"], segments["device_link"]
    _require((host == "ok") == (status == "linked"), "host_attr ok requires unique attribution")
    _require(not (host == "ok" and device == "broken"), "device records cannot have launch level B")
    if status == "ambiguous":
        _require(device == "ok", "ambiguous requires device evidence")
    level = "A" if host == "ok" else "B+" if device == "ok" else "C"
    _require("level" not in link or link["level"] == level, "level disagrees with segments")


@dataclass
class SchemaValidator:
    """Counters belong to the reader/writer instance, never a global singleton."""

    unknown_fields: int = 0
    unknown_types: int = 0
    unknown_enums: int = 0

    def decode(self, record: Mapping[str, Any], defaults: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        _require(isinstance(record, dict), "record must be a JSON object")
        event_type = record.get("t")
        _require(isinstance(event_type, str), "t is required")
        if event_type not in REQUIRED_FIELDS:
            self.unknown_types += 1
            return None
        known = set(REQUIRED_FIELDS[event_type]) | OPTIONAL_FIELDS[event_type] | ENVELOPE
        self.unknown_fields += len(set(record) - known)
        result = {k: v for k, v in record.items() if k in known}
        if not is_control(result):
            for key, value in (defaults or {}).items():
                if key in known:
                    result.setdefault(key, value)
        version = result.get("v")
        if version is not None:
            _require(isinstance(version, str) and bool(re.fullmatch(r"\d+\.\d+", version)), "invalid schema version")
            if int(version.split(".")[0]) > 1:
                raise UnsupportedVersion("unsupported schema major version: " + version)
            _require(event_type == "meta" and result.get("kind") == "shard_header", "v only belongs to shard_header")
        required = set(REQUIRED_FIELDS[event_type])
        if event_type == "route" and result.get("decision") == "hit":
            required.discard("reason")
        if event_type == "span" and result.get("ph") == "e":
            required -= {"cat", "name"}
        if event_type == "counter" and result.get("src") != "device":
            required.discard("dev")
        _require(required <= set(result), f"{event_type} missing fields: {sorted(required - set(result))}")
        control = is_control(result)
        if control:
            _require("seq" not in result, "control records must not have seq")
        else:
            _require(_integer(result.get("seq")), "produced records require seq")
        if event_type == "sig":
            _require(not (set(result) & (ENVELOPE - {"t"})), "sig has no event envelope")
            self._signature(result)
            return result
        for key, values in ENUM_FIELDS.get(event_type, {}).items():
            if key in result:
                _require(isinstance(result[key], str), f"{key} enum must be a string")
                if result[key] not in values:
                    result[key] = "other"
                    self.unknown_enums += 1
        _require("ts" in result and (result["ts"] is None or _integer(result["ts"])), "ts must be int or null")
        for key in {"seq", "pid", "tid", "rank", "scope", "seq_in_scope", "parent", "depth", "n", "dur_sum", "group_seq", "bytes", "count"} & set(result):
            _require(_integer(result[key]), f"{key} must be a nonnegative integer")
        for key in {"corr", "parent_corr"} & set(result):
            _require(_integer(result[key], maximum=U64_MAX), f"{key} must be u64")
        if "dur" in result:
            _require(_integer(result["dur"]) or (event_type in {"kernel", "memop", "comm"} and result["dur"] is None), "dur must be nonnegative ns (or unavailable for imported records)")
        if event_type in ('op', 'opsum') and 'dev' in result:
            dev = result['dev']
            _require(isinstance(dev, str) and bool(dev) or isinstance(dev, dict) and bool(dev) and
                     all(isinstance(k, str) and isinstance(v, str) and v for k, v in dev.items()), 'invalid call device')
        for key in {"sid", "osid"} & set(result):
            _require(isinstance(result[key], str) and bool(SID_RE.fullmatch(result[key])), f"invalid {key}")
        for key in {"name", "lib", "backend", "key", "boundary", "dtype", "probe_alg", "p", "clk"} & set(result):
            _require(isinstance(result[key], str), f"{key} must be a string")
        for key in {"ext", "data", "attrs", "stats", "values", "metrics", "stage_dur"} & set(result):
            _require(isinstance(result[key], dict), f"{key} must be an object")
        if "link" in result:
            validate_link(result["link"])
            if result["link"].get("reason") not in LINK_REASONS | {None}:
                result["link"] = dict(result["link"], reason="other")
                self.unknown_enums += 1
        if event_type == "opsum":
            _require("corr" not in result, "opsum has no corr")
            _require('nested' not in result or result['nested'] is True, 'opsum nested must be true when present')
            _require(isinstance(result["win"], list) and len(result["win"]) == 2 and all(_integer(x) for x in result["win"]), "win requires [start,end]")
            _require(result["win"][0] <= result["win"][1], "invalid win order")
        if event_type == "span":
            _require(result["ph"] != "x" or "dur" in result, "complete span requires dur")
        if event_type == "tensor":
            _require("src" not in result or isinstance(result["src"], list) and all(isinstance(v, str) for v in result["src"]), "src must be a string array")
            _require(result["call_key"] is not None or result.get("align") == "unscoped", "null call_key requires align=unscoped")
            _require(result.get("align") != "unscoped" or result["call_key"] is None, "unscoped tensor cannot carry call_key")
            _require(isinstance(result["shape"], list) and all(_integer(x) for x in result["shape"]), "invalid shape")
            _require(isinstance(result["chunks"], list) and isinstance(result["probe"], list), "chunks/probe must be arrays")
        if event_type == "meta" and result["kind"] == "run_start":
            data = result.get("data", {})
            _require({"run_id", "proc_key", "heartbeat_period_ns"} <= set(data), "run_start requires process key and heartbeat period")
            _require({"boot_id", "pid_ns", "pid", "proc_start_ticks"} <= set(data["proc_key"]), "incomplete proc_key")
            _require(_integer(data["heartbeat_period_ns"], minimum=1), "invalid heartbeat period")
        if event_type == "meta" and result["kind"] == "drop_report":
            data = result.get("data", {})
            _require(_integer(data.get("next_seq")) and isinstance(data.get("dropped"), dict), "drop_report requires next_seq/dropped")
            _require(all(_integer(x) for x in data["dropped"].values()), "invalid drop count")
        return result

    def _signature(self, record: Mapping[str, Any]) -> None:
        from .sig import compute_sid, validate_tree
        _require(isinstance(record["sid"], str) and bool(SID_RE.fullmatch(record["sid"])), "invalid sig.sid")
        _require(record["role"] in {"in", "out"}, "invalid signature role")
        _require(isinstance(record["name"], str) and isinstance(record["overload"], str), "invalid signature name")
        if record.get("trunc", False):
            _require(record["tree"] == [], "truncated signature must have empty tree")
            return
        try:
            validate_tree(record["tree"])
            expected = compute_sid(record["name"], record["overload"], record["role"], record["tree"])
        except ValueError as exc:
            raise SchemaError(str(exc)) from exc
        _require(record["sid"] == expected, "sig.sid disagrees with canonical content")


def validate_record(record: Mapping[str, Any], defaults: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    return SchemaValidator().decode(record, defaults)
