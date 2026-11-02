# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Bounded validation plans, derived from the reviewed B2X mechanism.

Values are checked on every call. Unusual shapes (including link) use the
canonical validator. This is internal; SchemaValidator remains authoritative.
"""
import sys
from . import schema as S
from .schema import SchemaValidator

_FAST_TYPES = {"op", "span", "route", "compile", "health", "counter", "opsum", "kernel", "memop", "comm"}
_INT_KEYS = {"seq", "pid", "tid", "rank", "scope", "seq_in_scope", "parent", "depth", "n", "dur_sum",
             "group_seq", "bytes", "count"}
_U64_KEYS = {"corr", "parent_corr"}
_STR_KEYS = {"name", "lib", "backend", "key", "boundary", "dtype", "probe_alg", "p", "clk"}
_DICT_KEYS = {"ext", "data", "attrs", "stats", "values", "metrics", "stage_dur"}
_U64 = S.U64_MAX


class CachedValidator(SchemaValidator):
    MAX_ENTRIES = 64
    MAX_BYTES = 256 << 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._plans = {}
        self.cache_bytes = sys.getsizeof(self._plans)

    def _plan(self, t, keys, record, defaults):
        known = set(S.REQUIRED_FIELDS[t]) | S.OPTIONAL_FIELDS[t] | S.ENVELOPE
        unknown = len(keys - known)
        add = tuple(k for k in (defaults or {}) if k in known and k not in keys)
        result_keys = (keys & known) | set(add)
        required = set(S.REQUIRED_FIELDS[t])
        if t == "route" and record.get("decision") == "hit":
            required.discard("reason")
        if t == "span" and record.get("ph") == "e":
            required -= {"cat", "name"}
        if t == "counter" and record.get("src") != "device":
            required.discard("dev")
        if not required <= result_keys or "seq" not in result_keys or "ts" not in result_keys:
            return None
        if t == "span" and record.get("ph") == "x" and "dur" not in result_keys:
            return None
        enums = tuple((k, frozenset(v)) for k, v in S.ENUM_FIELDS.get(t, {}).items() if k in result_keys)
        return dict(
            filter=bool(unknown), known=frozenset(known), unknown=unknown, add=add, enums=enums,
            ints=tuple(sorted(_INT_KEYS & result_keys)), u64=tuple(sorted(_U64_KEYS & result_keys)),
            dur=("dur" in result_keys), dur_null_ok=t in {"kernel", "memop", "comm"},
            sids=tuple(sorted({"sid", "osid"} & result_keys)), strs=tuple(sorted(_STR_KEYS & result_keys)),
            dicts=tuple(sorted(_DICT_KEYS & result_keys)), opsum=t == "opsum")

    def decode(self, record, defaults=None):
        if type(record) is not dict:
            return super().decode(record, defaults)
        t = record.get("t")
        if type(t) is not str or t not in _FAST_TYPES or "v" in record or "link" in record:
            return super().decode(record, defaults)
        if defaults is not None and (type(defaults) is not dict or
                not set(defaults) <= {"pid", "tid", "rank", "p", "clk", "ts", "seq"}):
            return super().decode(record, defaults)
        keys = frozenset(record)
        if len(keys) > 64 or any(type(k) is not str for k in keys):
            return super().decode(record, defaults)
        disc = (record.get("decision") if t == "route" else record.get("ph") if t == "span"
                else record.get("src") if t == "counter" else None)
        if type(disc) is not str and disc is not None:
            return super().decode(record, defaults)
        cache_key = (t, keys, disc, frozenset(defaults or {}))
        plan = self._plans.get(cache_key, False)
        if plan is False:
            if len(self._plans) >= self.MAX_ENTRIES:
                return super().decode(record, defaults)
            plan = self._plan(t, keys, record, defaults)
            charge = _size((cache_key, plan)) + 256
            if self.cache_bytes + charge > self.MAX_BYTES:
                return super().decode(record, defaults)
            self._plans[cache_key] = plan
            self.cache_bytes += charge
        if plan is None:
            return super().decode(record, defaults)
        known = plan["known"]
        result = {k: v for k, v in record.items() if k in known} if plan["filter"] else dict(record)
        for k in plan["add"]:
            result[k] = defaults[k]
        replaced = 0
        for k, values in plan["enums"]:
            v = result[k]
            if type(v) is not str:
                return super().decode(record, defaults)
            if v not in values:
                result[k] = "other"
                replaced += 1
        ts = result["ts"]
        if ts is not None and not (type(ts) is int and ts >= 0):
            return super().decode(record, defaults)
        for k in plan["ints"]:
            v = result[k]
            if not (type(v) is int and v >= 0):
                return super().decode(record, defaults)
        for k in plan["u64"]:
            v = result[k]
            if not (type(v) is int and 0 <= v <= _U64):
                return super().decode(record, defaults)
        if plan["dur"]:
            v = result["dur"]
            if not ((type(v) is int and v >= 0) or (plan["dur_null_ok"] and v is None)):
                return super().decode(record, defaults)
        for k in plan["sids"]:
            v = result[k]
            if type(v) is not str:
                return super().decode(record, defaults)
            if not S.SID_RE.fullmatch(v):
                return super().decode(record, defaults)
        for k in plan["strs"]:
            if not isinstance(result[k], str):
                return super().decode(record, defaults)
        for k in plan["dicts"]:
            if not isinstance(result[k], dict):
                return super().decode(record, defaults)
        if plan["opsum"]:
            return super().decode(record, defaults)
        self.unknown_fields += plan["unknown"]
        self.unknown_enums += replaced
        return result


def _size(value, seen=None):
    """Conservative recursive cache charge; sharing only reduces actual usage."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        total += sum(_size(k, seen)+_size(v, seen) for k, v in value.items())
    elif isinstance(value, (tuple, list, set, frozenset)):
        total += sum(_size(v, seen) for v in value)
    return total
