"""Frozen, device-free signature encoding (03 section 4.2 and D17).

Only the standard library is imported. Truncation belongs to the stored sig
record; a truncated record must retain the sid computed from its complete tree.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

Node = dict[str, Any]
Tree = list[Node]
_POSITION = re.compile(r"0|[1-9][0-9]*")
_FIELDS = {
    "tensor": {"p", "k", "shape", "dtype", "layout", "contiguous"},
    "list": {"p", "k", "n"}, "tuple": {"p", "k", "n"},
    "dict": {"p", "k", "n"}, "none": {"p", "k"},
    "bool": {"p", "k", "v"}, "int": {"p", "k", "v"},
    "str": {"p", "k", "v"}, "float": {"p", "k", "v"},
    "dtype": {"p", "k", "v"}, "layout": {"p", "k", "v"},
    "memory_format": {"p", "k", "v"}, "device_obj": {"p", "k"},
    "other": {"p", "k", "type"},
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any) -> bool:
    return type(value) is int


def _dimensions(value: Any, *, nonnegative: bool) -> bool:
    return (type(value) is list and
            all(_integer(x) and (not nonnegative or x >= 0) for x in value))


def _validate_node(node: Any, index: int) -> None:
    label = f"tree[{index}]"
    _require(type(node) is dict, f"{label}: node must be a dict")
    _require(isinstance(node.get("p"), str) and bool(node["p"]),
             f"{label}: p must be a nonempty string")
    kind = node.get("k")
    _require(isinstance(kind, str) and kind in _FIELDS,
             f"{label}: unknown node kind")
    fields = _FIELDS[kind]
    if kind == "tensor" and node.get("contiguous") is False:
        fields = fields | {"stride"}
    _require(set(node) == fields, f"{label}: fields must be {sorted(fields)}")
    if kind in ("list", "tuple", "dict"):
        _require(_integer(node["n"]) and node["n"] >= 0,
                 f"{label}: n must be a nonnegative integer")
    elif kind == "tensor":
        _require(_dimensions(node["shape"], nonnegative=True),
                 f"{label}: shape must contain concrete nonnegative integers")
        _require(type(node["contiguous"]) is bool,
                 f"{label}: contiguous must be bool")
        for field in ("dtype", "layout"):
            _require(isinstance(node[field], str) and node[field].startswith("torch."),
                     f"{label}: {field} must be a canonical torch name")
        if not node["contiguous"]:
            _require(_dimensions(node["stride"], nonnegative=False) and
                     len(node["stride"]) == len(node["shape"]),
                     f"{label}: stride must have one integer per dimension")
    elif kind in ("bool", "int", "str"):
        expected = {"bool": bool, "int": int, "str": str}[kind]
        _require(type(node["v"]) is expected,
                 f"{label}: {kind}.v has the wrong type")
    elif kind == "float":
        value = node["v"]
        _require(isinstance(value, str), f"{label}: float.v must be a repr string")
        try:
            valid = repr(float(value)) == value
        except (ValueError, OverflowError):
            valid = False
        _require(valid, f"{label}: float.v must be a Python float repr")
    elif kind in ("dtype", "layout", "memory_format"):
        _require(isinstance(node["v"], str) and node["v"].startswith("torch."),
                 f"{label}: metadata v must be a canonical torch name")
    elif kind == "other":
        _require(isinstance(node["type"], str) and "." in node["type"],
                 f"{label}: other.type must be a qualified type name")


def validate_tree(tree: Tree) -> None:
    """Validate node fields, container arity, paths and preorder; raise ValueError.

    The walk is iterative so container nesting has no implementation-imposed
    recursion limit. Unknown fields are rejected: in particular neither tensor
    nor device_obj may accidentally acquire device metadata.
    """
    _require(type(tree) is list, "tree must be a node array")
    stack: list[dict[str, Any]] = []
    seen: set[str] = set()
    next_position = 0
    last_keyword: str | None = None
    root_kind: str | None = None
    for index, node in enumerate(tree):
        _validate_node(node, index)
        path, kind = node["p"], node["k"]
        _require(path not in seen, f"tree[{index}]: duplicate path {path}")
        seen.add(path)
        if stack:
            parent = stack[-1]
            if parent["kind"] in ("list", "tuple"):
                expected = f"{parent['path']}[{parent['next']}]"
                _require(path == expected, f"tree[{index}]: expected child path {expected}")
            else:
                prefix = parent["path"] + "{"
                _require(path.startswith(prefix) and path.endswith("}"),
                         f"tree[{index}]: expected a dict child of {parent['path']}")
                key = path[len(prefix):-1]
                _require(parent["last_key"] is None or parent["last_key"] < key,
                         f"tree[{index}]: dict keys must be sorted and unique")
                parent["last_key"] = key
            parent["next"] += 1
            parent["remaining"] -= 1
        elif _POSITION.fullmatch(path):
            _require(root_kind != "out" and last_keyword is None and int(path) == next_position,
                     f"tree[{index}]: positional roots must precede kwargs and be contiguous")
            root_kind = "in"
            next_position += 1
        elif path.startswith("kw:") and len(path) > 3:
            keyword = path[3:]
            _require(root_kind != "out" and (last_keyword is None or last_keyword < keyword),
                     f"tree[{index}]: keyword roots must be sorted and unique")
            root_kind = "in"
            last_keyword = keyword
        else:
            _require(path == "r" and root_kind is None,
                     f"tree[{index}]: invalid or mixed input/output root {path}")
            root_kind = "out"
        if kind in ("list", "tuple", "dict") and node["n"]:
            stack.append(dict(path=path, kind=kind, remaining=node["n"], next=0, last_key=None))
        while stack and stack[-1]["remaining"] == 0:
            stack.pop()
    _require(not stack, "tree ends before all container children are present")


def canonical_bytes(name: str, overload: str, role: str, tree: Tree) -> bytes:
    """Encode the frozen four-field signature as sorted, compact UTF-8 JSON."""
    _require(isinstance(name, str) and bool(name), "name must be a nonempty string")
    _require(isinstance(overload, str), "overload must be a string")
    _require(role in ("in", "out"), "role must be in or out")
    validate_tree(tree)
    if tree:
        is_output = tree[0]["p"] == "r"
        _require(is_output == (role == "out"), "tree root does not match signature role")
    return json.dumps(dict(name=name, overload=overload, role=role, tree=tree),
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def compute_sid(name: str, overload: str, role: str, tree: Tree) -> str:
    """Return the first 64 hash bits as exactly 16 lowercase hex characters."""
    return hashlib.sha256(canonical_bytes(name, overload, role, tree)).hexdigest()[:16]
