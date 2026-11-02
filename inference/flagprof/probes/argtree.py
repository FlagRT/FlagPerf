"""Describe dispatcher values without reading tensor data or storing devices.

Dispatcher normalization is authoritative: no defaults are inserted and no
positional arguments are moved. Unsupported metadata safely falls back to
``other``; unresolved SymInt/non-string-key dictionaries/special layouts remain
pending designer cases in the golden corpus, not newly frozen encodings.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

Node = dict[str, Any]
Children = list[tuple[str, Any]]
Describe = Callable[[str, Any], tuple[Node, Children]]


class _UndefinedEncoding(Exception):
    """The contract deliberately does not define this value's metadata yet."""


def _qualified_type(value: Any) -> str:
    cls = type(value)
    try:
        return f"{cls.__module__}.{cls.__qualname__}"
    except Exception:
        return "builtins.object"


def _other(path: str, value: Any) -> tuple[Node, Children]:
    return dict(p=path, k="other", type=_qualified_type(value)), []


def _tensor(path: str, value: torch.Tensor) -> tuple[Node, Children]:
    if value.layout != torch.strided or value.is_quantized:
        raise _UndefinedEncoding("special-layout metadata awaits a designer decision")
    shape = list(value.shape)
    if any(type(dim) is not int for dim in shape):
        raise _UndefinedEncoding("symbolic dimensions await a designer decision")
    contiguous = value.is_contiguous()
    node = dict(p=path, k="tensor", shape=shape, dtype=str(value.dtype),
                layout=str(value.layout), contiguous=contiguous)
    if not contiguous:
        stride = list(value.stride())
        if any(type(dim) is not int for dim in stride):
            raise _UndefinedEncoding("symbolic strides await a designer decision")
        node["stride"] = stride
    return node, []


def _metadata(kind: str) -> Describe:
    def describe(path: str, value: Any) -> tuple[Node, Children]:
        return dict(p=path, k=kind, v=str(value)), []
    return describe


def _scalar(kind: str) -> Describe:
    def describe(path: str, value: Any) -> tuple[Node, Children]:
        expected = {"bool": bool, "int": int, "float": float, "str": str}[kind]
        if type(value) is not expected:
            return _other(path, value)
        return dict(p=path, k=kind, v=repr(value) if kind == "float" else value), []
    return describe


def _device(path: str, value: torch.device) -> tuple[Node, Children]:
    return dict(p=path, k="device_obj"), []


def _none(path: str, value: None) -> tuple[Node, Children]:
    return dict(p=path, k="none"), []


def _symint(path: str, value: Any) -> tuple[Node, Children]:
    raise _UndefinedEncoding("SymInt encoding awaits a designer decision")


def _sequence(kind: str) -> Describe:
    def describe(path: str, value: Any) -> tuple[Node, Children]:
        return (dict(p=path, k=kind, n=len(value)),
                [(f"{path}[{index}]", item) for index, item in enumerate(value)])
    return describe


def _mapping(path: str, value: dict[str, Any]) -> tuple[Node, Children]:
    if any(type(key) is not str for key in value):
        raise _UndefinedEncoding("non-string dictionary paths await a designer decision")
    return dict(p=path, k="dict", n=len(value)), [(f"{path}{{{key}}}", value[key]) for key in sorted(value)]


# Torch-specific knowledge is isolated in this table. bool precedes int.
_DISPATCH: tuple[tuple[type, Describe], ...] = (
    (torch.Tensor, _tensor), (torch.dtype, _metadata("dtype")),
    (torch.layout, _metadata("layout")), (torch.memory_format, _metadata("memory_format")),
    (torch.device, _device), (torch.SymInt, _symint),
    (type(None), _none), (bool, _scalar("bool")), (int, _scalar("int")),
    (float, _scalar("float")), (str, _scalar("str")),
    (list, _sequence("list")), (tuple, _sequence("tuple")), (dict, _mapping),
)


def _walk(roots: list[tuple[str, Any]]) -> list[Node]:
    result: list[Node] = []
    active_containers: set[int] = set()
    pending: list[tuple[bool, str, Any]] = [(False, path, value) for path, value in reversed(roots)]
    while pending:
        leaving, path, value = pending.pop()
        if leaving:
            active_containers.discard(id(value))
            continue
        try:
            is_container = isinstance(value, (list, tuple, dict))
            if is_container and id(value) in active_containers:
                node, children = _other(path, value)
            else:
                handler = next((describe for cls, describe in _DISPATCH if isinstance(value, cls)), _other)
                node, children = handler(path, value)
            result.append(node)
            if children:
                active_containers.add(id(value))
                pending.append((True, path, value))
                pending.extend((False, child_path, child) for child_path, child in reversed(children))
        except Exception:
            # Metadata access, unknown types and user container subclasses can
            # fail; capture still must not terminate the measured invocation.
            node, _ = _other(path, value)
            result.append(node)
    return result


def describe_args(args: Sequence[Any], kwargs: Mapping[str, Any] | None = None) -> list[Node]:
    """Return a recursive preorder tree of dispatcher args and sorted kwargs."""
    try:
        roots = [(str(index), value) for index, value in enumerate(args)]
        if kwargs:
            if any(type(key) is not str for key in kwargs):
                return []
            roots.extend((f"kw:{key}", kwargs[key]) for key in sorted(kwargs))
        return _walk(roots)
    except Exception:
        return []


def describe_result(result: Any) -> list[Node]:
    """Describe one returned object under r, recursively including containers."""
    try:
        return _walk([("r", result)])
    except Exception:
        return []
