# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Independent fixed-shape graph export and torch-free artifact reading."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExportSpec:
    """CPU eval module, example args/kwargs and ordered flat Tensor I/O names.

    Output may be a Tensor or a flat tuple/list of Tensors. Input names follow
    tensor leaves in (args, kwargs) order; scalar arguments are specialized.
    identity describes provenance, not a claim of device execution equivalence.
    """
    model: Any
    args: tuple
    input_names: tuple
    output_names: tuple
    kwargs: dict = field(default_factory=dict)
    identity: dict = field(default_factory=dict)


def export_graph(spec, out, *, formats=('fx', 'onnx'), bundle='structure', verify_onnx=False):
    from .core import export_graph as implementation
    return implementation(spec, out, formats=formats, bundle=bundle, verify_onnx=verify_onnx)


def verify_graph(path):
    from .data import verify_graph as implementation
    return implementation(path)


def render_graph(path, out=None):
    from .view import render_graph as implementation
    return implementation(path, out)
