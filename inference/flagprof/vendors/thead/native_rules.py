# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Reuse the explicitly shared Kineto native_rules syntax; PPU identity remains separate."""
from flagprof.vendors.nvidia.native_rules import build_native_rules, VERSION
from .format import NATIVE_CALL_CATEGORIES, ANNOTATION_CATEGORIES
