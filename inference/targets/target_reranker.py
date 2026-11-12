# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Vendor-agnostic reranker attach target (thin attach CLI).

Canonical implementation: flagprof/runner/rerank.py. This entry keeps the
attach contract (same flags, target-result.json surface) so existing
attach evidence chains stay comparable with runner-task runs.
"""
from flagprof.runner.rerank import attach_main

if __name__ == '__main__':
    attach_main()
