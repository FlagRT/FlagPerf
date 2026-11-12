# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Training v2 entry, sharing the installed inference profiling infrastructure.

The historical run_benchmarks/run.py entry remains independent.
"""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'inference'))
from flagprof.training.__main__ import main

if __name__=='__main__':
    raise SystemExit(main())
