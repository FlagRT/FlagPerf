# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Start flagprof modules from the same installation or source checkout as the caller."""
import os
from pathlib import Path

from .progress import child_env

IMPORT_ROOT = str(Path(__file__).resolve().parent.parent)


def module_environment():
    env = child_env()
    existing = env.get('PYTHONPATH')
    paths = existing.split(os.pathsep) if existing else []
    env['PYTHONPATH'] = os.pathsep.join([IMPORT_ROOT, *(p for p in paths if p != IMPORT_ROOT)])
    return env
