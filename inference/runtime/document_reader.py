"""Bounded legacy-document decoding; derived records never modify the source."""
from collections import OrderedDict
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import time

from runtime.artifacts import Store, Records
from runtime.stream_json import read_document
from runtime.verification import SESSION_ENV, identity, readonly

_cache = OrderedDict()
_scope = None
_bytes = 0
BUDGET = 256*1024*1024


def clear():
    global _scope, _bytes
    _cache.clear(); _scope, _bytes = None, 0


class TemporaryStore(Store):
    def __init__(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='json-records-', dir=directory)
        super().__init__(self.temporary.name)

    def __del__(self):
        try: self.temporary.cleanup()
        except Exception: pass


def _size(value):
    if isinstance(value, Records): return 512
    if isinstance(value, dict): return 128 + sum(len(str(k))*4 + _size(v) for k, v in value.items())
    if isinstance(value, list): return 64 + sum(_size(v) for v in value)
    if isinstance(value, str): return 64 + len(value)*4
    return 32


def read(path):
    global _scope, _bytes
    path = Path(path).resolve()
    scope = os.environ.get(SESSION_ENV)
    if scope != _scope:
        clear(); _scope = scope
    before = identity(path.stat())
    reusable = scope and readonly(path) and time.time_ns()-max(before[3], before[4]) >= 2_000_000_000
    key = (str(path), before)
    if reusable and key in _cache:
        _cache.move_to_end(key)
        return deepcopy(_cache[key][0])
    from runtime.tensor_io import work_directory
    store = TemporaryStore(work_directory())
    value = read_document(path, store)
    if identity(path.stat()) != before: raise ValueError('JSON document changed during decoding')
    if reusable:
        size = _size(value)
        while _cache and _bytes + size > BUDGET:
            _, (_, removed) = _cache.popitem(last=False); _bytes -= removed
        if size <= BUDGET:
            _cache[key] = (deepcopy(value), size); _bytes += size
    return value
