"""Per-task CPU tensor decoding shared by all accuracy pairings.

Large decoded tensors spill to private memory-mapped arrays. Source artifacts
remain untouched and are checked for replacement on every access.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import os
from pathlib import Path
import tempfile

from runtime.verification import identity

_session = ContextVar('tensor_read_session', default=None)


def work_directory():
    selected = _session.get()
    if selected is not None: return selected.root.parent
    root = os.environ.get('FLAGPERF_TASK_WORKSPACE')
    return Path(root) if root else Path(tempfile.gettempdir())


@contextmanager
def session(root, budget_bytes=256*1024*1024):
    work = Path(root)/'.work'; work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='tensors-', dir=work) as temporary:
        cache = Cache(temporary, budget_bytes)
        token = _session.set(cache)
        try: yield cache
        finally: _session.reset(token)


def _containers(value):
    if isinstance(value, dict): return {k: _containers(v) for k, v in value.items()}
    if isinstance(value, list): return [_containers(v) for v in value]
    if isinstance(value, tuple): return tuple(_containers(v) for v in value)
    return value


class Cache:
    def __init__(self, root, budget):
        self.root, self.budget = Path(root), budget
        self.entries, self.memory, self.bytes = {}, OrderedDict(), 0
        self.reads, self.hits, self.spilled_bytes = 0, 0, 0
        self.next_id = 0

    def _spill(self, value):
        import torch
        import numpy as np
        if isinstance(value, torch.Tensor):
            tensor = value.detach().contiguous()
            path = self.root/(str(self.next_id)+'.npy'); self.next_id += 1
            # uint8 preserves BF16 and all original bit patterns without casts.
            raw = tensor.reshape(-1).view(torch.uint8).numpy()
            array = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=raw.shape)
            array[:] = raw; array.flush(); del array
            self.spilled_bytes += raw.nbytes
            return ('tensor', str(path), tensor.dtype, tuple(tensor.shape))
        if isinstance(value, dict): return ('dict', [(k, self._spill(v)) for k, v in value.items()])
        if isinstance(value, list): return ('list', [self._spill(v) for v in value])
        if isinstance(value, tuple): return ('tuple', [self._spill(v) for v in value])
        return ('scalar', value)

    def _restore(self, value):
        kind, *parts = value
        if kind == 'tensor':
            import numpy as np
            import torch
            path, dtype, shape = parts
            # Private copy-on-write mappings cannot mutate the cached arrays.
            return torch.from_numpy(np.load(path, mmap_mode='c')).view(dtype).reshape(shape)
        if kind == 'dict': return {k: self._restore(v) for k, v in parts[0]}
        if kind == 'list': return [self._restore(v) for v in parts[0]]
        if kind == 'tuple': return tuple(self._restore(v) for v in parts[0])
        return parts[0]

    def load(self, path):
        import torch
        path = Path(path).resolve()
        stamp = identity(path.stat()); key = str(path)
        if key in self.entries:
            before, saved = self.entries[key]
            if before != stamp:
                raise ValueError('accuracy input changed within task: ' + key)
            self.hits += 1
            if key in self.memory:
                self.memory.move_to_end(key)
                return _containers(self.memory[key][0])
            return self._restore(saved)
        value = torch.load(path, map_location='cpu', weights_only=True)
        if identity(path.stat()) != stamp: raise ValueError('tensor changed while decoding: ' + key)
        self.reads += 1
        def size(item):
            if isinstance(item, torch.Tensor): return item.numel()*item.element_size()
            if isinstance(item, dict): return sum(size(v) for v in item.values())
            if isinstance(item, (list, tuple)): return sum(size(v) for v in item)
            return 0
        amount = size(value)
        while self.memory and self.bytes + amount > self.budget:
            old, (item, n) = self.memory.popitem(last=False)
            previous, _ = self.entries[old]
            self.entries[old] = (previous, self._spill(item)); self.bytes -= n
        if amount <= self.budget:
            self.memory[key] = (value, amount); self.bytes += amount
            self.entries[key] = (stamp, None)
        else:
            saved = self._spill(value)
            self.entries[key] = (stamp, saved)
            value = self._restore(saved)
        return _containers(value)


def load_cpu(path):
    selected = _session.get()
    if selected is not None: return selected.load(path)
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)


def concatenate(values):
    """Preserve concatenation order and dtype; spill large aggregate boundaries."""
    import numpy as np
    selected = _session.get()
    total = sum(value.size for value in values)
    dtype = np.result_type(*(value.dtype for value in values))
    if selected is None or total*dtype.itemsize <= selected.budget//4:
        return np.concatenate(values)
    path = selected.root/('aggregate-'+str(selected.next_id)+'.npy'); selected.next_id += 1
    result = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=(total,))
    position = 0
    for value in values:
        result[position:position+value.size] = value.reshape(-1)
        position += value.size
    result.flush()
    selected.spilled_bytes += result.nbytes
    return result
