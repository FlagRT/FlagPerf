"""Native XPYTORCH adapter bound to the current container's verified UUID."""
import hashlib
import importlib
import json
import os
from pathlib import Path

import torch

from base.vendors.kunlunxin.provider import binding_records


_binding = None
_identity = None


def initialize():
    global _binding, _identity
    context_path = Path(os.environ['FLAGPERF_HOST_CONTEXT'])
    raw = context_path.read_bytes()
    context = json.loads(raw)
    binding_path = Path(os.environ['FLAGPERF_RUNTIME_BINDINGS'])
    record = json.loads(binding_path.read_text())
    context_hash = hashlib.sha256(raw).hexdigest()
    if context.get('kind') != 'benchmark' or record.get('run_id') != context.get('run_id') or record.get('context_sha256') != context_hash:
        raise RuntimeError('P800 runtime binding belongs to a different context/run')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('USE_FLAGGEMS') != '0':
        raise RuntimeError('P800 runtime environment drift')
    if any(name in os.environ for name in ('XPU_VISIBLE_DEVICES', 'XPU_EVENT_KL3_ENABLE')):
        raise RuntimeError('unqualified P800 visibility/event mode')
    importlib.import_module('torch_xmlir')
    if not torch.cuda.is_available():
        raise RuntimeError('P800 runtime unavailable')
    observed = [str(torch.cuda.get_device_properties(index).uuid).lower() for index in range(torch.cuda.device_count())]
    expected = binding_records(context['host']['devices'], observed)
    if len(expected) != 1 or record.get('observed_uuids') != observed or record.get('bindings') != expected:
        raise RuntimeError('P800 worker UUID set differs from the container binding')
    identity = (context['run_id'], context_hash, expected[0]['serial_or_uuid'])
    if _identity is not None and identity != _identity:
        raise RuntimeError('P800 driver cannot switch context after initialization')
    if _binding is None:
        torch.cuda.set_device(expected[0]['framework_logical_id'])
    _binding, _identity = expected[0], identity
    return _binding


def device(local_rank=0):
    if _binding is None:
        initialize()
    if local_rank != _binding['framework_local_rank']:
        raise RuntimeError('rank differs from verified P800 binding')
    return torch.device(_binding['framework_device_name'])


def synchronize():
    torch.cuda.synchronize(device())


def seed(value):
    torch.cuda.manual_seed_all(value)


def memory_info():
    return torch.cuda.mem_get_info(device())


def is_out_of_memory(error):
    return isinstance(error, torch.OutOfMemoryError)


def set_float32():
    torch.backends.cuda.matmul.allow_tf32 = False


def evidence():
    initialize()
    return {'binding': _binding, 'run_id': _identity[0], 'context_sha256': _identity[1],
            'worker_pid': os.getpid(), 'worker_parent_pid': os.getppid(),
            'pid_namespace': os.readlink('/proc/self/ns/pid'),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'precision_scope': 'native FP32 inputs/outputs; CUDA control does not certify internal IEEE arithmetic',
            'fallback_scope': 'UUID, tensor placement and locked native route; no universal fallback exclusion'}
