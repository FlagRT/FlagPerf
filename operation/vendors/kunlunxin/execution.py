# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Audit P800 saved inputs before registration or target execution."""
import hashlib
import ctypes

from runtime.evidence import write, sha


def tensor_payload(tensor):
    value = tensor.detach().cpu().contiguous()
    raw = ctypes.string_at(value.data_ptr(), value.numel() * value.element_size())
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'sha256': hashlib.sha256(raw).hexdigest(),
            'requires_grad': bool(tensor.requires_grad)}


def payload_matches(expected, observed):
    return expected == observed


def verify_payload(root, adapter, bundle, inputs, fn, phase):
    import torch
    from runtime.worker import tensors
    adapter.synchronize()
    expected = [tensor_payload(t) for t in tensors(bundle['inputs'])]
    actual = [tensor_payload(t) for t in tensors(inputs)]
    expected_state = {k: tensor_payload(t) for k, t in (bundle['state'] or {}).items()}
    actual_state = ({k: tensor_payload(t) for k, t in fn.state_dict().items()}
                    if isinstance(fn, torch.nn.Module) else {})
    input_ok, state_ok = payload_matches(expected, actual), payload_matches(expected_state, actual_state)
    devices = [str(t.device) for t in tensors(inputs)]
    if isinstance(fn, torch.nn.Module):
        devices += [str(t.device) for t in fn.state_dict().values()]
    device_ok = bool(devices) and all(d == str(adapter.device) for d in devices)
    record = {'protocol': 'p800-execution-payload-v1', 'input_sha256': sha(root / 'inputs.pt'),
              'status': 'passed' if input_ok and state_ok and device_ok else 'failed',
              'inputs': {'passed': input_ok, 'expected': expected, 'observed': actual},
              'module_state': {'passed': state_ok, 'expected': expected_state, 'observed': actual_state},
              'device': str(adapter.device), 'observed_devices': devices, 'device_passed': device_ok,
              'phase': phase, 'boundary': 'bitwise logical tensor values after transfer; before current target calls and initial explicit Gems registration; reused workers may already be registered; outside timing'}
    write(root / (phase + '-payload.json'), record)
    if record['status'] != 'passed':
        raise ValueError('Saved input/module state or bound device differs from execution payload')
    write(root / 'input-roundtrip.json', {'passed': True, 'input_sha256': record['input_sha256'],
                                         'protocol': record['protocol']})
