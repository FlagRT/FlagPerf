"""Bitwise input/module/device audit for PPU target phases."""
import ctypes
import hashlib

from runtime.evidence import sha, write


def tensor_payload(tensor):
    value = tensor.detach().cpu().contiguous()
    raw = ctypes.string_at(value.data_ptr(), value.numel() * value.element_size())
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'sha256': hashlib.sha256(raw).hexdigest(), 'requires_grad': bool(tensor.requires_grad)}


def verify_payload(root, adapter, bundle, inputs, fn, phase):
    import torch
    from runtime.worker import tensors
    adapter.synchronize()
    expected = [tensor_payload(t) for t in tensors(bundle['inputs'])]
    actual = [tensor_payload(t) for t in tensors(inputs)]
    expected_state = {k: tensor_payload(t) for k, t in (bundle['state'] or {}).items()}
    actual_state = ({k: tensor_payload(t) for k, t in fn.state_dict().items()}
                    if isinstance(fn, torch.nn.Module) else {})
    devices = [str(t.device) for t in tensors(inputs)]
    if isinstance(fn, torch.nn.Module):
        devices += [str(t.device) for t in fn.state_dict().values()]
    device_ok = bool(devices) and all(d == str(adapter.device) for d in devices)
    passed = expected == actual and expected_state == actual_state and device_ok
    record = {'protocol': 'thead-ppu-execution-payload-v1', 'input_sha256': sha(root / 'inputs.pt'),
              'status': 'passed' if passed else 'failed', 'inputs': {'passed': expected == actual,
              'expected': expected, 'observed': actual}, 'module_state': {'passed': expected_state == actual_state,
              'expected': expected_state, 'observed': actual_state}, 'device': str(adapter.device),
              'observed_devices': devices, 'device_passed': device_ok, 'phase': phase,
              'boundary': 'logical tensor values after transfer; before target calls'}
    write(root / (phase + '-payload.json'), record)
    if not passed:
        raise ValueError('Saved input/module state or bound PPU device differs from execution payload')
