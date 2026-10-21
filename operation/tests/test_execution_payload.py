# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""CPU negative controls for the real P800 pre-call input audit."""
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vendors.kunlunxin.execution import tensor_payload, verify_payload
from runtime.evidence import read


class PayloadTests(unittest.TestCase):
    def test_scalar_bfloat16_and_noncontiguous_payloads(self):
        import torch
        for x in (torch.tensor(-0.), torch.tensor(float('nan')), torch.empty(0),
                  torch.arange(8).to(torch.bfloat16), torch.arange(12).reshape(3, 4).t()):
            self.assertEqual(tensor_payload(x), tensor_payload(x.clone()))
        self.assertNotEqual(tensor_payload(torch.tensor(0.)), tensor_payload(torch.tensor(-0.)))

    def test_wrong_input_state_or_device_fails_before_target_can_be_called(self):
        import torch
        for mutation in ('none', 'bytes', 'shape', 'dtype', 'gradient', 'state', 'state-name', 'device', 'count'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                model = torch.nn.Linear(2, 2)
                original = torch.tensor([1., 2.], requires_grad=True)
                bundle = {'inputs': [original], 'state': {k: v.clone() for k, v in model.state_dict().items()}}
                torch.save(bundle, root / 'inputs.pt')
                inputs = [original.detach().clone().requires_grad_(True)]
                if mutation == 'bytes': inputs[0].data[0] = 3
                if mutation == 'shape': inputs[0] = inputs[0].reshape(1, 2)
                if mutation == 'dtype': inputs[0] = inputs[0].double()
                if mutation == 'gradient': inputs[0] = inputs[0].detach()
                if mutation == 'count': inputs.append(original)
                if mutation == 'state': model.weight.data[0, 0] += 1
                if mutation == 'state-name': bundle['state']['wrong'] = bundle['state'].pop('weight')
                adapter = SimpleNamespace(device=torch.device('cuda:0' if mutation == 'device' else 'cpu'), synchronize=Mock())
                if mutation == 'none':
                    verify_payload(root, adapter, bundle, inputs, model, 'probe')
                else:
                    with self.assertRaisesRegex(ValueError, 'execution payload'):
                        verify_payload(root, adapter, bundle, inputs, model, 'probe')
                record = read(root / 'probe-payload.json')
                self.assertEqual(record['status'], 'passed' if mutation == 'none' else 'failed')
                self.assertEqual((root / 'input-roundtrip.json').exists(), mutation == 'none')
                adapter.synchronize.assert_called_once()


class WorkerPayloadTests(unittest.TestCase):
    def test_payload_gate_precedes_registration_and_preserves_real_outputs(self):
        import torch
        from runtime import worker
        for phase in ('probe', 'measure'):
            for kind in ('pass', 'wrong-result', 'compile-error', 'bad-payload'):
                with self.subTest(phase=phase, kind=kind), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    x = torch.tensor([1., -2.], requires_grad=True)
                    torch.save({'inputs': [x], 'state': None}, root / 'inputs.pt')
                    torch.save(x.detach().abs().double(), root / 'reference.pt')
                    calls = []
                    def target(value):
                        self.assertTrue((root / (phase + '-payload.json')).is_file())
                        calls.append(True)
                        if kind == 'compile-error':
                            raise RuntimeError('compiler failure fixture')
                        return value.abs() + (1 if kind == 'wrong-result' else 0)
                    adapter = SimpleNamespace(device=torch.device('cpu'),
                        bootstrap_task=lambda task: torch.device('cpu'), synchronize=lambda: None,
                        identity=lambda: {}, kernel_time=lambda call, cfg: None)
                    def audit(*args):
                        if kind == 'bad-payload':
                            args[2][0] = args[2][0].detach()  # requires_grad was lost
                        verify_payload(args[0], adapter, *args[1:])
                    adapter.verify_execution_payload = audit
                    task = {'vendor': 'kunlunxin', 'runtime_profile': 'p800-xpytorch29',
                        'oplib': 'flaggems', 'case': 'abs', 'dtype': 'FP32',
                        'case_config': {'WARMUP': 0, 'rounds': 1, 'ITERS': 1}}
                    # The stand-in registration happens only after the real worker gate.
                    gems = SimpleNamespace(enable=Mock(side_effect=lambda: self.assertTrue(
                        (root / (phase + '-payload.json')).is_file())))
                    with patch('vendors.get_vendor', return_value=adapter), \
                         patch.object(worker, 'build', return_value=(target, [x], False, lambda n: n)), \
                         patch.object(worker, '_GEMS_ENABLED', False), \
                         patch.dict(sys.modules, {'flag_gems': gems}):
                        if kind in ('compile-error', 'bad-payload'):
                            with self.assertRaises((RuntimeError, ValueError)):
                                worker.device_phase(root, task, phase)
                        else:
                            worker.device_phase(root, task, phase)
                    if kind == 'bad-payload':
                        self.assertFalse(calls)
                        gems.enable.assert_not_called()
                    else:
                        self.assertTrue(calls)
                        gems.enable.assert_called_once()
                    if kind in ('pass', 'wrong-result'):
                        name = 'output.pt' if phase == 'probe' else 'measurement-output.pt'
                        out = torch.load(root / name, weights_only=True)
                        self.assertTrue(torch.equal(out, x.detach().abs() + (kind == 'wrong-result')))
                        record = read(root / ('correctness.json' if phase == 'probe' else 'measurement.json'))
                        check = record if phase == 'probe' else record['correctness']
                        self.assertEqual(check['status'], 'passed' if kind == 'pass' else 'failed')


if __name__ == '__main__':
    unittest.main()
