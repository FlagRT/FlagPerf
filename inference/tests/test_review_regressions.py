# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Public-entry regressions for the model-adaptation review findings."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class VendorIdentityTests(unittest.TestCase):
    def test_runner_help_does_not_require_torch(self):
        result = subprocess.run([sys.executable, '-B', '-S', '-m', 'flagprof.runner', '--help'],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--arch', result.stdout)

    def test_nested_code_change_changes_identity_and_loaded_implementation(self):
        from flagprof.runner.vendor import code_files_identity, load_with_vendor_code
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'sub').mkdir()
            (root/'config.json').write_text(json.dumps({'auto_map': {'AutoModel': 'modeling_test.Model'}}))
            (root/'modeling_test.py').write_text('from .sub.leaf import VALUE\nclass Model:\n    value = VALUE\n')
            (root/'sub/__init__.py').write_text('')
            leaf = root/'sub/leaf.py'
            with patch.object(sys, 'dont_write_bytecode', True):
                leaf.write_text('VALUE = 1\n')
                before = code_files_identity(root)
                first = load_with_vendor_code(root, 'float32')
                leaf.write_text('VALUE = 222\n')
                after = code_files_identity(root)
                second = load_with_vendor_code(root, 'float32')
            self.assertIn('sub/leaf.py', before)
            self.assertNotEqual(before, after)
            self.assertEqual((first.value, second.value), (1, 222))
            self.assertNotEqual(type(first).__module__, type(second).__module__)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional torch CPU runtime')
class PackageExecutionTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.targets = patch.object(sys, 'path', [str(ROOT/'targets'), *sys.path])
        self.targets.start(); self.addCleanup(self.targets.stop)
        import make_ctr_package
        self.package = self.root/'ctr'
        with patch.object(sys, 'argv', ['make_ctr_package', '--samples', '4', '--out', str(self.package)]), contextlib.redirect_stdout(io.StringIO()):
            make_ctr_package.main()

    def ctr(self, name='run', **options):
        from flagprof.runner.ctr import run_ctr
        return run_ctr(package=self.package, arch='dlrm', seed=options.pop('seed', 531),
                       device='cpu', dtype='float32', outputs=self.root/name,
                       samples=['0'], batch_size=1, forwards=1, **options)

    def capture(self, label, call):
        from flagprof import attach
        from flagprof.store import create_run, finalize
        run = create_run(self.root/label)
        with patch('flagprof.devices.accelerator', return_value=True):
            with attach(run, 'ascend', preset='accuracy', config={'tensor_boundaries': 'model'}) as handle:
                result = call()
                self.assertEqual(handle.probe_manager.errors, [])
        finalize(run)
        return run, result

    def compare(self, a, b, label):
        from flagprof.report import generate_oracle
        out = self.root/label
        generate_oracle(a, b, out)
        return json.loads((out/'result.json').read_text())['results']

    def assert_paired_scores(self, result):
        self.assertEqual(result['oracle.identity_diff']['status'], 'ok')
        self.assertEqual(result['oracle.layer']['status'], 'ok')
        self.assertEqual(result['oracle.first_divergence']['status'], 'ok')
        self.assertTrue(any(r['boundary']=='scores' for r in result['oracle.layer']['data']))
        self.assertTrue(all(r['digest_equal'] for r in result['oracle.layer']['data']))

    def test_ctr_capture_store_oracle_and_weight_mismatch(self):
        a, ra = self.capture('store-a', lambda: self.ctr('a'))
        b, rb = self.capture('store-b', lambda: self.ctr('b'))
        self.assertEqual(ra['score_batches_sha256'], rb['score_batches_sha256'])
        self.assert_paired_scores(self.compare(a, b, 'paired'))
        c, _ = self.capture('store-c', lambda: self.ctr('c', seed=532))
        result = self.compare(a, c, 'weight-change')
        self.assertEqual(result['oracle.identity_diff']['notes'], ['identity_mismatch'])
        fields = {r['field']: r for r in result['oracle.identity_diff']['data']}
        self.assertFalse(fields['weight_sha256']['equal'])

    def test_rerank_capture_store_oracle_both_scoring_modes(self):
        torch = self.torch
        from flagprof.runner.rerank import run_rerank
        from flagprof.runner.rerank_common import sha
        model_dir = self.root/'model'; model_dir.mkdir()
        (model_dir/'weights.bin').write_bytes(b'fixture-weights')
        class TinyModel(torch.nn.Module):
            def __init__(self, causal):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 4)
                self.linear = torch.nn.Linear(4, 8 if causal else 1)
                self.causal = causal
                self.config = SimpleNamespace(use_cache=False)
            def forward(self, input_ids, attention_mask, return_dict=True, use_cache=False):
                hidden = self.embedding(input_ids)
                if not self.causal:
                    hidden = (hidden*attention_mask.unsqueeze(-1)).sum(1)/attention_mask.sum(1, keepdim=True)
                return SimpleNamespace(logits=self.linear(hidden))
        tokenizer = SimpleNamespace(unk_token_id=-1, convert_tokens_to_ids=lambda word: {'yes': 1, 'no': 2}[word])
        hf = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
             AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: TinyModel(True)),
             AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=lambda *a, **k: TinyModel(False)))
        for score in ('sequence_logit', 'causal_yesno'):
            with self.subTest(score=score), patch.dict(sys.modules, {'transformers': hf}):
                package = self.root/score; (package/'prepared').mkdir(parents=True)
                torch.save([dict(sample_id='a', input_ids=torch.tensor([1, 2, 3]))], package/'prepared/inputs.pt')
                (package/'prepared/samples.json').write_text('{"samples":["a"]}')
                (package/'manifest.json').write_text(json.dumps(dict(oracle_id=score, score_semantics=score,
                    template_sha='template', contract=dict(batch_size=1), files={name: dict(sha256=sha(package/name))
                    for name in ('prepared/inputs.pt','prepared/samples.json')})))
                def execute(label):
                    return run_rerank(package=package, model_path=str(model_dir), score=score,
                                      device='cpu', dtype='float32', outputs=self.root/label, forwards=1)
                a, _ = self.capture(score+'-a', lambda: execute(score+'-out-a'))
                b, _ = self.capture(score+'-b', lambda: execute(score+'-out-b'))
                self.assert_paired_scores(self.compare(a, b, score+'-report'))
                # Task semantics are mandatory even if every tensor is unchanged.
                state_path = next((b/'attachments').glob('*/state.json'))
                state = json.loads(state_path.read_text())
                state['oracle_identity']['score_semantics'] = 'other'
                state_path.write_text(json.dumps(state))
                self.assertEqual(self.compare(a, b, score+'-mismatch')['oracle.identity_diff']['notes'], ['identity_mismatch'])

    def test_component_activation_cleanup_and_numerics_are_effective(self):
        from flagprof.runner.__main__ import main
        output = self.root/'configured'
        component = SimpleNamespace(identity={'source_sha256': 'fixture'}, close=Mock())
        include = self.root/'include.json'; include.write_text('["add"]')
        argv = ['runner','--task','ctr','--package',str(self.package),'--arch','dlrm',
                '--device','cpu','--dtype','float32','--forwards','1','--repeats','2',
                '--outputs',str(output),'--flaggems','on','--flagtree','on',
                '--flaggems-include',str(include)]
        with patch.object(sys,'argv',argv), patch('flagprof.components.compiler.select') as compiler, \
             patch('flagprof.components.flaggems.FlagGems.enable', return_value=component) as enable, \
             contextlib.redirect_stdout(io.StringIO()):
            main()
        self.assertTrue(any(c.args[0]['flagtree']=='on' for c in compiler.call_args_list))
        enable.assert_called_once(); component.close.assert_called_once()
        result = json.loads((output/'runner.json').read_text())
        self.assertEqual(result['oracle_identity']['flaggems'], 'on')
        self.assertEqual(result['component_identity'], component.identity)
        self.assertEqual(len(result['runs']), 2)
        # Reference numerics on CPU must fail, not silently execute defaults.
        with self.assertRaisesRegex(ValueError, 'reference numerics require CUDA'):
            self.ctr('bad-numerics', numerics='reference')
        component.close.reset_mock()
        with patch('flagprof.components.flaggems.FlagGems.enable', return_value=component), \
             patch('flagprof.runner.ctr._run_ctr', side_effect=RuntimeError('fixture failure')):
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                self.ctr('failed', flaggems='on', flaggems_include=str(include))
        component.close.assert_called_once()

    def test_attachment_component_conflict_fails_before_forward(self):
        from flagprof import attach
        from flagprof.store import create_run
        with attach(create_run(self.root/'conflict-store'), 'ascend', preset='accuracy'):
            with self.assertRaisesRegex(ValueError, 'component configuration conflicts'):
                self.ctr(flaggems='on')
            with self.assertRaisesRegex(ValueError, 'compiler configuration conflicts'):
                self.ctr(flagtree='on')

    def test_package_cli_rejects_unused_options_and_invalid_numerics(self):
        from flagprof.runner.__main__ import main
        for task in ('ctr','rerank'):
            argv = ['runner','--task',task,'--package',str(self.package),
                    '--device','cpu','--outputs',str(self.root/'unused')]
            argv += ['--arch','dlrm'] if task=='ctr' else ['hf:missing','--score','sequence_logit']
            with self.subTest(task=task), patch.object(sys,'argv',argv+['--pooling','cls']), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()
            with self.subTest(task=task), patch.object(sys,'argv',argv+['--numerics','reference']), \
                 self.assertRaisesRegex(ValueError, 'reference numerics require CUDA'):
                main()

    def test_cls_cli_defaults_to_right_padding(self):
        from flagprof.runner.__main__ import main
        tokenizer = Mock(return_value=object())
        hf = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer),
             AutoModel=SimpleNamespace(from_pretrained=Mock(side_effect=ValueError('stop before weights'))))
        argv = ['runner','hf:local','--inputs','texts.jsonl','--pooling','cls',
                '--device','cpu','--outputs',str(self.root/'cls')]
        with patch.dict(sys.modules, {'transformers':hf}), patch.object(sys,'argv',argv), \
             self.assertRaisesRegex(ValueError, 'stop before weights'):
            main()
        self.assertEqual(tokenizer.call_args.kwargs['padding_side'], 'right')

    def test_embed_target_requires_explicit_vendor_opt_in(self):
        import target_embedder
        root = self.root/'custom'; root.mkdir()
        (root/'config.json').write_text(json.dumps({'auto_map': {'AutoModel': 'modeling_custom.Model'}}))
        (root/'modeling_custom.py').write_text("raise RuntimeError('CUSTOM_CODE_EXECUTED')\n")
        texts = self.root/'texts.jsonl'; texts.write_text('"hello"\n')
        standard = Mock(side_effect=ValueError('custom model requires trust'))
        hf = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: object()),
                             AutoModel=SimpleNamespace(from_pretrained=standard))
        argv = ['target_embedder','--model','hf:'+str(root),'--inputs',str(texts),
                '--batch-size','1','--device','cpu','--outputs',str(self.root/'out')]
        with patch.dict(sys.modules, {'transformers': hf}):
            with patch.object(sys,'argv',argv), self.assertRaisesRegex(ValueError, 'custom model requires trust'):
                target_embedder.main()
            self.assertFalse(standard.call_args.kwargs['trust_remote_code'])
            with patch.object(sys,'argv',argv+['--trust-remote-code']), self.assertRaisesRegex(RuntimeError, 'CUSTOM_CODE_EXECUTED'):
                target_embedder.main()

    def test_embedding_loss_uses_fp32_under_bf16_autocast(self):
        torch = self.torch
        from flagprof.training.embedding import build_loss_fn
        from torch.utils._python_dispatch import TorchDispatchMode
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.projection = torch.nn.Linear(4, 4); self.outputs = []
            def forward(self, input_ids, attention_mask):
                hidden = self.projection(torch.nn.functional.one_hot(input_ids, 4).float())
                self.outputs.append(hidden)
                return SimpleNamespace(last_hidden_state=hidden)
        matrix_dtypes = []
        class Observe(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if str(func)=='aten.mm.default': matrix_dtypes.append([t.dtype for t in args])
                return func(*args, **(kwargs or {}))
        batch = dict(anchor_ids=torch.tensor([[1,2],[2,3]]), anchor_mask=torch.ones(2,2,dtype=torch.long),
                     cand_ids=torch.tensor([[1,2],[0,1],[2,3],[1,3]]), cand_mask=torch.ones(4,2,dtype=torch.long))
        model = Encoder()
        with Observe(), torch.autocast('cpu',dtype=torch.bfloat16):
            loss = build_loss_fn('mean', .05)(model, batch)
        self.assertTrue(all(v.dtype==torch.bfloat16 for v in model.outputs))
        self.assertEqual(matrix_dtypes[-1], [torch.float32, torch.float32])
        a, b = [torch.nn.functional.normalize(v.float().mean(1), dim=-1) for v in model.outputs]
        expected = torch.nn.functional.cross_entropy(a@b.t()/.05, torch.tensor([0,2]))
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))


if __name__=='__main__':
    unittest.main()
