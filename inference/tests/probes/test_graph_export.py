"""Real exporters and numerical consumers, run in the registered CPU image."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    raise unittest.SkipTest('optional torch CPU graph runtime')
from flagprof.graph import ExportSpec, export_graph, verify_graph
from flagprof.graph.core import compare_outputs, validate_onnx
from flagprof.runner.embedding import embedding_outputs

HAS_ONNX = all(importlib.util.find_spec(name) is not None for name in ('onnx', 'onnxscript'))
HAS_ORT = importlib.util.find_spec('onnxruntime') is not None


class TinyEmbedding(torch.nn.Module):
    def __init__(self, mean=False):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.mean = mean
    def forward(self, x, mask):
        return embedding_outputs(self.linear(x), mask, mean=self.mean)


class GraphExportTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(531); torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)/'export'
        self.model = TinyEmbedding().eval()
        self.args = (torch.randn(2,4,3), torch.tensor([[0,0,1,1],[1,1,1,0]]))
        self.spec = ExportSpec(self.model, self.args, ('x','mask'), ('pooled','output'))

    @unittest.skipUnless(HAS_ONNX and HAS_ORT, 'optional graph dependencies; run ci/graph.sh for the full graph lane')
    def test_full_fx_onnx_and_changed_inputs(self):
        result = export_graph(self.spec, self.out, bundle='full', verify_onnx=True)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(verify_graph(self.out)['status'], 'verified')
        loaded = torch.export.load(self.out/'model.pt2').module()
        args = (self.args[0]+2, self.args[1].flip(1))
        with torch.inference_mode():
            expected = self.model(*args)
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(expected,loaded(*args))))
            checked = validate_onnx(self.out/'model.onnx', args, {}, expected, self.spec.input_names, self.spec.output_names)
            self.assertEqual(checked['status'], 'passed')
            with self.assertRaises((RuntimeError,ValueError,AssertionError)):
                loaded(args[0][:1], args[1][:1])
        graph = json.loads((self.out/'fx.json').read_text())
        ids = {n['id'] for n in graph['nodes']}
        self.assertTrue(any(n['modules'] for n in graph['nodes']))
        self.assertTrue(any(n['inputs'] for n in graph['nodes']))
        self.assertTrue(all(set(n['inputs']).issubset(ids) for n in graph['nodes']))
        self.assertNotIn('tensor(', (self.out/'fx.json').read_text())
        onnx_graph=json.loads((self.out/'onnx.json').read_text())
        self.assertTrue(any(n['modules'] for n in onnx_graph['nodes']))
        with self.assertRaisesRegex(ValueError,'names or order'):
            validate_onnx(self.out/'model.onnx',args,{},expected,self.spec.input_names,self.spec.output_names[::-1])
        with (self.out/'model.pt2').open('ab') as stream:stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError,'checksum'):verify_graph(self.out)

    def test_structure_and_mean_semantics(self):
        self.model.mean = True
        with torch.inference_mode():
            hidden = self.model.linear(self.args[0]); weights = self.args[1].unsqueeze(-1).float()
            pooled = (hidden.float()*weights).sum(1)/weights.sum(1)
            self.assertTrue(torch.equal(pooled, self.model(*self.args)[0]))
        result = export_graph(self.spec, self.out, formats=('fx',))
        self.assertEqual(result['status'], 'completed', result)
        self.assertFalse((self.out/'model.pt2').exists())
        self.assertFalse((self.out/'.work').exists())
        self.assertFalse((self.out/'example.pt').exists())
        verify_graph(self.out)

    def test_missing_onnx_keeps_fx(self):
        with patch.dict('sys.modules', {'onnx':None}):
            result = export_graph(self.spec, self.out, bundle='full')
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['formats']['fx']['status'], 'completed')
        self.assertEqual(result['formats']['onnx']['status'], 'failed')
        self.assertTrue((self.out/'model.pt2').exists())

    @unittest.skipUnless(HAS_ONNX, 'ONNX exporter required for missing-runtime test')
    def test_missing_runtime_is_failure(self):
        with patch.dict('sys.modules', {'onnxruntime':None}):
            result = export_graph(self.spec, self.out, verify_onnx=True)
        self.assertEqual(result['status'], 'partial')
        self.assertIn('onnxruntime', result['formats']['onnx']['error'])

    def test_comparison_count_order_dtype_and_error(self):
        with torch.inference_mode():expected=self.model(*self.args)
        self.assertEqual(compare_outputs(expected, expected[::-1], self.spec.output_names, exact=True)['status'], 'failed')
        self.assertEqual(compare_outputs(expected, [v+1 for v in expected], self.spec.output_names, exact=False)['status'], 'failed')
        for actual in (expected[:1], [v.double() for v in expected], [torch.full_like(v,float('nan')) for v in expected]):
            with self.assertRaises(ValueError):compare_outputs(expected,actual,self.spec.output_names,exact=True)

    def test_capture_failure_retains_diagnostics(self):
        with patch('torch.export.export', side_effect=RuntimeError('capture failure')):
            result=export_graph(self.spec,self.out)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('capture failure',result['error'])
        self.assertTrue((self.out/'manifest.json').exists())

    def test_prepare_fixed_package(self):
        from types import SimpleNamespace
        from flagprof.graph.command import prepare
        class Qwen(torch.nn.Module):
            config=SimpleNamespace(model_type='qwen3',use_cache=True)
            def forward(inner,input_ids,attention_mask,**kw):
                return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1).expand(-1,-1,3))
        path=Path(self.tmp.name)/'model';path.mkdir();(path/'config.json').write_text('{}')
        inputs=dict(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.tensor([[0,1,1]]))
        config=dict(model='hf:'+str(path),dtype='float32',inputs_from='fixed',batch_size=None,
                    batch_index=0,inputs=None,pooling='last',padding_side='left')
        with patch('flagprof.runner.embedding.load_model',return_value=Qwen().eval()), patch('flagprof.runner.fixed.load',return_value=([(['s'],inputs)],{'source':'fixed'})):
            spec=prepare(config)
        self.assertEqual(spec.identity['samples'],['s'])
        self.assertEqual(spec.identity['padding_side'],'prepared')
        self.assertTrue(torch.equal(spec.model(*spec.args)[0],torch.full((1,3),3.)))

    def test_collision_and_active_attachment(self):
        self.out.mkdir()
        with self.assertRaises(FileExistsError):export_graph(self.spec,self.out)
        with patch('flagprof.current_attachment', return_value=object()), self.assertRaisesRegex(ValueError,'attachment'):
            export_graph(self.spec,self.out/'nested')

    def test_fx_verification_rejects_wrong_loaded_output(self):
        class BadLoaded:
            def module(inner):return lambda *a:(torch.zeros(2,3),torch.zeros(2,3))
        with patch('torch.export.load',return_value=BadLoaded()):
            result=export_graph(self.spec,self.out,formats=('fx',))
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['formats']['fx']['roundtrip']['status'],'failed')


if __name__ == '__main__':unittest.main()
