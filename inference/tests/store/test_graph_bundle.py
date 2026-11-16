"""Torch-free public graph bundle consumer and CLI contract."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from flagprof.graph import render_graph, verify_graph
from flagprof.graph.command import add_parser, configuration
from flagprof.graph.data import SCHEMA, file_info, write_json
from flagprof.graph.view import write_view


class GraphBundleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'bundle'
        self.root.mkdir()
        self.manifest = dict(schema=SCHEMA, status='completed', bundle='structure',
                             requested_formats=['fx'],formats={'fx':{'status':'completed'}})
        self.graph = dict(nodes=[dict(id='x', target='x', inputs=[], users=['add'], modules=[]),
                                  dict(id='add', target='aten.add', inputs=['x'], users=[], modules=[])])
        write_json(self.root/'fx.json', self.graph)
        (self.root/'model.fx.txt').write_text('x -> add')
        write_view(self.root/'index.html', self.manifest, {'fx':self.graph})
        self.manifest['files'] = {p.name:file_info(p) for p in self.root.iterdir()}
        self.save()

    def save(self):
        write_json(self.root/'manifest.json', self.manifest)

    def test_read_and_render_without_torch(self):
        self.assertEqual(verify_graph(self.root)['files'], 3)
        before = file_info(self.root/'manifest.json')
        out = Path(self.tmp.name)/'view.html'
        render_graph(self.root, out)
        self.assertIn('aten.add', out.read_text())
        self.assertEqual(before, file_info(self.root/'manifest.json'))
        script = 'import sys; import flagprof.graph, flagprof.graph.view, flagprof.cli; assert "torch" not in sys.modules; assert "onnx" not in sys.modules'
        subprocess.run([sys.executable, '-B', '-S', '-c', script], check=True)

    def test_sealed_render_rejected(self):
        with self.assertRaisesRegex(ValueError,'outside'):
            render_graph(self.root, self.root/'other.html')

    def test_missing_corrupt_and_manifest_omission(self):
        (self.root/'fx.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'checksum'):
            verify_graph(self.root)
        self.manifest['files'].pop('fx.json')
        self.save()
        with self.assertRaisesRegex(ValueError,'missing required'):
            verify_graph(self.root)

    def test_full_requires_payload_and_external(self):
        self.manifest.update(bundle='full', requested_formats=['onnx'],formats={'onnx':dict(status='completed',external_data=['weights.data'])})
        self.save()
        with self.assertRaisesRegex(ValueError,'missing required'):
            verify_graph(self.root)

    def test_failed_not_verified(self):
        self.manifest['status'] = 'partial'; self.save()
        with self.assertRaisesRegex(ValueError,'not completed'):
            verify_graph(self.root)

    def test_requested_format_cannot_disappear(self):
        self.manifest['requested_formats'].append('onnx'); self.save()
        with self.assertRaisesRegex(ValueError,'formats are incomplete'):
            verify_graph(self.root)

    def test_html_is_data(self):
        self.graph['nodes'][0]['target'] = '</script><script>alert("DATA")</script>&'
        self.manifest['identity'] = {'model':'DATA <script>INFO</script>'}
        write_view(self.root/'index.html', self.manifest, {'fx':self.graph})
        text = (self.root/'index.html').read_text()
        self.assertNotIn('<script>alert', text)
        self.assertIn('DATA &lt;script&gt;INFO&lt;/script&gt;', text)
        embedded = text.split('id="graph-data">')[1].split('</script>')[0]
        self.assertEqual(json.loads(embedded)['fx'], self.graph)

    def test_cli_contract(self):
        p = argparse.ArgumentParser(); add_parser(p.add_subparsers(dest='verb'))
        good = ['export','hf:/model','--inputs','in.jsonl','--out','new']
        self.assertEqual(configuration(p.parse_args(good))['formats'], ['fx','onnx'])
        for extra in (['--batch-size','0'], ['--formats','fx,fx'], ['--verify-onnx'], ['--batch-index','1']):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                configuration(p.parse_args(good+extra))
        args = p.parse_args(good+['--dtype','float32','--verify-onnx'])
        self.assertTrue(configuration(args)['verify_onnx'])


if __name__ == '__main__': unittest.main()
