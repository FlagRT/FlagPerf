# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Schema v1 public workflow checks for CPU graph export and batch receipts."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from flagprof.cli import main
from flagprof.workflow import TEMPLATES
from flagprof.workflow.executor import Execution, reserve
from flagprof.workflow.suite_receipts import batch_receipt

ROOT = Path(__file__).resolve().parents[1]
HAS_YAML = importlib.util.find_spec('yaml') is not None


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.recipes = self.root/'recipes'
        self.recipes.mkdir()
        model = self.recipes/'Qwen3-Embedding-test'
        model.mkdir()
        (model/'config.json').write_text('{"model_type":"qwen3"}')
        self.config = dict(schema_version=1, mode='export', target=dict(model=model.name, texts=['sample']),
                           runner=dict(batch_size=1))

    def cli(self, config, *options, code=0):
        source = self.recipes/'workflow.json'
        source.write_text(json.dumps(config))
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = main(['run',str(source),*map(str,options),'--out',str(self.root/'job'),'--json'])
        reply = json.loads(stdout.getvalue())
        self.assertEqual(status, code, reply.get('error'))
        return reply

    def test_custom_command_drill_uses_runtime_sample_binding(self):
        config = dict(schema_version=1, profile={'preset': 'accuracy'}, hardware={'vendor': 'nvidia'},
                      collection={'tensor_ops_in': '<root>.encoder'},
                      target={'type': 'command', 'command': [sys.executable, '-c', 'pass']})
        plan = self.cli(config, '--dry-run')
        argv = plan['steps'][0]['argv']
        self.assertEqual(argv[argv.index('--tensor-ops-in') + 1], '<root>.encoder')
        self.assertNotIn('runner', plan['config'])
        # Check command planning independently of the optional torch dependency.
        with patch('flagprof.workflow.dependencies.check_dependencies', return_value=[]):
            checked = self.cli(config, '--check')
        self.assertIn('bind_tensor_inputs([sample_id], inputs)', checked['runtime_requirement'])
        self.assertIn('register_model', checked['runtime_requirement'])
        self.assertIn('不接受 runner', self.cli(config, '--set', 'runner.batch_size=1', '--check', code=2)['error'])
        config['target'] = self.config['target']
        for batch in (None, 1, 2):
            options = [] if batch is None else ['--set', f'runner.batch_size={batch}']
            with self.subTest(batch=batch):
                plan = self.cli(config, *options, '--dry-run', code=0 if batch == 1 else 2)
                if batch == 1:
                    self.assertNotIn('runtime_requirement', plan)
                else:
                    self.assertIn('runner.batch_size=1', plan['error'])

    def test_templates_match_checkout(self):
        self.assertEqual(set(TEMPLATES), {'quickstart','custom-script','cpu-demo','compare','tp2','export','batch','replay'})
        for filename in TEMPLATES.values():
            self.assertEqual((ROOT/'configs'/filename).read_bytes(),
                             (ROOT/'flagprof/workflow/templates'/filename).read_bytes())

    def test_local_cross_precision_requires_explicit_accuracy_declaration(self):
        config=dict(self.config,mode='compare',profile={'preset':'accuracy'},
                    hardware={'vendor':'nvidia'},compare={
                        'ref':{'runner':{'dtype':'float32'}},'cmp':{'runner':{'dtype':'bfloat16'}}})
        self.assertIn('runner.dtype',self.cli(config,'--dry-run',code=2)['error'])
        config['compare']['declare']=['dtype']
        plan=self.cli(config,'--dry-run')
        self.assertEqual(plan['declaration']['differences'],['dtype'])
        config['compare']['cmp']['runner']['warmup']=1
        self.assertIn('runner.warmup',self.cli(config,'--dry-run',code=2)['error'])
        config['profile']['preset']='perf'
        config['compare']['cmp']['runner']={'dtype':'float32'}
        self.assertIn('dtype',self.cli(config,'--dry-run',code=2)['error'])

    def test_export_dry_run_and_invalid_combinations(self):
        out = self.cli(self.config,'--set','export.formats=onnx,fx','--set','runner.dtype=float32',
                       '--set','export.verify_onnx=true','--dry-run')
        argv = out['steps'][0]['argv']
        self.assertEqual(argv[argv.index('--formats')+1], 'onnx,fx')
        self.assertIn('--verify-onnx',argv)
        self.assertEqual(out['config']['target']['model'],'hf:'+str(self.recipes/'Qwen3-Embedding-test'))
        self.assertEqual(out['component_owner'],'CPU export worker')
        for options, message in [
            (['export.verify_onnx=true'],'float32'), (['runner.dtype=float32','export.formats=fx','export.verify_onnx=true'],'onnx'),
            (['export.formats=["fx","fx"]'],'formats'), (['export.batch_index=1'],'batch-index'),
            (['export.timeout=0'],'positive'), (['runner.batch_size=2'],'至少'), (['runner.dtype=float16'],'dtype'),
            (['hardware.device=cpu'],'hardware'), (['output.verify=true'],'verify'),
            (['target.inputs=other'],'inputs'), (['runner.pooling=cls'],'pooling')]:
            args=[item for option in options for item in ('--set',option)]
            with self.subTest(options=options):
                self.assertIn(message,self.cli(self.config,*args,'--check',code=2)['error'])
        self.assertFalse((self.root/'job').exists())

    def test_fixed_package_batch_index_after_rebatch(self):
        root=self.recipes/'fixed'
        (root/'prepared').mkdir(parents=True)
        (root/'manifest.json').write_text('{}')
        for name in ('inputs.pt','samples.json'):
            (root/'prepared'/name).touch()
        (root/'prepared/forward-inputs.json').write_text(json.dumps({'batches':[{'sample_ids':['x','y','z']}]}))
        del self.config['target']['texts']
        self.config['target']['inputs_from']='fixed'
        ok=self.cli(self.config,'--set','runner.batch_size=2','--set','export.batch_index=1','--dry-run')
        self.assertIn('--inputs-from',ok['steps'][0]['argv'])
        self.assertIn('batch_index',self.cli(self.config,'--set','runner.batch_size=2','--set','export.batch_index=2','--check',code=2)['error'])
        self.assertFalse((self.root/'job').exists())

    def test_failed_graph_keeps_partial_format_and_code(self):
        job=reserve(self.root/'partial',False)
        bundle=job/'export-bundle';bundle.mkdir()
        from flagprof.graph.data import SCHEMA
        for name in ('index.html','fx.json'):
            (bundle/name).write_text('x')
        manifest=dict(schema=SCHEMA,status='partial',bundle='structure',formats={
            'fx':dict(status='completed',roundtrip={'status':'passed'}),
            'onnx':dict(status='failed',error='native diagnostic')},files={'index.html':{},'fx.json':{}})
        (bundle/'manifest.json').write_text(json.dumps(manifest))
        step=dict(id='export',variant='export',argv=[sys.executable,'-c','raise SystemExit(37)'],cwd=str(job),depends_on=[])
        plan=dict(output_type='graph-bundle',component_owner='CPU export worker',vendor=None,directory=str(job),
                  steps=[step],runners=[],skipped={})
        config=dict(self.config,export=dict(formats=['fx','onnx'],verify_onnx=False,bundle='structure'))
        result=Execution(config,self.recipes/'workflow.json',{},job,plan,None).run({})
        self.assertEqual(result['exit_code'],37)
        self.assertEqual(result['steps'][0]['exit_code'],37)
        self.assertIn(('export.onnx','failed'),[(r['subject'],r['status']) for r in result['evidence']])
        self.assertIn('native diagnostic',str(result['evidence']))
        self.assertTrue(any(a['type']=='graph-viewer' for a in result['artifacts']))

    @unittest.skipUnless(HAS_YAML,'batch driver needs PyYAML')
    def test_batch_path_origins_and_builtin_snapshot(self):
        from flagprof.workflow.loader import load, snapshot
        from flagprof.workflow.suites import prepare
        recipe=self.recipes/'workflow.json'
        def normalized(config, *overrides):
            recipe.write_text(json.dumps(config))
            data,origins=load(recipe,overrides,None)
            plan,assets=prepare(data,origins,recipe,self.root/'job')
            return data,plan,next(iter(assets.values()))['models']
        cfg=dict(schema_version=1,mode='batch',batch={'config':'builtin:cpu','filter':'dlrm'})
        data,plan,rows=normalized(cfg)
        self.assertEqual(plan['model_ids'],['ctr-dlrm'])
        self.assertEqual(plan['roster_source'],'builtin:cpu')
        self.assertEqual(rows[0]['package'],str(self.recipes/'packages/ctr-v1'))
        self.assertNotIn('--device',plan['steps'][0]['argv'])
        self.assertNotIn('config',snapshot(data)['batch'])
        # Put the resolved snapshot in a different directory, like commands.sh does.
        snap=self.root/'snapshot.json';snap.write_text(json.dumps(snapshot(data)))
        reloaded,ori=load(snap,[],None)
        _,second=prepare(reloaded,ori,snap,self.root/'job2')
        self.assertEqual(next(iter(second.values()))['models'],rows)
        external=self.root/'lists';external.mkdir()
        row=dict(id='attach',task='attach',target='./echo.py',args={'package':'./package$$1','model':'./local'})
        (external/'external.yaml').write_text(json.dumps({'models':[row]}))
        cfg['batch']={'config':'../lists/external.yaml'}
        _,plan,rows=normalized(cfg)
        self.assertEqual(plan['input_type'],'external')
        self.assertEqual(rows[0]['args']['package'],str(external/'package$1'))
        self.assertEqual(rows[0]['args']['model'],str(external/'local'))
        _,plan,rows=normalized(cfg,'batch='+json.dumps({'models':[row]}),'hardware.device=cpu')
        self.assertIn('--device',plan['steps'][0]['argv'])
        self.assertEqual(rows[0]['target'],str(Path.cwd()/'echo.py'))

    @unittest.skipUnless(HAS_YAML,'batch driver needs PyYAML')
    def test_batch_preflight_rejects_ambiguous_inputs(self):
        row={'id':'good','task':'ctr','arch':'dlrm'}
        for batch in ({}, {'config':'builtin:cpu','models':[row]}, {'models':[row,row]}, {'models':[{}]},
                      {'config':'builtin:missing'},{'models':[dict(row,id='../outside')]}):
            with self.subTest(batch=batch):
                self.cli(dict(schema_version=1,mode='batch',batch=batch),'--check',code=2)
        self.assertFalse((self.root/'job').exists())

    def test_interrupt_keeps_completed_model_rows(self):
        out=self.root/'batch-results';out.mkdir()
        row=dict(id='done',task='ctr',status='blocked',runs=[])
        (out/'battery-state.json').write_text(json.dumps({'models':[row]}))
        summary={'artifacts':[],'evidence':[]}
        batch_receipt(summary,self.root,{'model_ids':['done','pending']},143)
        self.assertEqual(summary['evidence'][0]['status'],'partial')
        self.assertEqual(summary['evidence'][0]['blocked'],1)
        self.assertEqual(summary['evidence'][1]['model_id'],'done')
        self.assertNotIn('batch-summary',[a['type'] for a in summary['artifacts']])

    def test_batch_pass_requires_two_actual_receipts(self):
        root=self.root/'batch-results';root.mkdir()
        row=dict(id='a',task='ctr',status='passed',runs=[])
        (root/'battery-summary.json').write_text(json.dumps(dict(total=1,passed=1,failed=0,blocked=0,models=[row])))
        (root/'battery-summary.md').touch()
        with self.assertRaisesRegex(ValueError,'A/A'):
            batch_receipt({'artifacts':[],'evidence':[]},self.root,{'model_ids':['a']},0)

    def test_json_batch_without_pyyaml_explains_dependency(self):
        cfg=dict(schema_version=1,mode='batch',batch={'config':'builtin:cpu'})
        source=self.recipes/'workflow.json';source.write_text(json.dumps(cfg))
        result=subprocess.run([sys.executable,'-B','-S',str(ROOT/'run.py'),'run',str(source),'--check','--json'],
            cwd=self.root,env={**os.environ,'PYTHONPATH':str(ROOT)},capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn('PyYAML',json.loads(result.stdout)['error'])


if __name__=='__main__':
    unittest.main()
