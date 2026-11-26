# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Replay planning and receipt failures that must stop the Oracle step."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flagprof.cli import main
from flagprof.replay import digest
from flagprof.store import create_run, finalize


class ReplayWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.job = self.root/'job'
        self.key = "block #1 'quoted'/$VARIABLE"
        replay = {'call_key': self.key}
        for role in ('ref', 'cmp'):
            run = create_run(self.root/role)
            state = run/'attachments/test/state.json'
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps(dict(id='test', oracle_identity={'device':'cpu'})))
            finalize(run)
            replay[role] = {'run': str(run.relative_to(self.root))}
        self.config = dict(schema_version=1, mode='replay', replay=replay)
        self.source = self.root/'config.json'

    def invoke(self, *flags, out=None):
        self.source.write_text(json.dumps(self.config))
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = main(['run', str(self.source), '--out', str(out or self.job), '--json', *flags])
        return code, json.loads(stream.getvalue())

    def test_plan_uses_exact_key_and_guards_inputs(self):
        replay = self.config['replay']
        replay['perf_cmp_run'] = replay['cmp']['run']
        code, plan = self.invoke('--dry-run')
        self.assertEqual(code, 0, plan)
        prepare, ref, cmp, report = [s['argv'] for s in plan['steps']]
        self.assertEqual(prepare[prepare.index('--call-key')+1], self.key)
        self.assertEqual(ref[ref.index('--device')+1], 'cpu')
        self.assertEqual(cmp[cmp.index('--side')+1], 'cmp')
        self.assertEqual(report[report.index('--perf-cmp-run')+1], str(self.root/replay['cmp']['run']))
        self.assertFalse(self.job.exists())
        self.assertEqual(self.invoke('--dry-run', out=self.root/replay['ref']['run']/'nested')[0], 2)
        replay['cmp']['device'] = 'cuda:1'
        self.assertIn('replay.cmp.device', self.invoke('--dry-run')[1]['error'])

    def test_group_rejected_without_creating_job(self):
        path = self.root/'group'/'group.json'
        path.parent.mkdir()
        path.write_text('{}')
        self.config['replay']['ref']['run'] = str(path.relative_to(self.root))
        code, summary = self.invoke('--check')
        self.assertEqual(code, 2)
        self.assertIn('单个 accuracy Store', summary['error'])
        self.assertFalse(self.job.exists())

    def test_failed_self_reproduction_and_foreign_store_stop_report(self):
        def producer(argv, *, cwd, env, stdout, stderr):
            from subprocess import CompletedProcess
            package = self.job/'package'
            manifest = package/'replay.json'
            if 'prepare' in argv:
                package.mkdir()
                manifest.write_text('{}')
                receipt = dict(package=str(package), manifest=digest(manifest), call_key=self.key)
                code = 0
            else:
                store = self.root/'foreign' if foreign else Path(argv[argv.index('--store')+1])
                run = create_run(store)
                finalize(run)
                state = dict(schema='operator-replay-result/1', status='self_reproduction_failed', side='ref',
                             device='cpu', package=digest(manifest), call_key=self.key,
                             cases={'ref': {'self_reproduction': {'digest_equal': False}}, 'cmp': {}})
                (run/'replay-result.json').write_text(json.dumps(state))
                receipt = dict(run=str(run), status=state['status'], cases=state['cases'])
                code = 0 if foreign else 2
            stdout.write(json.dumps(receipt).encode())
            return CompletedProcess(argv, code)

        with patch('flagprof.workflow.executor.subprocess.run', producer):
            for foreign in (False, True):
                with self.subTest(foreign=foreign):
                    self.job = self.root/('foreign-receipt' if foreign else 'self-check')
                    code, summary = self.invoke()
                    self.assertEqual(code, 2)
                    self.assertIn('意外的 Store' if foreign else '自复现失败', summary['error'])
                    self.assertEqual([s['id'] for s in summary['steps']], ['prepare', 'run.ref'])
                    self.assertIn('oracle', summary['skipped'])
                    receipts = [a for a in summary['artifacts'] if a['type'] == 'replay-result']
                    self.assertEqual(len(receipts), 0 if foreign else 1)
