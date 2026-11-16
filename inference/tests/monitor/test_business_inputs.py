import json
import os
from pathlib import Path
import tempfile
import unittest
from flagprof.store import Writer,WriterConfig,StoreView,create_run,finalize
from flagprof.monitor.session import configuration,initialize
from flagprof.monitor.retention import RollingWriter
from flagprof.monitor.diagnostic_inputs import Inputs
from flagprof.telemetry.contract import raw_clock


class InputsTests(unittest.TestCase):
    def test_partial_destination_failure_restarts_from_committed_prefix(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);node=base/'node';run=create_run(base/'target')
            initialize(node,'nvidia','0',dict(configuration(),min_free_bytes=0))
            folder=node/'targets/history';folder.mkdir(parents=True)
            (folder/'target.json').write_text(json.dumps(dict(diagnostic_identity=dict(run_id=run.name),
                references=[os.path.relpath(run/'monitor-ref.json',node)])))
            with Writer(run) as target:
                for i in range(150):target.emit(dict(t='health',kind='heartbeat',sev='info',ts=raw_clock(),
                    detail=dict(scope='business',version=1,action='test',ordinal=i)))
            finalize(run)
            writer=RollingWriter(node,dict(entities=[],descriptors=[]));inputs=Inputs(node,writer)
            emit=writer.emit_many
            def fail(events):
                emit(events[:64]);writer.writer.flush()
                raise OSError('injected destination write failure')
            with patch.object(writer,'emit_many',fail):
                with self.assertRaisesRegex(OSError,'destination'):inputs.poll()
            inputs.close();writer.close()
            writer=RollingWriter(node,dict(entities=[],descriptors=[]));inputs=Inputs(node,writer)
            for _ in range(10):
                if inputs.poll():break
            inputs.close();writer.close()
            with StoreView(node) as view:
                rows=[r['detail'] for r in view.iter_monitor_events(('health',)) if r.get('detail',{}).get('scope')=='business_input']
            self.assertEqual([r['record']['detail']['ordinal'] for r in rows],list(range(150)))
            self.assertEqual(len({tuple(r['source_identity']) for r in rows}),150)

    def test_committed_prefix_survives_restart_rotation_and_compression(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);node=base/'node';run=create_run(base/'target')
            initialize(node,'nvidia','0',dict(configuration(),min_free_bytes=0))
            folder=node/'targets/history';folder.mkdir(parents=True)
            (folder/'target.json').write_text(json.dumps(dict(diagnostic_identity=dict(run_id=run.name),
                references=[os.path.relpath(run/'monitor-ref.json',node)])))
            def event(index):return dict(t='health',kind='heartbeat',sev='info',ts=raw_clock(),
                detail=dict(scope='business',version=1,action='test',ordinal=index))
            target=Writer(run,config=WriterConfig(shard_records=30,compress=True))
            writer=RollingWriter(node,dict(entities=[],descriptors=[]));inputs=Inputs(node,writer)
            for i in range(75):target.emit(event(i))
            target.flush()
            for _ in range(10):
                if inputs.poll():break
            writer.writer.flush();inputs.close();writer.close()
            for i in range(75,150):target.emit(event(i))
            target.close();finalize(run)
            writer=RollingWriter(node,dict(entities=[],descriptors=[]));inputs=Inputs(node,writer)
            for _ in range(10):
                if inputs.poll():break
            inputs.poll();inputs.close();writer.close()
            with StoreView(node) as view:
                rows=[r['detail'] for r in view.iter_monitor_events(('health',)) if r.get('detail',{}).get('scope')=='business_input']
            self.assertEqual([r['record']['detail']['ordinal'] for r in rows],list(range(150)))
            self.assertEqual(len({tuple(r['source_identity']) for r in rows}),150)

    def test_unreadable_registered_manifest_is_a_source_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);node=base/'node';run=create_run(base/'target')
            initialize(node,'nvidia','0',dict(configuration(),min_free_bytes=0))
            folder=node/'targets/history';folder.mkdir(parents=True)
            (folder/'target.json').write_text(json.dumps(dict(diagnostic_identity=dict(run_id=run.name),references=[os.path.relpath(run/'monitor-ref.json',node)])))
            (run/'manifest.json').write_text('{partial')
            writer=RollingWriter(node,dict(entities=[],descriptors=[]));inputs=Inputs(node,writer)
            self.assertFalse(inputs.poll());inputs.close();writer.close()
            with StoreView(node) as view:
                ticks=[r for r in view.iter_monitor_events(('health',)) if r.get('detail',{}).get('scope')=='business_tick']
            self.assertEqual(ticks[-1]['detail']['sources'][run.name],'unavailable')

    def test_partial_line_is_backlog_then_recovers_without_duplicate_input(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);node=base/'node';run=create_run(base/'target')
            initialize(node,'nvidia','0',dict(configuration(),min_free_bytes=0))
            folder=node/'targets/history';folder.mkdir(parents=True)
            (folder/'target.json').write_text(json.dumps(dict(diagnostic_identity=dict(run_id=run.name),references=[os.path.relpath(run/'monitor-ref.json',node)])))
            target=Writer(run,config=WriterConfig(compress=False))
            target.emit(dict(t='health',kind='heartbeat',sev='info',ts=raw_clock(),detail=dict(scope='business',action='first')));target.flush()
            shard=next((run/'events').glob('w-*/*.jsonl'));size=shard.stat().st_size
            with shard.open('ab') as stream:stream.write(b'{"partial":')
            writer=RollingWriter(node,dict(entities=[],descriptors=[]));inputs=Inputs(node,writer)
            self.assertFalse(inputs.poll())
            with shard.open('r+b') as stream:stream.truncate(size)
            target.emit(dict(t='health',kind='heartbeat',sev='info',ts=raw_clock(),detail=dict(scope='business',action='second')));target.flush()
            self.assertTrue(inputs.poll());target.close();inputs.close();writer.close()
            with StoreView(node) as view:
                rows=[r['detail'] for r in view.iter_monitor_events(('health',)) if r.get('detail',{}).get('scope')=='business_input']
            self.assertEqual([r['record']['detail']['action'] for r in rows],['first','second'])
            self.assertEqual([r['source_status'] for r in rows],['backlog','ready'])
