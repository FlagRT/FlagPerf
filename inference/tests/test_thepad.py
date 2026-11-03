# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""PPU identity/format negatives and Linux process primitive regression."""
import json
import mmap
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from flagprof.registry import Registry
from flagprof.store import _pidfd
from flagprof.vendors.contracts import evaluate_conditions
from flagprof.vendors.thead.evidence import derive
from flagprof.vendors.thead.format import Taxonomy


class TheadTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux', 'Linux persisted Store')
    def test_two_window_clocks_do_not_double_count_overlapping_timestamps(self):
        import tempfile
        from flagprof.analyzers.gap import gap
        from flagprof.analyzers.workspace import Workspace
        from flagprof.offline.pipeline import import_prepared, prepare
        from flagprof.store import StoreView
        from flagprof.store._offline_writer import offline_writer
        from flagprof.store.manifest import create_run
        fixture = json.loads((Path(__file__).parent / 'fixtures/thead/gemm-fragment.json').read_bytes())
        with tempfile.TemporaryDirectory() as work:
            work = Path(work)
            run = create_run(work / 'store')
            with offline_writer(run) as writer:
                for index in range(2):
                    doc = json.loads(json.dumps(fixture))
                    doc['window_test_identity'] = index
                    doc['traceEvents'] = [e for e in doc['traceEvents'] if e.get('cat') != 'kernel'] + [
                        dict(ph='X', cat='kernel', name='isolated', pid=0, tid=1, ts=2, dur=3, args=dict(device=0, stream=1)),
                        dict(ph='X', cat='Trace', name='PyTorch Profiler window', pid=1, tid=1, ts=0, dur=10, args={})]
                    source = work / ('source' + str(index) + '.json')
                    source.write_text(json.dumps(doc))
                    prepared = prepare(source, work / ('prepared' + str(index)), vendor='thead', analysis_input=dict(
                        capture_normal_close=False, clock=dict(clock_valid=False)))
                    import_prepared(source, writer, prepared, vendor='thead')
            with StoreView(run) as view:
                workspace = Workspace(view, work / 'analysis')
                try:
                    workspace.ingest()
                    gap(workspace, {})
                    rows = list(workspace.table('perf.gap'))
                    self.assertEqual(len(rows), 2)
                    self.assertEqual([r['all_activity_union_ns'] for r in rows], [3000, 3000])
                    self.assertEqual([r['categories']['compute']['sum_ns'] for r in rows], [3000, 3000])
                    self.assertEqual(view.device_analysis_context()['source_clock'], 'per_window')
                finally:
                    workspace.db.close()

    @unittest.skipUnless(sys.platform == 'linux', 'Linux persisted Store')
    def test_window_projection_is_persisted_and_tampering_is_rejected(self):
        import tempfile
        from flagprof.offline.pipeline import import_prepared, prepare
        from flagprof.store import StoreView
        from flagprof.store._offline_writer import offline_writer
        from flagprof.store.manifest import create_run
        root = Path(__file__).parent / 'fixtures/thead'
        with tempfile.TemporaryDirectory() as work:
            work = Path(work)
            prepared = prepare(root / 'gemm-fragment.json', work / 'prepared', vendor='thead',
                analysis_input=dict(capture_normal_close=False,
                    capture_scope='scheduled_window_not_complete_request',
                    clock=dict(clock_valid=False), window_identity=dict(window_kind='scheduled_window')))
            run = create_run(work / 'store')
            with offline_writer(run) as writer:
                imported = import_prepared(root / 'gemm-fragment.json', writer, prepared, vendor='thead')
            self.assertEqual(imported['status'], 'complete')
            with StoreView(run) as view:
                list(view.iter_native_relations())
                context = view.device_analysis_context()
                self.assertEqual(context['status'], 'available')
                self.assertFalse(context['capture_normal_close'])
                rows = list(view.iter_device_events(projection='analysis'))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['category'], 'compute')
                path = next((Path(run) / 'derived').glob('*/analysis_intervals.jsonl'))
                path.write_bytes(path.read_bytes() + b'\n')
                with self.assertRaises(ValueError):
                    list(view.iter_device_events(projection='analysis'))

    def test_windowed_cli_forwards_requested_timeout(self):
        from unittest.mock import Mock, patch
        from flagprof.cli import main
        target = Mock(return_value=124)
        with patch('flagprof.record.record', target):
            result = main(['record', '--vendor', 'thead', '--capture-mode', 'windowed',
                           '--timeout', '1.25', '--store', 'unused', '--', sys.executable, 'target.py'])
        self.assertEqual(result, 124)
        self.assertEqual(target.call_args.kwargs['timeout'], 1.25)

    def test_real_exporter_fragment_keeps_device_and_missing_project_link(self):
        import hashlib
        from flagprof.vendors.prepare import prepare_import
        root = Path(__file__).parent / 'fixtures/thead'
        raw = (root / 'gemm-fragment.json').read_bytes()
        provenance = json.loads((root / 'provenance.json').read_text())
        self.assertEqual(hashlib.sha256(raw).hexdigest(), provenance['fragment_sha256'])
        result = prepare_import(raw, vendor='thead')
        self.assertEqual(len(result.events), 1)
        event = result.events[0]
        self.assertIn('gemm', event['name'])
        self.assertEqual(event['dev'], 0)
        self.assertIsInstance(event['stream'], int)
        self.assertEqual(event['link']['status'], 'unlinked')
        self.assertFalse(any(row.get('state') == 'owned' for row in result.native))

    def test_host_identity_requires_exact_uuid_and_sourced_fields(self):
        import tempfile
        from types import SimpleNamespace
        from flagprof.vendors.thead.session import hardware_identity
        torch = SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 0,
            get_device_properties=lambda index: SimpleNamespace(uuid='GPU-observed'),
            get_device_name=lambda index: 'PPU-ZW810E'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'identity.json'
            host = dict(source='ppu-smi', devices=[dict(model='PPU-ZW810E', uuid='observed',
                pci_bdf='00000001:C9:00.0', driver='1.5.1-1d747a', host_device_node='/dev/alixpu_ppu0')])
            path.write_text(json.dumps(host))
            with patch.dict(sys.modules, torch=torch), patch.dict(os.environ, FLAGPROF_THEAD_IDENTITY=str(path)):
                self.assertEqual(hardware_identity()['model'], 'PPU-ZW810E')
                host['devices'][0]['uuid'] = 'wrong'
                path.write_text(json.dumps(host))
                with self.assertRaisesRegex(ValueError, 'UUID mismatch'):
                    hardware_identity()
                host['devices'][0]['uuid'] = 'observed'
                host['devices'][0].pop('driver')
                path.write_text(json.dumps(host))
                with self.assertRaisesRegex(ValueError, 'missing sourced PPU driver'):
                    hardware_identity()
                host['devices'][0]['driver'] = '1.5.1-1d747a'
                host['source'] = 'xpu-smi'
                path.write_text(json.dumps(host))
                with self.assertRaisesRegex(ValueError, 'sourced PPU-ZW810E'):
                    hardware_identity()

    def test_explicit_collective_metadata_classifies_and_unknown_names_stay_compute(self):
        from flagprof.vendors.thead.analysis import activity
        row = activity(dict(ph='X', cat='kernel', name='some_all_reduce_impl',
                            args={'device': 0, 'stream': 9, 'Collective name': 'allreduce'}), {}, '')
        self.assertEqual(row['category'], 'comm')
        self.assertEqual(row['classification_basis'], 'explicit_collective_metadata')
        row = activity(dict(ph='X', cat='kernel', name='void ppu_allreduce_v1(float*)',
                            args={'device': 0, 'stream': 9}), {}, '')
        self.assertEqual(row['category'], 'compute')
        self.assertEqual(row['classification_basis'], 'cat')
        row = activity(dict(ph='X', cat='kernel', name='gemm_ktype0_aiu1_mtype1',
                            args={'device': 0, 'stream': 1}), {}, '')
        self.assertEqual(row['category'], 'compute')

    def test_clock_mapping_requires_pinned_exporter_and_unique_base(self):
        from flagprof.vendors.thead.analysis import clock
        capture = dict(trace=dict(sha256='0' * 64), metadata=dict(identities=dict(collector=dict(
            facts=dict(collector=dict(version='2.10.0')),
            evidence=dict(observations=[dict(trace_metadata=dict(baseTimeNanoseconds=123))])))))
        result = clock('.', capture)
        self.assertEqual(result['project_transforms'][0]['base_time_ns'], 123)
        self.assertFalse(result['clock_valid'])
        collector = capture['metadata']['identities']['collector']
        collector['facts']['collector']['version'] = 'unknown'
        self.assertEqual(clock('.', capture)['project_transforms'], [])
        collector['facts']['collector']['version'] = '2.10.0'
        collector['evidence']['observations'].append(dict(trace_metadata=dict(baseTimeNanoseconds=456)))
        self.assertEqual(clock('.', capture)['project_transforms'], [])

    def test_contract_is_distinct_and_rejects_wrong_stack(self):
        thead = Taxonomy()
        from flagprof.vendors.nvidia.format import Taxonomy as Nvidia
        self.assertNotEqual(thead.identity()['sha256'], Nvidia().identity()['sha256'])
        self.assertEqual(Registry().describe('thead')['format_vendor'], 'thead')
        facts = dict(hardware=dict(model='P800', driver='1.5.1-1d747a'),
                     collector=dict(name='torch.profiler', version='2.9.0'),
                     actual=dict(record_shapes=True), exporter=dict(schema_version=1))
        failures = {x['id'] for x in evaluate_conditions(thead.contract['conditions'], facts) if x['state'] == 'fail'}
        self.assertEqual(failures, {'hardware.model', 'collector.version'})

    def test_missing_exporter_flags_are_unknown(self):
        config = {}
        exporter = {}
        evidence = dict(observations=[])
        derive(dict(schemaVersion=1, traceEvents=[], record_shapes=1), dict(name='torch.profiler', version='2.10.0'),
               dict(path='trace.json'), {}, config, exporter, evidence)
        self.assertIs(config['actual']['values']['record_shapes'], True)
        self.assertIsNone(config['actual']['values']['profile_memory'])
        derive(dict(schemaVersion=1, traceEvents=[]), dict(name='torch.profiler', version='2.10.0'),
               dict(path='trace.json'), {}, config, exporter, evidence)
        self.assertIsNone(config['actual']['values']['record_shapes'])

    @unittest.skipUnless(_pidfd.supported(), 'Linux procfs')
    def test_unreaped_child_is_not_an_active_writer(self):
        from flagprof.store.manifest import process_alive, proc_key
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(.1)'])
        try:
            key = proc_key(child.pid)
            self.assertTrue(process_alive(key))
            deadline = time.monotonic() + 5
            fields = None
            while time.monotonic() < deadline:
                fields = Path('/proc/' + str(child.pid) + '/stat').read_text().rsplit(')', 1)[1].split()
                if fields[0] == 'Z':
                    break
                time.sleep(.01)
            self.assertEqual(fields[0], 'Z')
            self.assertFalse(process_alive(key))
            self.assertFalse(process_alive(dict(key, proc_start_ticks=key['proc_start_ticks'] + 1)))
        finally:
            child.wait(timeout=5)

    @unittest.skipUnless(_pidfd.supported(), 'Linux memory files')
    def test_raw_memfd_is_anonymous_shared_memory(self):
        from types import SimpleNamespace
        with patch.object(_pidfd, 'os', SimpleNamespace(fsencode=os.fsencode, strerror=os.strerror)):
            fd = _pidfd.create_memfd('flagprof-test')
        try:
            os.ftruncate(fd, 4096)
            with mmap.mmap(fd, 4096) as memory:
                memory[:4] = b'test'
                self.assertEqual(os.pread(fd, 4, 0), b'test')
            self.assertFalse(os.get_inheritable(fd))
        finally:
            os.close(fd)

    @unittest.skipUnless(_pidfd.supported(), 'Linux process groups')
    def test_owned_timeout_stops_target(self):
        from flagprof.offline.process import run_owned
        start = time.monotonic()
        result = run_owned([sys.executable, '-c', 'import time;time.sleep(30)'], timeout=.2)
        self.assertEqual(result.returncode, 124)
        self.assertTrue(result.timed_out)
        self.assertLess(time.monotonic() - start, 8)
        with self.assertRaises(ProcessLookupError):
            os.kill(result.pid, 0)

    @unittest.skipUnless(_pidfd.supported(), 'Linux pidfds')
    def test_raw_pidfd_wrapper_signals_exact_child(self):
        import signal as signal_module
        from types import SimpleNamespace
        from flagprof.store import _pidfd as raw
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
        fd = None
        try:
            with patch.object(raw, 'os', SimpleNamespace(strerror=os.strerror)):
                fd = raw.open_pidfd(child.pid)
            self.assertTrue(process_alive_child(child.pid))
            raw.send_signal(fd, signal_module.SIGKILL)
            child.wait(timeout=5)
        finally:
            if fd is not None:
                os.close(fd)


def process_alive_child(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


if __name__ == '__main__':
    unittest.main()
