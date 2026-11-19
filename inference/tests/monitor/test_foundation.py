import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from flagprof.monitor.compat import descriptor
from flagprof.monitor.supervision import Heartbeat, reclaim
from flagprof.store import StoreView, Writer, create_run, finalize
from flagprof.store.manifest import atomic_json
from flagprof.telemetry.contract import Call, Observation, raw_clock
from flagprof.telemetry.sampler import Sampler
from flagprof.vendors.ascend.telemetry_health import health_events


class FoundationTests(unittest.TestCase):
    def test_smi_round_jitter_preserves_frozen_sampling_configuration(self):
        from flagprof.monitor.node import sampling_interval
        self.assertEqual(sampling_interval(1000,dict(source='npu-smi',interval_quantum_ms=1000,minimum_interval_ms=1182)),2000)
        self.assertEqual(sampling_interval(1000,dict(source='npu-smi',interval_quantum_ms=1000,minimum_interval_ms=1700),2000),2000)
        with self.assertRaisesRegex(ValueError,'sampling floor increased'):
            sampling_interval(1000,dict(source='npu-smi',interval_quantum_ms=1000,minimum_interval_ms=2100),2000)

    def test_wall_clock_jump_does_not_extend_deadline(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'heartbeat'
            with patch('time.monotonic', side_effect=[10, 11, 21]), patch('time.time', side_effect=AssertionError):
                heart = Heartbeat(path, 10)
                self.assertFalse(heart.expired())
                self.assertTrue(heart.expired())

    def test_unreapable_child_stops_with_finite_waits(self):
        waits = []
        class Child:
            returncode = None
            def poll(self): return None
            def terminate(self): pass
            def kill(self): pass
            def wait(self, timeout):
                waits.append(timeout)
                raise subprocess.TimeoutExpired('own-child', timeout)
        self.assertFalse(reclaim(Child())['confirmed'])
        self.assertEqual(waits, [3, 3])

    def test_process_churn_releases_scheduler_maps(self):
        sampler = Sampler([], 100)
        for i in range(1000):
            entity = 'proc:test:'+str(i)
            call = Call(entity, entity, entity, ('rss',), lambda: None, 'slow')
            sampler.calls.append(call)
            sampler.next[entity] = i
            sampler.previous[(entity,'slow')] = i
            sampler.remove([entity])
        self.assertEqual((sampler.calls, sampler.next, sampler.previous), ([], {}, {}))

    def test_health_survives_sampling_and_public_reader(self):
        with tempfile.TemporaryDirectory() as folder:
            run = create_run(Path(folder)/'monitor')
            atomic_json(run/'inventory.json', dict(status='complete', entities=[],
                descriptors=[dict(name='nvidia.violation.thermal',unit='ns',kind='cumulative')]))
            sampler = Sampler([], 100)
            point = Observation('chip:0', {'health':2}, 10, 11,
                                health=health_events('chip:0', 2, 11))
            sampler._publish([(Call('a','a','chip:0',('health',),None),point)])
            points, _ = sampler.tick(schedule=False)
            with Writer(run,kind='counter') as writer:
                for event in points[0].health: writer.emit(event)
            finalize(run)
            with StoreView(run) as view:
                events = list(view.iter_monitor_events(('health',)))
                self.assertEqual(events[0]['kind'], 'device_error')
                self.assertEqual(events[0]['sev'], 'error')
                self.assertEqual(events[0]['evidence_id'][0], 'event')
                self.assertEqual(view.monitor_descriptors()[0]['roles'], ['restriction.cumulative.thermal'])

    def test_explicit_roles_are_not_inferred_from_vendor_name(self):
        self.assertEqual(descriptor(dict(name='nvidia.violation.thermal',roles=['custom']))['roles'], ['custom'])

    def test_device_process_memory_follows_registered_process_lifecycle(self):
        host=Call('proc:test','proc:test','proc:test',('rss',),None,'slow')
        sdk=Call('sdk','chip:0','chip:0',('process.count',),None,'slow')
        sampler=Sampler([host,sdk],100)
        point=Observation('proc:test/chip:0',{'process.device_memory':123},10,11)
        sampler._publish([(sdk,point)])
        points,_=sampler.tick(schedule=False)
        self.assertEqual(points[0].values['process.device_memory'],123)
        sampler.remove([host.key])
        self.assertFalse(sampler.previous)
        sampler._publish([(sdk,point)])
        self.assertEqual(sampler.tick(schedule=False)[0],[])


if __name__ == '__main__': unittest.main()
