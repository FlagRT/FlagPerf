import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from flagprof.monitor.retention import RollingWriter, CapacityError, collect, read_lease
from flagprof.monitor.session import configuration, initialize, load, edit, owned_size
from flagprof.store import StoreView
from flagprof.telemetry.contract import raw_clock


class RetentionTests(unittest.TestCase):
    def test_archived_sharing_remains_after_live_history_is_reclaimed(self):
        from flagprof.store.manifest import atomic_json,proc_key
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);writer=self.make(root/'node');writer.close();session=load(root/'node')
            ref=root/'monitor-ref.json'
            atomic_json(ref,dict(version=2,session_id=session['id'],session_path='node',window=[session['started_raw_ns'],raw_clock()],
                worker_proc_key=proc_key(),devices=[dict(entity='chip:0',status='available')],shared_devices=['chip:0']))
            with StoreView(ref) as view:self.assertEqual(view.monitor_status()['reference']['shared_devices'],['chip:0'])

    def test_control_write_checks_capacity_before_replacing_metadata(self):
        from flagprof.monitor.session import control_json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);writer=self.make(root)
            original=(root/'session.json').read_bytes();state=load(root)
            with patch('flagprof.monitor.session.owned_size',return_value=state['config']['total_bytes']):
                with self.assertRaises(CapacityError):control_json(root,state,root/'session.json',state)
            self.assertEqual((root/'session.json').read_bytes(),original)
            writer.close()

    def make(self, root, **options):
        config=configuration();config.update(min_free_bytes=0,pre_seconds=0,**options)
        initialize(root,'nvidia','0',config)
        return RollingWriter(root,dict(entities=[dict(entity='chip:0')],descriptors=[]))

    def point(self,value):return dict(t='counter',ts=raw_clock(),src='device',dev='chip:0',values={'memory.used':value})

    def test_reader_lease_pin_and_gc_have_one_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);writer=self.make(root)
            writer.emit(self.point(1));writer.rotate()
            writer.emit(self.point(2));writer.rotate()
            writer.emit(self.point(3));writer.close()
            segments=load(root)['segments']
            with edit(root) as state:
                state['config'].update(raw_seconds=0,raw_bytes=0)
                state['pins']['proof']=dict(window=segments[0]['window'])
            with read_lease(root,segments[1]['id']):
                with edit(root) as state:collect(root,state,raw_clock()+1_000_000)
                self.assertEqual([s['id'] for s in load(root)['segments']],[s['id'] for s in segments[:2]])
            with edit(root) as state:collect(root,state,raw_clock()+1_000_000)
            self.assertEqual(len(load(root)['segments']),1)
            with StoreView(root) as view:
                self.assertEqual([p['values']['memory.used'] for p in view.iter_monitor_events(('counter',))],[1])

    def test_capacity_stops_before_unreserved_write_and_keeps_fixed_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);writer=self.make(root,total_bytes=4<<20,control_reserve_bytes=128<<10)
            writer.emit(self.point(1))
            with edit(root) as state:
                state['pins']['proof']=dict(window=[state['started_raw_ns'],None])
            with patch('flagprof.monitor.retention.owned_size',return_value=4<<20):
                with self.assertRaises(CapacityError):writer.emit(self.point(2))
            writer.close()
            with StoreView(root) as view:
                points=list(view.iter_monitor_events(('counter',)))
            self.assertEqual([p['values']['memory.used'] for p in points],[1])
            self.assertLess(owned_size(root),4<<20)

    def test_missing_segment_is_visible_and_does_not_hide_later_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);writer=self.make(root)
            writer.emit(self.point(1));writer.rotate();writer.emit(self.point(2));writer.close()
            first=load(root)['segments'][0]
            original=root/'segments'/first['id'];original.rename(root/'retained-for-test')
            with StoreView(root) as view:
                points=list(view.iter_monitor_events(('counter',)))
                self.assertEqual([p['values']['memory.used'] for p in points],[2])
                self.assertEqual(len(view.monitor_coverage()['read_report']['errors']),1)


if __name__=='__main__':unittest.main()
