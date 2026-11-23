import json
from pathlib import Path
import tempfile
import unittest
from flagprof.monitor.reference import bind, relocate, resolve_reference
from flagprof.store.manifest import atomic_json,proc_key


class ReferenceTests(unittest.TestCase):
    def test_partial_live_ambiguity_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);target=root/'target';target.mkdir()
            ref=target/'monitor-ref.json'
            data=bind(ref,root=root,topology='host',worker=proc_key(),devices=[dict(entity='chip:1',status='available')])
            data['window']=[100,300];atomic_json(ref,data)
            def monitor(name,status='complete'):
                path=root/'monitor'/name;path.mkdir(parents=True)
                atomic_json(path/'manifest.json',dict(status=status))
                atomic_json(path/'inventory.json',dict(status=status,started_raw_ns=150,ended_raw_ns=350,
                    boot_id=proc_key()['boot_id'],entities=[dict(entity='chip:1')]))
                return path
            first=monitor('a')
            result,path,_=resolve_reference(ref)
            self.assertEqual(result['status'],'partial');self.assertEqual(path,first)
            monitor('b')
            self.assertEqual(resolve_reference(ref)[0]['reason'],'ambiguous')
            bind(ref,monitor=first)
            moved=root/'analysis'/'result';moved.mkdir(parents=True)
            relocate(ref,moved/'monitor-ref.json')
            self.assertEqual(resolve_reference(moved/'monitor-ref.json')[1],first)
            inv=json.loads((first/'inventory.json').read_text());inv.update(status='running',ended_raw_ns=None)
            atomic_json(first/'inventory.json',inv)
            self.assertEqual(resolve_reference(ref)[0]['status'],'running')

    def test_identity_update_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ref=root/'monitor-ref.json'
            bind(ref,root=root,topology='host')
            bind(ref,worker=proc_key(),devices=[dict(entity='chip:3',status='available')])
            files=list((root/'targets').glob('*.json'))
            self.assertEqual(len(files),1)
            self.assertEqual(json.loads(files[0].read_text())['devices'][0]['entity'],'chip:3')


    def test_all_five_lifecycle_states(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ref=root/'target/monitor-ref.json';ref.parent.mkdir()
            self.assertEqual(resolve_reference(ref)[0]['status'],'absent')
            monitor=root/'monitor/m';monitor.mkdir(parents=True)
            data=bind(ref,root=root,monitor=monitor,worker=proc_key(),devices=[dict(entity='chip:1',status='available')])
            data['window']=[100,200];atomic_json(ref,data)
            for status in ('running','complete','partial','failed'):
                atomic_json(monitor/'manifest.json',dict(status=status))
                atomic_json(monitor/'inventory.json',dict(status=status,started_raw_ns=50,ended_raw_ns=300,
                    boot_id=proc_key()['boot_id'],entities=[dict(entity='chip:1')]))
                self.assertEqual(resolve_reference(ref)[0]['status'],status)

if __name__=='__main__':unittest.main()
