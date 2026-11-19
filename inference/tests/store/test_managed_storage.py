"""Exercise the public storage contract with real worker processes and real SQLite spill."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from flagprof import cli
from flagprof.delivery import publish_tree, restore
from flagprof.offline.bundle import make_bundle
from flagprof.offline.service import import_bundle, inspect_run
from flagprof.offline.workspace import private_name
from flagprof.store import StoreView


SOURCE = Path(__file__).resolve().parents[1] / 'correlate/fixtures/t8.1/nvidia.json'


class ManagedStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.capture = self.root/'producer-mount'/'sample'
        self.store = self.root/'result-mount'/'runs'
        raw = self.root/'raw'
        raw.mkdir()
        shutil.copyfile(SOURCE, raw/'trace.json')
        make_bundle(raw, raw/'trace.json', self.capture, vendor='nvidia',
                    metadata=dict(normal_close=True, target_returncode=0))

    def test_removed_options_are_rejected(self):
        for command in (['import','src','--store','dst','--work','w'],
                        ['import','src','--store','dst','--scratch','s'],
                        ['page','run','--work','w'], ['inspect','run','--bounded'],
                        ['report','run','--scratch','s'], ['report','oracle','--perf-run','r'],
                        ['launch','--store','s','--flagcx'],
                        ['record','--store','s','--vendor','nvidia','--capture-filesystem','overlay']):
            with self.subTest(command=command), redirect_stderr(StringIO()) as err, self.assertRaises(SystemExit):
                try:
                    cli.main(command)
                finally:
                    self.assertIn('unrecognized arguments', err.getvalue())

    def test_retry_concurrency_checkpoint_corruption_and_source_integrity(self):
        def run(name, options=None):
            return import_bundle(self.capture, store=self.store, run_id=name, options=options)
        with ThreadPoolExecutor(2) as pool:
            both = list(pool.map(run, ('same', 'same')))
        self.assertEqual(both[0]['import_key'], both[1]['import_key'])
        key = both[0]['import_key']
        check = self.store/private_name()/'checkpoints'/key
        (check/'prepared/committed.json').write_text('{}')
        self.assertEqual(run('after-damage')['reused'], ['scan'])
        with self.assertRaises(FileExistsError):
            run('same', dict(kind='compute'))
        self.assertNotEqual(run('modified', dict(kind='compute'))['import_key'], key)
        self.assertFalse(list((self.store/private_name()/'tmp').iterdir()))
        with (self.capture/'sealed-trace/data/trace.json').open('ab') as stream:
            stream.write(b' ')
        with self.assertRaisesRegex(ValueError,'content mismatch'):
            run('changed')

    def test_cleanup_preserves_active_checkpoints_and_user_data(self):
        from unittest.mock import patch
        from flagprof.offline.workspace import checkpoint
        keep = self.store / 'user-result'
        keep.mkdir(parents=True)
        (keep / 'evidence').write_text('original')
        with checkpoint(self.store, 'a' * 64) as active:
            with checkpoint(self.store, 'b' * 64) as old:
                (old / 'index').touch()
            with checkpoint(self.store, 'c' * 64) as recent:
                (recent / 'index').touch()
            os.utime(old, (1, 1))
            os.utime(active, (1, 1))
            with checkpoint(self.store, 'd' * 64):
                self.assertFalse(old.exists())
                self.assertTrue(recent.exists())
            disk = shutil.disk_usage(self.store)._replace(free=0)
            with patch('shutil.disk_usage', return_value=disk), checkpoint(self.store, 'd' * 64):
                self.assertFalse(recent.exists())
                self.assertTrue(active.exists())
                self.assertEqual((keep / 'evidence').read_text(), 'original')

    def test_nested_archive_work_on_a_leased_volume(self):
        # A monitor release restores its archive under an existing operation.
        body = """import sys
from flagprof.offline.workspace import attempt, publish_run
with attempt(sys.argv[1]) as (outer, lease):
    with attempt(outer) as (inner, nested_lease):
        staged = inner / 'staged'
        staged.mkdir()
        (staged / 'payload').write_text('restored')
        publish_run(staged, outer / 'published')
    assert (outer / 'published' / 'payload').read_text() == 'restored'
"""
        subprocess.run([sys.executable, '-B', '-c', body, str(self.store)], check=True, timeout=15)

    def test_recover_interrupted_publication(self):
        # Stop after independent verification, just before the no-replace rename.
        body = '''import os, sys
from flagprof.offline import worker, workspace, service
with workspace.attempt(sys.argv[2]) as (worker._active, lease):
    workspace.publish_run=lambda *_: os._exit(78)
    service._import_bundle(sys.argv[1],sys.argv[2],'recovered',None,None)
'''
        child = subprocess.run([sys.executable,'-B','-c',body,str(self.capture),str(self.store)], timeout=30)
        self.assertEqual(child.returncode, 78)
        self.assertFalse((self.store/'recovered').exists())
        self.assertEqual(len(list(self.store.glob('.flagprof-import-*'))), 1)
        result = import_bundle(self.capture, store=self.store, run_id='recovered')
        self.assertEqual(result['reused'], ['derived'])
        self.assertFalse(list(self.store.glob('.flagprof-import-*')))
        self.assertFalse(list((self.store/private_name()/'tmp').iterdir()))

    def test_capture_reuses_committed_raw_after_interrupted_handoff(self):
        from flagprof.offline.capture import finish_capture, seal_capture
        from flagprof.offline.checkpoints import fileinfo
        from flagprof.store.manifest import atomic_json
        capture = self.root/'recorded'
        official = capture/'official'
        official.mkdir(parents=True)
        shutil.copyfile(SOURCE, official/'trace.json')
        metadata = dict(normal_close=True, target_returncode=0, raw_path='official')
        atomic_json(capture/'seal-request.json', dict(metadata=metadata, vendor='nvidia',
                    target_exit=dict(returncode=0,timed_out=False)))
        seal_capture(capture, metadata, 'nvidia')
        before = fileinfo(capture/'sealed-raw/committed.json')
        (capture/'capture-input.json').rename(capture/'capture-input.json.partial')
        (capture/'capture.json.partial').write_text('unfinished')
        result = finish_capture(capture)
        self.assertTrue(result['normal_close'])
        self.assertEqual(result['raw']['manifest'], before)
        self.assertEqual(result, finish_capture(capture))

    def test_archive_reopens_after_both_sources_move(self):
        original = import_bundle(self.capture, store=self.store, run_id='sample')
        package = self.root/'archive'
        # Both logical roots are named sample and reside in unrelated trees.
        publish_tree([original['run']], package)
        manifest = (package/'manifest.json').read_text()
        self.assertNotIn('producer-mount', manifest)
        self.assertNotIn('result-mount', manifest)
        self.assertNotIn(str(self.root), manifest)
        self.capture.rename(self.capture.with_name('held-capture'))
        self.store.rename(self.store.with_name('held-store'))
        restored = restore(package, self.root/'client')
        run = next(Path(p) for p in restored['mapping'].values() if (Path(p)/'plan-b-capture.json').exists())
        self.assertEqual(inspect_run(run, offset=0, limit=1)['total'], original['count'])
        with StoreView(run) as view:
            self.assertTrue(Path(view.analysis_source()['capture_directory']).is_relative_to(self.root/'client'))

    def test_private_sqlite_spill_and_readonly_queries(self):
        system_tmp = self.root/'system-temp'
        system_tmp.mkdir()
        audit = self.root/'audit'
        audit.mkdir()
        # A fresh interpreter initializes SQLite's VFS directory. Force it to
        # open a real delete-on-close TEMP file, then observe the live fd path.
        (audit/'sitecustomize.py').write_text("""import os, sqlite3, stat
from pathlib import Path
p=Path(os.environ.get('SQLITE_TMPDIR','/'))
def regular_fds():
    result = {}
    for fd in Path('/proc/self/fd').iterdir():
        try:
            if stat.S_ISREG(os.fstat(int(fd.name)).st_mode):
                result[fd.name] = os.readlink(fd)
        except OSError:
            pass
    return result
if p.parent.name=='tmp' and p.parent.parent.name.startswith('.flagprof-'):
    c=sqlite3.connect(':memory:')
    c.execute('pragma temp_store=FILE')
    c.execute('pragma temp.cache_size=8')
    before=regular_fds()
    c.execute('create temp table spill(x)')
    c.executemany('insert into spill values (zeroblob(4096))',[()]*1024)
    paths=[name for fd,name in regular_fds().items() if before.get(fd)!=name]
    Path(os.environ['TEST_SPILL_AUDIT'],str(os.getpid())+'.txt').write_text(os.linesep.join(paths))
    c.close()
""")
        env = dict(os.environ, TMPDIR=str(system_tmp), SQLITE_TMPDIR='/unrelated', FLAGPROF_SCRATCH='/removed-setting', TEST_SPILL_AUDIT=str(audit),
                   PYTHONPATH=os.pathsep.join([str(audit), str(SOURCE.parents[4])] + [os.environ.get('PYTHONPATH','')]))
        def call(argv):
            proc = subprocess.run([sys.executable,'-B','-m','flagprof',*map(str,argv)], env=env,
                                  capture_output=True, text=True, timeout=40)
            self.assertEqual(proc.returncode,0,proc.stderr)
            return json.loads(proc.stdout)
        def recorded():
            return {p: p.read_text().splitlines() for p in audit.glob('*.txt')}
        result = call(['import',self.capture,'--store',self.store,'--run-id','readonly'])
        first = recorded()
        self.assertEqual(len(first), 1)
        self.assertTrue(all(Path(name).is_relative_to(self.store) for name in next(iter(first.values()))))
        run = Path(result['run'])
        inputs = [run, self.capture]
        modes = {p: p.stat().st_mode for root in inputs for p in [root,*root.rglob('*')]}
        snapshot = {p: p.stat().st_mtime_ns for p in modes}
        try:
            for path in modes:
                path.chmod(0o555 if path.is_dir() else 0o444)
            with ThreadPoolExecutor(2) as pool:
                for value in pool.map(call, (['inspect',run,'--json'], ['page',run,'--limit','1'])):
                    self.assertTrue(value)
            second = recorded()
            clients = [v for p,v in second.items() if p not in first]
            self.assertEqual(len(clients), 2)
            self.assertTrue(all(v and Path(v[0]).is_relative_to(system_tmp) for v in clients), (system_tmp, clients))
            self.assertNotEqual(Path(clients[0][0]).parent, Path(clients[1][0]).parent)
            out = self.root/'report-volume'/'out'
            call(['report',run,'--out',out])
            reports = [v for p,v in recorded().items() if p not in second]
            self.assertEqual(len(reports), 1)
            self.assertTrue(reports[0] and Path(reports[0][0]).is_relative_to(out.parent))
            self.assertTrue(next(iter(first.values())))
            self.assertTrue((out/'report.html').is_file())
            self.assertEqual(snapshot, {p:p.stat().st_mtime_ns for p in modes})
        finally:
            for path,mode in modes.items():
                path.chmod(mode)
