"""PPU local cleanup must reap adopted compiler children in its own group."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vendors.thead.host import cleanup_process


class LocalCleanupTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux', 'Linux process groups and subreaper required')
    def test_reaps_adopted_compiler_children_without_touching_other_groups(self):
        # Use an isolated subreaper to reproduce a local CLI running as PID 1.
        # The compiler child ignores TERM; both the leader and its descendant
        # must disappear, while an unrelated session remains alive.
        code = r'''
import ctypes, json, os, signal, subprocess, sys, time
from vendors.thead.host import cleanup_process
assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)"
leader = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c',sys.argv[1]],stdout=subprocess.PIPE,text=True); print(p.pid,p.stdout.readline().strip(),flush=True); time.sleep(60)"
other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
proc = subprocess.Popen([sys.executable, '-c', leader, child], start_new_session=True, stdout=subprocess.PIPE, text=True)
try:
    line = proc.stdout.readline().split()
    assert len(line) == 2 and line[1] == 'ready', line
    descendant = int(line[0])
    result = cleanup_process(proc, timeout=0.4)
    assert result['process_group_absent'], result
    assert not os.path.exists('/proc/%d' % descendant), 'compiler child remains'
    assert proc.returncode is not None, 'leader not reaped'
    assert other.poll() is None, 'unrelated group was affected'
    print(json.dumps(result))
finally:
    try: os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    proc.wait(timeout=3)
    other.terminate(); other.wait(timeout=3)
'''
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                   PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-B', '-c', code], env=env,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertTrue(json.loads(result.stdout)['process_group_absent'])

    def test_unconfirmed_group_does_not_pass_cleanup(self):
        proc = SimpleNamespace(pid=123456789, poll=lambda: None, returncode=None)
        with patch('vendors.thead.host.os.killpg'), \
                patch('vendors.thead.host.os.waitpid', side_effect=ChildProcessError):
            result = cleanup_process(proc, timeout=0)
        self.assertFalse(result['process_group_absent'])
        self.assertEqual(result['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
