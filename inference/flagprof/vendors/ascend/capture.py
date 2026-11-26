"""CANN offline analysis of a private copy of sealed raw output."""
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from flagprof.offline.limits import walk_files


def check_analysis_log(path):
    """The pinned CANN analyser can report failed tasks while exiting zero."""
    with Path(path).open(errors='replace') as stream:
        for line in stream:
            if 'Failed to export CANN' in line or ('Task [' in line and '] run failed' in line):
                raise ValueError(f'official offline analysis failed: {line.strip()} (log: {path})')


def analyse_trace(directory, raw, temporary, timeout):
    working = Path(temporary) / 'analysis'
    source = raw / 'data'
    working.mkdir(mode=0o700)
    # CANN requires private, writable analysis inputs even when the capture
    # was sealed on a group-writable volume or mounted read-only afterwards.
    for original in walk_files(source, max_files=4096, max_entries=8192):
        relative = original.relative_to(source)
        parent = working
        for name in relative.parts[:-1]:
            parent = parent / name
            parent.mkdir(mode=0o700, exist_ok=True)
        copied = parent / relative.name
        shutil.copyfile(original, copied)
        copied.chmod(0o600)
    log_path = directory/('analyse-'+uuid.uuid4().hex+'.log')
    with log_path.open('xb') as log:
        from flagprof.offline.process import run_owned
        from flagprof.offline.worker import lease_fds
        cp=run_owned([sys.executable,'-B','-m','flagprof.vendors.ascend.capture',str(working)],stdout=log,stderr=subprocess.STDOUT,timeout=timeout,pass_fds=lease_fds())
    if cp.returncode:raise ValueError(f'official offline analysis failed (exit {cp.returncode}, log: {log_path})')
    check_analysis_log(log_path)
    traces=[]
    for path in walk_files(working,max_files=4096,max_entries=8192):
        if path.name=='trace_view.json':traces.append(path)
        if len(traces)>1:raise ValueError('single-trace budget')
    if len(traces)!=1:raise ValueError(f'one analysed trace required (log: {log_path})')
    return traces[0]


if __name__ == '__main__':
    from torch_npu.profiler.profiler import analyse
    analyse(sys.argv[1], max_process_number=1)
