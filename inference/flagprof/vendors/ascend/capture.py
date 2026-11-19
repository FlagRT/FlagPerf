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
                raise ValueError('official offline analysis failed: '+line.strip()+'; see the analysis log')


def analyse_trace(directory, raw, temporary, timeout):
    working = Path(temporary) / 'analysis'
    shutil.copytree(raw/'data', working)
    log_path = directory/('analyse-'+uuid.uuid4().hex+'.log')
    with log_path.open('xb') as log:
        from flagprof.offline.process import run_owned
        from flagprof.offline.worker import lease_fds
        cp=run_owned([sys.executable,'-B','-m','flagprof.vendors.ascend.capture',str(working)],stdout=log,stderr=subprocess.STDOUT,timeout=timeout,pass_fds=lease_fds())
    if cp.returncode:raise ValueError('official offline analysis failed; sealed raw retained')
    check_analysis_log(log_path)
    traces=[]
    for path in walk_files(working,max_files=4096,max_entries=8192):
        if path.name=='trace_view.json':traces.append(path)
        if len(traces)>1:raise ValueError('single-trace budget')
    if len(traces)!=1:raise ValueError('one analysed trace required')
    return traces[0]


if __name__ == '__main__':
    from torch_npu.profiler.profiler import analyse
    analyse(sys.argv[1], max_process_number=1)
