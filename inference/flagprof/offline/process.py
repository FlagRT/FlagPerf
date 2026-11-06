# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Bounded parent waits and owned process-group signal cleanup."""
import os
import signal
import subprocess
import threading
import time
import math


def run_owned(command,*,timeout,**options):
    if not math.isfinite(timeout) or timeout<=0:raise ValueError('positive finite process timeout required')
    previous={};received=[]
    with subprocess.Popen(command,start_new_session=True,**options) as child:
        from flagprof.store.manifest import proc_key
        try:process_key=proc_key(child.pid)
        except OSError:process_key=None
        signal_deadline=[None]
        def forward(signum,frame):
            received.append(signum);signal_deadline[0]=time.monotonic()+5
            try:os.killpg(child.pid,signum)
            except ProcessLookupError:pass
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT,signal.SIGTERM):
                previous[sig]=signal.signal(sig,forward)
        timed_out=False
        try:
            deadline=time.monotonic()+timeout
            try:
                while True:
                    until=min(deadline,signal_deadline[0] or deadline)
                    remaining=until-time.monotonic()
                    if remaining<=0:raise subprocess.TimeoutExpired(command,timeout)
                    try:
                        code=child.wait(timeout=min(.25,remaining));break
                    except subprocess.TimeoutExpired:continue
            except subprocess.TimeoutExpired:
                timed_out=not received
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=0 if received else 5)
                except subprocess.TimeoutExpired:pass
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait();code=124
            if received:code=128+received[0]
            result=subprocess.CompletedProcess(command,code)
            result.pid=child.pid;result.process_key=process_key
            result.timed_out=timed_out;result.signal=received[0] if received else None
            return result
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
