# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""torchrun worker reusing attachments/capture, without inference TP2 rules."""
import argparse
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', required=True)
    parser.add_argument('--vendor', required=True, choices=('thead','kunlunxin','nvidia'))
    parser.add_argument('--preset', default='perf-lite', choices=('none','perf-lite','coverage','accuracy','perf','modules'))
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('training_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    rank = int(os.environ.get('RANK', '-1')); local_rank = int(os.environ.get('LOCAL_RANK', '-1'))
    if rank < 0 or local_rank < 0 or int(os.environ.get('WORLD_SIZE', '0')) < 2:
        raise ValueError('distributed observer requires torchrun with at least two ranks')
    if '--ddp' not in args.training_args:
        raise ValueError('training worker requires explicit --ddp')
    arguments = args.training_args[1:] if args.training_args[:1] == ['--'] else args.training_args
    destination = Path(args.store)/('rank-'+str(rank))
    destination.mkdir(parents=True, exist_ok=False)
    if args.vendor == 'kunlunxin':
        if os.environ.get('XPU_ENABLE_PROFILER_TRACING') != '1':
            raise ValueError('set XPU_ENABLE_PROFILER_TRACING=1 before torchrun')
        os.environ['XPU_CUPTI_ENABLE_DEVICE'] = str(local_rank)
    import torch
    torch.cuda.set_device(local_rank)
    os.environ['FLAGPROF_DEVICE'] = 'cuda:'+str(local_rank)
    # The existing capture target initializes each child before attachment.
    os.environ['FLAGPROF_LAUNCH_WORKER'] = '1'
    handle = None; code = 1; run = None; error = None
    try:
        if args.preset in ('perf','modules'):
            from flagprof.offline.capture import record_final
            code = record_final([sys.executable,'-B','-m','flagprof.training',*arguments],
                                store=destination/'store', vendor=args.vendor, preset=args.preset,
                                rank=rank, timeout=args.timeout, scratch=destination/'scratch')
        else:
            if args.preset != 'none':
                from flagprof.store import create_run
                from flagprof.runtime import attach
                run = create_run(destination/'store', identity=dict(training_ddp=True, rank=rank))
                handle = attach(run, args.vendor, preset=args.preset, rank=rank)
                if handle.state['status'] != 'running':
                    raise RuntimeError('training attachment did not start')
            from .__main__ import main as training
            code = training(arguments)
    except BaseException as exc:
        error = type(exc).__name__+': '+str(exc)
        raise
    finally:
        if handle:
            state = handle.close(normal=code==0,returncode=code)
            if state['status'] != 'closed': code = code or 2
        (destination/'worker.json').write_text(json.dumps(dict(schema='training-worker/1',rank=rank,
             local_rank=local_rank, world_size=int(os.environ['WORLD_SIZE']),returncode=code,
             status='complete' if code==0 else 'failed',run=str(run) if run else None,error=error),indent=2)+'\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
