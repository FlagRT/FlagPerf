"""Only test harnesses inject a fake provider; shipped CLI has no fake switch."""
import argparse
from unittest.mock import patch
from flagprof.monitor.node import start, worker
from flagprof.monitor.session import configuration
from flagprof.registry import Registry
from .fake_entry import factory, isolated_leases


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',required=True)
    parser.add_argument('--worker',action='store_true'); parser.add_argument('--config')
    args=parser.parse_args()
    with isolated_leases(), patch.object(Registry,'create',factory):
        raise SystemExit(worker(args.root) if args.worker else start(args.root,'nvidia','0',
            configuration(args.config),module='tests.monitor.fake_agent'))
