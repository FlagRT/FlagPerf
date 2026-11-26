"""Subprocess harness. Production has no fake-provider switch."""
import sys
import tempfile
from pathlib import Path
from contextlib import contextmanager
from flagprof.monitor.leases import Leases


@contextmanager
def isolated_leases():
    acquire = Leases.acquire
    with tempfile.TemporaryDirectory(prefix='fake-device-leases-') as directory:
        def local(self, keys, directory_override=None):
            return acquire(self, keys, directory_override or Path(directory)/'leases')
        with patch.object(Leases, 'acquire', local):
            yield

from unittest.mock import patch
from flagprof.registry import Registry
from .fake_provider import Provider

create=Registry.create


def factory(self,ident,**kwargs):
    if ident.startswith('telemetry.'):
        return Provider(),dict(shipped_source_closure={'sha256':'test'})
    return create(self,ident,**kwargs)


if __name__=='__main__':
    from flagprof.monitor.__main__ import main
    if '--sample' not in sys.argv:sys.argv.append('--sample')
    with isolated_leases(), patch.object(Registry,'create',factory):
        raise SystemExit(main())
