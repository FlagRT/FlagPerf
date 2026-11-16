"""Subprocess harness. Production has no fake-provider switch."""
import sys
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
    with patch.object(Registry,'create',factory):
        raise SystemExit(main())
