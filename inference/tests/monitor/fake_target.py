"""CPU-only subprocess injection for public final and launch paths."""
import runpy
import sys
from unittest.mock import patch
from flagprof.registry import Registry
from tests.store.test_final_lifecycle import FinalBackend

original=Registry.create

def factory(self,ident,**kwargs):
    if ident=='nvidia':return FinalBackend(),{}
    return original(self,ident,**kwargs)

if __name__=='__main__':
    module=sys.argv.pop(1)
    with patch.object(Registry,'create',factory), \
            patch('flagprof.vendors.nvidia.execution.initialize',return_value='cuda:0'), \
            patch('flagprof.vendors.nvidia.execution.physical_identity',return_value=dict(entity='chip:0')), \
            patch('flagprof.vendors.nvidia.execution.allocator_memory',return_value=dict(allocated_bytes=0,reserved_bytes=0)):
        runpy.run_module(module,run_name='__main__',alter_sys=True)
