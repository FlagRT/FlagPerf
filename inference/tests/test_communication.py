# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Real Gloo communication hooks preserve functional and imperative outputs."""
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'optional torch CPU runtime')
class CommunicationTests(unittest.TestCase):
    def test_functional_group_identity_payload_and_restore(self):
        import torch
        import torch.distributed as dist
        import torch.distributed._functional_collectives as functional
        from flagprof.probes.communication import CommunicationProbe
        rows=[]
        context=SimpleNamespace(closed=False,current_scope=lambda:None,
            probe_state=lambda:dict(generation=dict(generation_step=0)),
            writer=SimpleNamespace(emit=rows.append))
        manager=SimpleNamespace(context=context,config={},state=dict(probes={}))
        originals=(dist.all_reduce,functional.all_reduce,functional.all_gather_tensor)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,FLAGPROF_GROUP_ID='test-group'):
            dist.init_process_group('gloo',init_method='file://'+str(Path(directory)/'rendezvous'),rank=0,world_size=1)
            control=dist.new_group(backend='gloo')
            probe=CommunicationProbe(manager,control)
            try:
                probe.install()
                x=torch.arange(4,dtype=torch.float32)
                dist.all_reduce(x)
                y=functional.all_reduce(x,'sum',dist.group.WORLD).wait()
                z=functional.all_gather_tensor(x,0,dist.group.WORLD).wait()
                self.assertTrue(torch.equal(x,y))
                self.assertTrue(torch.equal(x,z))
                self.assertEqual([r['op'] for r in rows],['allreduce','allreduce','allgather'])
                self.assertEqual([r['group_seq'] for r in rows],[0,1,2])
                self.assertTrue(all(r['group']=='test-group/model' and r['bytes']==16 for r in rows))
                self.assertTrue(all(r['ext']['communication']['logical_output_bytes']==16 for r in rows))
            finally:
                probe.uninstall()
                dist.destroy_process_group(control)
                dist.destroy_process_group()
        self.assertEqual((dist.all_reduce,functional.all_reduce,functional.all_gather_tensor),originals)


if __name__=='__main__':unittest.main()
