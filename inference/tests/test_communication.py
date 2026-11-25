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


def observed_batch_transfer(rank, directory):
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    from flagprof.probes.communication import CommunicationProbe
    from flagprof.training import observation_group
    dist.init_process_group('gloo', init_method='file://'+str(Path(directory)/'init'),
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        with patch.dict(os.environ):
            os.environ.pop('FLAGPROF_GROUP_ID', None)
            group_id, control = observation_group(rank, 2)
            ids = [None, None]
            dist.all_gather_object(ids, group_id, group=control)
            assert ids == [group_id, group_id]
            rows = []
            context = SimpleNamespace(closed=False, diagnostic=None, current_scope=lambda: None,
                                      probe_state=lambda: {}, writer=SimpleNamespace(emit=rows.append))
            manager = SimpleNamespace(context=context, config={}, state=dict(probes={}, group_id=group_id))
            probe = CommunicationProbe(manager, control)
            try:
                probe.install()
                tensor = torch.tensor([5 if rank == 0 else 0])
                op = dist.P2POp(dist.isend if rank == 0 else dist.irecv, tensor, 1-rank)
                for work in dist.batch_isend_irecv([op]):
                    work.wait()
                assert tensor.item() == 5
                assert [row['group'] for row in rows] == [group_id+'/p2p']
                assert rows[0]['ext']['communication']['batch_ops'] == [['isend'], ['irecv']][rank]
                assert 'FLAGPROF_GROUP_ID' not in os.environ
            finally:
                probe.uninstall()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'optional torch CPU runtime')
class CommunicationCoverageTests(unittest.TestCase):
    def test_spec_fields_match_live_torch_signatures(self):
        """Parameter names drift across torch releases; every payload, output
        and peer field must exist in the live signature the wrapper binds."""
        import inspect
        import torch.distributed as dist
        import torch.distributed._functional_collectives as functional
        from flagprof.probes.communication import (DIST_NAMES, FUNCTIONAL_NAMES, SPEC,
                                                   FUNCTIONAL_SPEC, PEER_FIELDS)
        for name in DIST_NAMES:
            params = inspect.signature(getattr(dist, name)).parameters
            payload, output = SPEC[name]
            # Each field is asserted on its own: an any() over them lets a
            # renamed parameter hide behind an unchanged sibling.
            for field in (payload, output):
                if field:
                    self.assertIn(field, params, (name, sorted(params)))
            for field in PEER_FIELDS.get(name, ()):
                self.assertIn(field, params, (name, sorted(params)))
        for name in FUNCTIONAL_NAMES:
            params = inspect.signature(getattr(functional, name)).parameters
            self.assertIn(FUNCTIONAL_SPEC[name][0], params, name)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'optional torch CPU runtime')
class CommunicationTests(unittest.TestCase):
    def test_dist_entries_record_identity_payload_and_restore(self):
        """Single-rank gloo pass over the rooted collective entries gloo
        implements: each call keeps its functional output, records the right
        op label and byte count, and the imperative entries are restored.
        all_to_all and the point-to-point family are NCCL-only here; the
        NCCL diagnostic car covers them with asserted byte accounting."""
        import torch
        import torch.distributed as dist
        from flagprof.probes.communication import CommunicationProbe
        rows = []
        context = SimpleNamespace(closed=False, diagnostic=None, current_scope=lambda: None,
            probe_state=lambda: dict(generation=dict(generation_step=0)),
            writer=SimpleNamespace(emit=rows.append))
        manager = SimpleNamespace(context=context, config={}, state=dict(probes={}))
        names = ('broadcast', 'reduce', 'scatter', 'gather', 'barrier')
        originals = tuple(getattr(dist, name) for name in names)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, FLAGPROF_GROUP_ID='test-group'):
            dist.init_process_group('gloo', init_method='file://'+str(Path(directory)/'rendezvous'), rank=0, world_size=1)
            probe = CommunicationProbe(manager, control=object())
            try:
                probe.install()
                x = torch.arange(4, dtype=torch.float32)
                dist.broadcast(x, 0)
                self.assertTrue(torch.equal(x, torch.arange(4, dtype=torch.float32)))
                dist.reduce(x, dst=0)
                self.assertTrue(torch.equal(x, torch.arange(4, dtype=torch.float32)))
                out = torch.zeros(4)
                dist.scatter(out, [torch.full((4,), 2.)], 0)
                self.assertTrue(torch.equal(out, torch.full((4,), 2.)))
                gathered = [torch.zeros(4)]
                dist.gather(torch.full((4,), 3.), gathered, 0)
                self.assertTrue(torch.equal(gathered[0], torch.full((4,), 3.)))
                dist.barrier()
            finally:
                probe.uninstall()
                dist.destroy_process_group()
        self.assertEqual([r['op'] for r in rows],
                         ['broadcast', 'reduce', 'scatter', 'gather', 'barrier'])
        self.assertEqual([r['bytes'] for r in rows], [16, 16, 16, 16, 0])
        self.assertEqual([r['ext']['communication']['peer'] for r in rows], [0, 0, 0, 0, None])
        self.assertEqual([r['group'] for r in rows], ['test-group/model'] * 5)
        self.assertEqual(tuple(getattr(dist, name) for name in names), originals)

    def test_two_rank_observed_batch_transfer_shares_group_id(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(observed_batch_transfer, args=(directory,), nprocs=2)

    def test_install_error_restores_public_api(self):
        import torch.distributed as dist
        import torch.distributed._functional_collectives as functional
        from flagprof.probes.communication import CommunicationProbe
        manager = SimpleNamespace(context=SimpleNamespace(), config={}, state=dict(probes={}))
        probe = CommunicationProbe(manager, control=object())
        before = (dist.isend, dist.P2POp.__new__, dist.P2POp.__init__, functional.all_reduce)
        wrap = probe._wrap
        def inject(name, native, functional=False):
            if functional:
                raise RuntimeError('injected wrapper failure')
            return wrap(name, native)
        with patch.object(probe, '_wrap', side_effect=inject), self.assertRaisesRegex(RuntimeError, 'injected'):
            probe.install()
        self.assertFalse(probe.installed)
        self.assertEqual((dist.isend, dist.P2POp.__new__, dist.P2POp.__init__, functional.all_reduce), before)

    def test_p2p_op_preserves_native_identity_and_restores_constructor(self):
        import torch
        import torch.distributed as dist
        from flagprof.probes.communication import CommunicationProbe
        context = SimpleNamespace(closed=False)
        manager = SimpleNamespace(context=context, config={}, state=dict(probes={}, group_id='job'))
        methods = (dist.P2POp.__new__, dist.P2POp.__init__)
        native_send, native_recv = dist.isend, dist.irecv
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group('gloo', init_method='file://'+str(Path(directory)/'rendezvous'),
                                    rank=0, world_size=1)
            probe = CommunicationProbe(manager, control=object())
            try:
                probe.install()
                tensor = torch.empty(1)
                for candidate, native in ((dist.isend, native_send), (dist.irecv, native_recv),
                                          (native_send, native_send), (native_recv, native_recv)):
                    self.assertIs(dist.P2POp(op=candidate, tensor=tensor, peer=0).op, native)
            finally:
                probe.uninstall()
                dist.destroy_process_group()
        self.assertEqual((dist.P2POp.__new__, dist.P2POp.__init__), methods)
        self.assertEqual((dist.isend, dist.irecv), (native_send, native_recv))

    def test_functional_group_identity_payload_and_restore(self):
        import torch
        import torch.distributed as dist
        import torch.distributed._functional_collectives as functional
        from flagprof.probes.communication import CommunicationProbe
        rows=[]
        context=SimpleNamespace(closed=False,diagnostic=None,current_scope=lambda:None,
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
