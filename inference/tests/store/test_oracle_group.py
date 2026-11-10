# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Paired TP2 precision reports retain local shards and count samples once."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from flagprof.store import create_run,Writer,finalize
from flagprof.report import generate_oracle


class GroupOracleTests(unittest.TestCase):
    def fixture(self,root,backend,delta=0):
        root.mkdir();ranks=[];gid=root.name
        for rank in range(2):
            run=create_run(root/('rank-'+str(rank)))
            identity=dict(model_files={'m':'same'},oracle_id='fixture',inputs_sha256='same',samples=['s'],batches=[[1]],
                dtype='float32',attention='eager',pooling='last',boundaries=['<root>#output'],probe_alg='p4v1',
                device='cpu:'+str(rank),software={},numerics={},flaggems='off',include=None,
                parallelism=dict(world_size=2,plan='auto'),rank=rank,comm_backend=backend)
            state=run/'attachments/f/state.json';state.parent.mkdir(parents=True)
            state.write_text(json.dumps(dict(id='f',oracle_identity=identity)))
            with Writer(run,rank=rank) as writer:
                for key,value in [('<root>#0#output#s',1.+delta),
                                  ('<root>.layers.0#0#op:aten::mul.Tensor/0/0#s',rank+2.+delta)]:
                    raw=struct.pack('<f',value);name=hashlib.sha256(key.encode()).hexdigest()+'.bin'
                    (run/'blobs'/name).write_bytes(raw);digest='sha256:'+hashlib.sha256(raw).hexdigest()
                    writer.emit(dict(t='tensor',call_key=key,role='out',boundary=key.split('#')[2],shape=[1],dtype='torch.float32',
                        stats={},chunks=[],probe=[],probe_alg='p4v1',digest=digest,
                        blob=dict(path='blobs/'+name,offset=0,nbytes=4,hash=digest,fmt='raw-le:float32:[1]')))
                writer.emit(dict(t='comm',op='allreduce',group=gid+'/model',group_seq=0,bytes=4,dur=1,
                    lib=backend,phase='enqueue',dtype='torch.float32',count=1,
                    ext={'communication':{'role':'model'}}))
            finalize(run)
            ranks.append(dict(rank=rank,group_id=gid,status='complete',run=str(run.relative_to(root))))
        path=root/'group.json';path.write_text(json.dumps(dict(schema='flagprof-group/1',group_id=gid,
            status='complete',world_size=2,backend=backend,ranks=ranks)))
        return path

    def test_local_shards_pair_by_rank_and_samples_count_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=self.fixture(root/'ref','hccl');b=self.fixture(root/'cmp','flagcx',.25)
            out=root/'report';generate_oracle(a,b,out,declare={'differences':['comm_backend']})
            value=json.loads((out/'result.json').read_text())
            self.assertEqual(value['samples'],1);self.assertEqual(len(value['ranks']),2)
            self.assertIn('逻辑输入样本：1',(out/'report.html').read_text())
            for rank in range(2):
                result=json.loads((out/('rank-'+str(rank))/'result.json').read_text())['results']
                self.assertEqual(result['oracle.identity_diff']['status'],'ok')
                self.assertEqual(result['oracle.layer']['data'][0]['worst']['max_abs']['value'],.25)
            with self.assertRaisesRegex(ValueError,'topology/backend'):
                manifest=json.loads(b.read_text());manifest['ranks'].reverse()
                manifest['ranks'][0]['rank']=0;manifest['ranks'][1]['rank']=1;b.write_text(json.dumps(manifest))
                generate_oracle(a,b,root/'swapped',declare={'differences':['comm_backend']})

    def test_missing_rank_and_undeclared_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=self.fixture(root/'ref','hccl');b=self.fixture(root/'cmp','flagcx')
            generate_oracle(a,b,root/'blocked')
            result=json.loads((root/'blocked/result.json').read_text())
            self.assertEqual(result['status'],'partial')
            value=json.loads(b.read_text());value['ranks'].pop();b.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'ranks'):generate_oracle(a,b,root/'missing')


if __name__=='__main__':unittest.main()
