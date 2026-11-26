# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""CPU execution controls for the four pure overloads in the fixed torch image."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
try:
    import torch
except ImportError:
    raise unittest.SkipTest('torch execution environment required')
from flagprof.probes.argtree import describe_args
from flagprof.replay import prepare,run
from flagprof.runner.fixed import numerics,software
from flagprof.store import Writer,create_run,finalize
from flagprof.store.sig import compute_sid


class ReplayExecutionTests(unittest.TestCase):
    def test_overloads_dtypes_defaults_and_noncontiguous_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for index,(operator,tail,kwargs,dtype) in enumerate([
                (torch.ops.aten.mean.dim,([-1],True),{'dtype':torch.float64},torch.float32),
                (torch.ops.aten.softmax.int,(-1,),{},torch.bfloat16),
                (torch.ops.aten.rsqrt.default,(),{},torch.float32),
                (torch.ops.aten.pow.Tensor_Scalar,(2,),{},torch.bfloat16)]):
                stores=[];schema=operator._schema;key='<root>.layer#0#op:'+schema.name+'/'+str(index)+'#s'
                for side,delta in [('ref',0.),('cmp',.5)]:
                    value=(torch.arange(6,dtype=torch.float32)+1+delta).reshape(3,2).T.to(dtype)
                    args=(value,*tail);tree=describe_args(args,kwargs);sid=compute_sid(schema.name,schema.overload_name,'in',tree)
                    config=[dict(name=a.name,value=describe_args([args[i] if i<len(args) else kwargs.get(a.name,a.default_value)])) for i,a in enumerate(schema.arguments)]
                    path=create_run(root/(str(index)+side));stores.append(path)
                    identity=dict(model_files={},oracle_id='fixture',inputs_sha256='fixture',samples=['s'],batches=[],dtype=str(dtype),
                        attention='eager',pooling='fixture',boundaries=['<root>.layer'],probe_alg='p4v1',device='cpu',
                        software=software(),numerics=numerics('cpu','default'),flaggems='off',include=None)
                    from flagprof.components.compiler import identity as compiler_identity
                    identity['compiler']=compiler_identity()
                    state=path/'attachments/f/state.json';state.parent.mkdir(parents=True)
                    state.write_text(json.dumps(dict(id='f',probes={},oracle_identity=identity)))
                    with Writer(path) as writer:
                        writer.register_signature(dict(name=schema.name,overload=schema.overload_name,role='in',tree=tree))
                        for name,tensor in [('input',value),('output',operator(*args,**kwargs))]:
                            raw=tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes();digest='sha256:'+hashlib.sha256(raw).hexdigest()
                            blob=path/'blobs'/(name+'.bin');blob.write_bytes(raw)
                            input_key='<root>.layer#0#in#s'
                            event=dict(t='tensor',role='in' if name=='input' else 'out',call_key=input_key if name=='input' else key,
                                boundary=name,shape=list(tensor.shape),dtype=str(tensor.dtype),stats={},chunks=[],probe=[],probe_alg='p4v1',digest=digest,
                                blob=dict(path=str(blob.relative_to(path)),offset=0,nbytes=len(raw),hash=digest,fmt='raw-le:'+str(tensor.dtype).removeprefix('torch.')+':'+str(list(tensor.shape))))
                            if name=='output':event.update(sid=sid,src=[input_key],ext={'call_config':config})
                            self.assertTrue(writer.emit(event))
                    finalize(path)
                packet=root/('packet'+str(index));prepare(*stores,key,packet)
                for side in ('ref','cmp'):
                    result=run(packet,side=side,device='cpu',store=root/('replayed'+str(index)+side))
                    self.assertEqual(result['status'],'complete')
                    self.assertTrue(result['cases'][side]['self_reproduction']['digest_equal'])
            # Reuse the last operator's real Stores through the public workflow.
            source=root/'replay.json'
            declare={'differences': [], 'note': 'same device'}
            source.write_text(json.dumps(dict(schema_version=1, mode='replay', replay=dict(
                ref={'run':str(stores[0].relative_to(root))},cmp={'run':str(stores[1].relative_to(root))},
                call_key=key,declare=declare))))
            job=root/'workflow'
            command=[sys.executable, '-B', '-m', 'flagprof', 'run', str(source), '--out', str(job), '--json']
            subprocess.run([*command, '--check'], check=True, capture_output=True, timeout=90)
            self.assertFalse(job.exists())
            result=subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary=json.loads(result.stdout)
            self.assertEqual([step['id'] for step in summary['steps']], ['prepare','run.ref','run.cmp','oracle'])
            argv=summary['steps'][-1]['argv']
            for side in ('ref','cmp'):
                path=Path(argv[argv.index('--replay-'+side+'-run')+1])
                self.assertEqual(path.parent, job/side/'store')
                self.assertTrue((path/'replay-result.json').is_file())
            report=json.loads((job/'report/result.json').read_text())
            self.assertEqual(report['declaration'], declare)
            self.assertEqual(report['results']['oracle.replay']['status'], 'ok')
            manifest=packet/'replay.json';value=json.loads(manifest.read_text())
            compiler=value['sides']['ref']['identity']['compiler']
            failure='compiler source mismatch'
            if compiler['status']!='available':
                compiler.update(status='available',provider='vendor')
                failure='install vendor Triton|compiler source mismatch'
            compiler['source_sha256']='changed'
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,failure):
                run(packet,side='ref',device='cpu',store=root/'wrong-compiler')
            for rank,path in enumerate(stores):
                state_path=path/'attachments/f/state.json';state=json.loads(state_path.read_text())
                state['oracle_identity'].update(rank=rank,parallelism={'world_size':2,'plan':'auto'})
                state_path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError,'source identity mismatch'):
                prepare(*stores,key,root/'cross-rank-packet')


if __name__=='__main__':unittest.main()
