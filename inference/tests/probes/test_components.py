# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Real tiny Qwen TP2 CPU forward and model/block capture consistency."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class TinyQwenTPTests(unittest.TestCase):
    def test_direct_attachment_failure_does_not_enter_business_body(self):
        from flagprof import attach
        from flagprof.store import create_run
        with tempfile.TemporaryDirectory() as tmp:
            run=create_run(tmp)
            with patch('flagprof.components.flaggems.FlagGems.enable',side_effect=RuntimeError('registration failure')):
                with self.assertRaisesRegex(RuntimeError,'attachment did not start'):
                    with attach(run,'ascend',preset='accuracy',config={'flaggems':'on','flaggems_include':['add']}):
                        self.fail('failed component must not execute business body')

    def test_failed_explicit_selection_never_executes_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);marker=root/'target-executed'
            child=subprocess.run([sys.executable,'-B',*(['-S'] if sys.flags.no_site else []),'-m','flagprof','record','--store',str(root/'store'),
                '--vendor','ascend','--preset','accuracy','--flagtree','off','--vendor-compiler',str(root/'missing'),
                '--',sys.executable,'-c','from pathlib import Path;Path('+repr(str(marker))+').touch()'],
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
            self.assertNotEqual(child.returncode,0,child.stdout)
            self.assertFalse(marker.exists(),child.stdout)

    def test_failed_component_registration_never_executes_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);marker=root/'target-executed';include=root/'include.json'
            include.write_text('["missing_function_for_failure_control"]')
            child=subprocess.run([sys.executable,'-B',*(['-S'] if sys.flags.no_site else []),'-m','flagprof','record','--store',str(root/'store'),
                '--vendor','ascend','--preset','accuracy','--flaggems','on','--flaggems-include',str(include),
                '--',sys.executable,'-c','from pathlib import Path;Path('+repr(str(marker))+').touch()'],
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
            self.assertNotEqual(child.returncode,0,child.stdout)
            self.assertFalse(marker.exists(),child.stdout)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                         'optional torch and Transformers CPU runtime')
    def test_tp2_embedding_runner_and_replicated_boundaries(self):
        import torch
        from transformers import Qwen3Config,Qwen3Model
        from flagprof.runner.fixed import sha
        from flagprof.store import StoreView
        torch.set_num_threads(1);torch.manual_seed(531)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model_path=root/'model'
            model=Qwen3Model(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
                num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,head_dim=8,
                attn_implementation='eager')).eval()
            model.save_pretrained(model_path)
            prepared=root/'oracle/prepared';prepared.mkdir(parents=True)
            inputs={'input_ids':torch.tensor([[1,2,3]]),'attention_mask':torch.ones(1,3,dtype=torch.long)}
            torch.save([{'inputs':inputs}],prepared/'inputs.pt')
            (prepared/'forward-inputs.json').write_text(json.dumps({'batches':[{'sample_ids':['s']}]}))
            (prepared/'samples.json').write_text('[]')
            manifest={'oracle_id':'tiny-qwen','files':{'prepared/'+p.name:{'sha256':sha(p)} for p in prepared.iterdir()},
                'contract':{'model_files':{p.name:sha(p) for p in model_path.iterdir()},'inputs':{'batch_size':1}}}
            (root/'oracle/manifest.json').write_text(json.dumps(manifest))
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            children=[]
            try:
                for rank in range(2):
                    env=dict(os.environ,MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),RANK=str(rank),
                        LOCAL_RANK=str(rank),WORLD_SIZE='2',FLAGPROF_DEVICE='cpu',FLAGPROF_COMM_BACKEND='gloo')
                    children.append(subprocess.Popen([sys.executable,'-B','-m','tests.probes.tp_embedding_target',str(root)],env=env,
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True))
                for child in children:
                    output,_=child.communicate(timeout=90)
                    self.assertEqual(child.returncode,0,output)
            finally:
                for child in children:
                    if child.poll() is None:child.kill();child.wait()
            records=[]
            for rank in range(2):
                run=next((root/('rank-'+str(rank))).glob('*/manifest.json')).parent
                with StoreView(run) as view:records.append({r.record['call_key']:r.record['digest'] for r in view.iter_events(types=('tensor',))})
            self.assertEqual(records[0],records[1]);self.assertEqual(len(records[0]),5)
            a=json.loads((root/'outputs/rank-0/runner.json').read_text())
            b=json.loads((root/'outputs/rank-1/runner.json').read_text())
            self.assertEqual(a['runs'][0]['outputs'][0]['sha256'],b['runs'][0]['outputs'][0]['sha256'])


if __name__=='__main__':unittest.main()
