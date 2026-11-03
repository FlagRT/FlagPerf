# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""P800 identity/format negatives and Linux process primitive regression."""
import json
import os
from pathlib import Path
import select
import mmap
import signal
import subprocess
import sys
import unittest
from unittest.mock import patch

from flagprof.registry import Registry
from flagprof.vendors.kunlunxin.format import Taxonomy
from flagprof.vendors.kunlunxin.evidence import derive
from flagprof.vendors.contracts import evaluate_conditions
from flagprof.store import _pidfd


class P800Tests(unittest.TestCase):
    @unittest.skipUnless(sys.platform=='linux','Linux persisted Store')
    def test_window_projection_is_persisted_and_tampering_is_rejected(self):
        import tempfile
        from flagprof.offline.pipeline import prepare, import_prepared
        from flagprof.store.manifest import create_run
        from flagprof.store._offline_writer import offline_writer
        from flagprof.store import StoreView
        root=Path(__file__).parent/'fixtures/p800'
        with tempfile.TemporaryDirectory() as work:
            work=Path(work)
            prepared=prepare(root/'silu-fragment.json',work/'prepared',vendor='kunlunxin',
                analysis_input=dict(capture_normal_close=False,
                    capture_scope='scheduled_window_not_complete_request',
                    clock=dict(clock_valid=False),window_identity=dict(window_kind='scheduled_window')))
            run=create_run(work/'store')
            with offline_writer(run) as writer:
                imported=import_prepared(root/'silu-fragment.json',writer,prepared,vendor='kunlunxin')
            self.assertEqual(imported['status'],'complete')
            with StoreView(run) as view:
                list(view.iter_native_relations())
                context=view.device_analysis_context()
                self.assertEqual(context['status'],'available')
                self.assertFalse(context['capture_normal_close'])
                rows=list(view.iter_device_events(projection='analysis'))
                self.assertEqual(len(rows),1)
                self.assertEqual(rows[0]['category'],'compute')
                path=next((Path(run)/'derived').glob('*/analysis_intervals.jsonl'))
                path.write_bytes(path.read_bytes()+b'\n')
                with self.assertRaises(ValueError):list(view.iter_device_events(projection='analysis'))

    def test_windowed_cli_forwards_requested_timeout(self):
        from flagprof.cli import main
        from unittest.mock import Mock
        target=Mock(return_value=124)
        with patch('flagprof.record.record',target):
            result=main(['record','--vendor','kunlunxin','--capture-mode','windowed',
                         '--timeout','1.25','--store','unused','--',sys.executable,'target.py'])
        self.assertEqual(result,124)
        self.assertEqual(target.call_args.kwargs['timeout'],1.25)

    def test_real_exporter_fragment_keeps_device_and_missing_project_link(self):
        import hashlib
        from flagprof.vendors.prepare import prepare_import
        root=Path(__file__).parent/'fixtures/p800'
        raw=(root/'silu-fragment.json').read_bytes()
        provenance=json.loads((root/'provenance.json').read_text())
        self.assertEqual(hashlib.sha256(raw).hexdigest(),provenance['fragment_sha256'])
        result=prepare_import(raw,vendor='kunlunxin')
        self.assertEqual(len(result.events),1)
        event=result.events[0]
        self.assertEqual(event['name'],'silu_forward_kernel_rank_1')
        self.assertEqual((event['dev'],event['stream']),(0,0))
        self.assertEqual(event['link']['status'],'unlinked')
        self.assertFalse(any(row.get('state')=='owned' for row in result.native))

    def test_host_identity_requires_exact_uuid_and_sourced_fields(self):
        import tempfile
        from types import SimpleNamespace
        from flagprof.vendors.kunlunxin.session import hardware_identity
        torch=SimpleNamespace(cuda=SimpleNamespace(current_device=lambda:0,
            get_device_properties=lambda index:SimpleNamespace(uuid='GPU-observed'),
            get_device_name=lambda index:'GPU'))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'identity.json'
            host=dict(source='xpu-smi',devices=[dict(model='P800',uuid='observed',pci_bdf='0000:01:00.0',
                driver='5.0.21.47',xpu_smi_runtime='10.2',host_device_node='/dev/xpu0')])
            path.write_text(json.dumps(host))
            with patch.dict(sys.modules,torch=torch),patch.dict(os.environ,FLAGPROF_P800_IDENTITY=str(path)):
                self.assertEqual(hardware_identity()['model'],'P800')
                host['devices'][0]['uuid']='wrong';path.write_text(json.dumps(host))
                with self.assertRaisesRegex(ValueError,'UUID mismatch'):hardware_identity()
                host['devices'][0]['uuid']='observed';host['devices'][0].pop('driver');path.write_text(json.dumps(host))
                with self.assertRaisesRegex(ValueError,'missing sourced P800 driver'):hardware_identity()

    def test_pinned_collective_names_do_not_include_arithmetic_reductions(self):
        from flagprof.vendors.kunlunxin.analysis import activity
        for name in ['void kl3_all_reduce<float>(char*)','void kl3_all_reduce<bfloat16>(char*)']:
            row=activity(dict(ph='X',cat='kernel',name=name,args={'device':0,'stream':9}),{},'')
            self.assertEqual(row['category'],'comm')
            self.assertEqual(row['collective'],'allreduce')
        for name in ['void xpukernel_xpu3::reduce_mt<1,float>(float*)','gather_2d','nccl_unknown','void kl3_all_reduce<double>(char*)']:
            self.assertEqual(activity(dict(ph='X',cat='kernel',name=name,args={}),{},'')['category'],'compute')

    def test_clock_mapping_requires_pinned_exporter_and_unique_base(self):
        from flagprof.vendors.kunlunxin.analysis import clock
        capture=dict(trace=dict(sha256='0'*64),metadata=dict(identities=dict(collector=dict(
            facts=dict(collector=dict(version='2.9.0+cu129')),
            evidence=dict(observations=[dict(trace_metadata=dict(baseTimeNanoseconds=123))])))))
        result=clock('.',capture)
        self.assertEqual(result['project_transforms'][0]['base_time_ns'],123)
        self.assertFalse(result['clock_valid'])
        collector=capture['metadata']['identities']['collector']
        collector['facts']['collector']['version']='unknown'
        self.assertEqual(clock('.',capture)['project_transforms'],[])
        collector['facts']['collector']['version']='2.9.0+cu129'
        collector['evidence']['observations'].append(dict(trace_metadata=dict(baseTimeNanoseconds=456)))
        self.assertEqual(clock('.',capture)['project_transforms'],[])

    def test_pinned_tp_plan_preserves_vocab_shard_and_full_output(self):
        from types import SimpleNamespace
        from flagprof.runner.parallel import onerec_plan
        plan={'model.layers.*.self_attn.q_proj':'colwise','lm_head':'colwise_gather_output'}
        with patch.dict(sys.modules,transformers=SimpleNamespace(__version__='4.57.1')):
            result=onerec_plan(plan)
        self.assertEqual(result,dict(plan,lm_head='colwise_rep'))
        self.assertEqual(plan['lm_head'],'colwise_gather_output')
        with patch.dict(sys.modules,transformers=SimpleNamespace(__version__='other')):
            self.assertEqual(onerec_plan(plan),plan)

    def test_missing_launch_keys_do_not_gain_proximity_links(self):
        from flagprof.vendors.prepare import prepare_import
        rows=[dict(ph='X',cat='cpu_op',name='aten::mm',pid=20,tid=21,ts=0,dur=20,args={'External id':7}),
              dict(ph='X',cat='cuda_runtime',name='cudaLaunchKernel',pid=20,tid=21,ts=2,dur=1,args={'External id':7,'correlation':3}),
              dict(ph='X',cat='kernel',name='test_kernel',pid=0,tid=1,ts=4,dur=5,args={'device':0,'stream':1})]
        result=prepare_import(json.dumps(dict(traceEvents=rows)).encode(),vendor='kunlunxin')
        self.assertEqual(len(result.events),1)
        self.assertNotEqual(result.native[0]['state'],'owned')
        self.assertEqual(result.events[0]['link']['status'],'unlinked')

    def test_p800_prerequisites_require_exact_device_and_environment(self):
        from flagprof.vendors.kunlunxin.session import KunlunxinProfiler
        from types import SimpleNamespace
        backend=KunlunxinProfiler()
        expected=[dict(name='XPU_ENABLE_PROFILER_TRACING',value='1'),dict(name='XPU_CUPTI_ENABLE_DEVICE',value='1')]
        with patch.dict(sys.modules,torch=SimpleNamespace(cuda=SimpleNamespace(current_device=lambda:1))):
            with patch.dict(os.environ,{'XPU_ENABLE_PROFILER_TRACING':'1','XPU_CUPTI_ENABLE_DEVICE':'1'}):
                self.assertTrue(backend.approve_prerequisites(expected))
                self.assertFalse(backend.approve_prerequisites(expected+[dict(name='OTHER',value='1')]))
                self.assertFalse(backend.approve_prerequisites(expected[:1]))
            with patch.dict(os.environ,{'XPU_ENABLE_PROFILER_TRACING':'1','XPU_CUPTI_ENABLE_DEVICE':'0'}):
                self.assertFalse(backend.approve_prerequisites(expected))

    @unittest.skipUnless(_pidfd.supported(),'Linux procfs')
    def test_unreaped_child_is_not_an_active_writer(self):
        from flagprof.store.manifest import proc_key, process_alive
        import time
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.1)'])
        try:
            key=proc_key(child.pid)
            self.assertTrue(process_alive(key))
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                fields=Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()
                if fields[0]=='Z':break
                time.sleep(.01)
            self.assertEqual(fields[0],'Z')
            self.assertFalse(process_alive(key))
            self.assertFalse(process_alive(dict(key,proc_start_ticks=key['proc_start_ticks']+1)))
        finally:child.wait(timeout=5)

    @unittest.skipUnless(_pidfd.supported(),'Linux memory files')
    def test_raw_memfd_is_anonymous_shared_memory(self):
        from types import SimpleNamespace
        with patch.object(_pidfd,'os',SimpleNamespace(fsencode=os.fsencode,strerror=os.strerror)):
            fd=_pidfd.create_memfd('flagprof-test')
        try:
            os.ftruncate(fd,4096)
            with mmap.mmap(fd,4096) as memory:
                memory[:4]=b'test'
                self.assertEqual(os.pread(fd,4,0),b'test')
            self.assertFalse(os.get_inheritable(fd))
        finally:os.close(fd)

    @unittest.skipUnless(_pidfd.supported(),'Linux process groups')
    def test_owned_timeout_stops_target(self):
        from flagprof.offline.process import run_owned
        import time
        start=time.monotonic()
        result=run_owned([sys.executable,'-c','import time;time.sleep(30)'],timeout=.2)
        self.assertEqual(result.returncode,124)
        self.assertTrue(result.timed_out)
        self.assertLess(time.monotonic()-start,8)
        with self.assertRaises(ProcessLookupError):os.kill(result.pid,0)

    def test_contract_is_distinct_and_rejects_nvidia_and_wrong_stack(self):
        p800=Taxonomy()
        from flagprof.vendors.nvidia.format import Taxonomy as Nvidia
        self.assertNotEqual(p800.identity()['sha256'],Nvidia().identity()['sha256'])
        self.assertEqual(Registry().describe('kunlunxin')['format_vendor'],'kunlunxin')
        facts=dict(hardware=dict(model='NVIDIA A100-SXM4-80GB',driver='5.0.21.47',xpu_smi_runtime='10.2'),
                   collector=dict(name='torch.profiler',version='2.8.0+cu126'),actual=dict(record_shapes=True),exporter=dict(schema_version=1))
        failures={x['id'] for x in evaluate_conditions(p800.contract['conditions'],facts) if x['state']=='fail'}
        self.assertEqual(failures,{'hardware.model','collector.version'})

    def test_missing_exporter_flags_are_unknown(self):
        config={};exporter={};evidence=dict(observations=[])
        derive(dict(schemaVersion=1,traceEvents=[],record_shapes=1),dict(name='torch.profiler',version='2.9.0+cu129'),
               dict(path='trace.json'),{},config,exporter,evidence)
        self.assertIs(config['actual']['values']['record_shapes'],True)
        self.assertIsNone(config['actual']['values']['profile_memory'])

    @unittest.skipUnless(_pidfd.supported(),'Linux pidfds')
    def test_raw_pidfd_wrapper_signals_exact_child(self):
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
        fd=None
        try:
            # Exercise the compatibility path even on CPython exporting wrappers.
            with patch.object(_pidfd,'os',__import__('types').SimpleNamespace(strerror=os.strerror)):
                fd=_pidfd.open_pidfd(child.pid)
            with patch.object(_pidfd,'signal',__import__('types').SimpleNamespace()):
                _pidfd.send_signal(fd,signal.SIGTERM)
            child.wait(timeout=5)
            self.assertTrue(select.select([fd],[],[],1)[0])
            self.assertEqual(child.returncode,-signal.SIGTERM)
        finally:
            if child.poll() is None:child.kill();child.wait()
            if fd is not None:os.close(fd)


if __name__=='__main__':unittest.main()
