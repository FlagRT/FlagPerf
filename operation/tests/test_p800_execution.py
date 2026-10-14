# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""P800 execution gates: stdlib-only negative controls and evidence isolation."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime import cli
from runtime.evidence import read, write, seal, merge_runs
from vendors.kunlunxin import bootstrap, routing
from vendors.kunlunxin.p800 import Adapter, PROFILE

MARKERS = 'OPERATION_TARGET_BEGIN\nOPERATION_TARGET_END'


def probe_fixture():
    launch = {'actual_c_return':True, 'class':'XPULauncher',
              'driver':'/env/triton/backends/xpu/driver.py', 'kernel':'abs_func_kernel_rank_1',
              'implementation':[{'path':'/env/flag_gems/runtime/backend/_kunlunxin/ops/abs.py','function':'abs'}]}
    return {'device':'cuda:0','output_devices':['cuda:0'],
            'p800_route':{'protocol':'p800-xpu-launch-v1','case':'abs','oplib':'flaggems','backward':False,
                          'windows':{k:{'complete':True,'launches':[copy.deepcopy(launch)]} for k in ('first','cached')}}}


class RouteTests(unittest.TestCase):
    def test_target_ancestor_does_not_qualify_nested_helper_or_unknown_kernel(self):
        for kind in ('nested-copy', 'unknown', 'missing', 'wrong-target', 'nested-with-target-name'):
            p = probe_fixture()
            for window in p['p800_route']['windows'].values():
                row = window['launches'][0]
                if kind.startswith('nested'):
                    row['implementation'].insert(0, {'path':'/env/flag_gems/ops/copy.py','function':'copy'})
                if kind == 'nested-copy': row['kernel'] = 'copy_kernel'
                if kind == 'unknown': row['kernel'] = 'unknown'
                if kind == 'missing': row.pop('kernel')
                if kind == 'wrong-target': row['kernel'] = 'add_func_kernel_rank_1'
            self.assertEqual(Adapter().route(p, MARKERS)['status'], 'partial', kind)

    def test_reviewed_add_passes_and_unreviewed_mm_stays_partial(self):
        for case, kernel, expected in (('add', 'add_func_kernel_rank_1', 'passed'), ('mm', 'mm_kernel', 'partial')):
            p = probe_fixture();p['p800_route']['case'] = case
            for window in p['p800_route']['windows'].values():
                row = window['launches'][0];row['kernel'] = kernel
                row['implementation'][0].update(path='/env/flag_gems/ops/' + case + '.py', function=case)
            self.assertEqual(Adapter().route(p, MARKERS)['status'], expected)

    def test_actual_first_and_cached_launch(self):
        result = Adapter().route(probe_fixture(), MARKERS)
        self.assertEqual(result['status'],'passed');self.assertTrue(result['cached_launch_proven'])

    def test_missing_truncated_and_reversed_markers(self):
        for log in ('','OPERATION_TARGET_BEGIN',MARKERS+MARKERS,'OPERATION_TARGET_END\nOPERATION_TARGET_BEGIN'):
            self.assertEqual(Adapter().route(probe_fixture(),log)['status'],'partial')

    def test_helper_compile_only_and_unrelated_launcher_do_not_pass(self):
        for kind in ('helper','compile-only','wrong-driver','wrong-class','incomplete','register-only'):
            p = probe_fixture()
            for w in p['p800_route']['windows'].values():
                row = w['launches'][0]
                if kind == 'helper': row['implementation'][0].update(path='/env/flag_gems/ops/copy.py',function='copy')
                if kind == 'compile-only': row['actual_c_return'] = False;w['compilation_calls'] = 1
                if kind == 'register-only': w['launches'] = []
                if kind == 'wrong-driver': row['driver'] = '/env/triton/backends/ascend/driver.py'
                if kind == 'wrong-class': row['class'] = 'NPULauncher'
                if kind == 'incomplete': w['complete'] = False
            self.assertEqual(Adapter().route(p,MARKERS)['status'],'partial',kind)

    def test_cpu_output_fails_and_native_or_backward_is_partial(self):
        p=probe_fixture();p['output_devices']=['cpu']
        self.assertEqual(Adapter().route(p,MARKERS)['status'],'failed')
        for key,value in [('oplib','nativetorch'),('backward',True),('case','linear')]:
            p=probe_fixture();p['p800_route'][key]=value
            self.assertEqual(Adapter().route(p,MARKERS)['status'],'partial')

    def test_cached_python_call_alone_is_not_execution(self):
        p=probe_fixture();p['p800_route']['windows']['cached']['launches']=[]
        p['calls']=[{'path':'/env/triton/compiler/compiler.py','function':'runner'}]
        self.assertEqual(Adapter().route(p,MARKERS)['status'],'partial')

    def test_observer_requires_matching_c_call_return_and_ancestry(self):
        observer=routing.Observer({'case':'abs','oplib':'flaggems'},False)
        launcher=type('XPULauncher',(),{})();launcher.launch=object();launcher._fn_name='abs_func_kernel_rank_1'
        parent=SimpleNamespace(f_code=SimpleNamespace(co_filename='/env/flag_gems/ops/abs.py',co_name='abs'),f_back=None)
        frame=SimpleNamespace(f_code=SimpleNamespace(co_filename='/env'+routing.DRIVER,co_name='__call__'),
                              f_locals={'self':launcher},f_back=parent)
        for name in ('first','cached'):
            observer.begin(name)
            observer.event(frame,'call',None)
            observer.event(frame,'c_return',launcher.launch)  # no matching call
            self.assertEqual(observer.window['launches'],[])
            observer.event(frame,'c_call',launcher.launch)
            observer.event(frame,'c_return',launcher.launch);observer.end(0.1)
        p=probe_fixture();p['p800_route']=observer.record()
        self.assertEqual(Adapter().route(p,MARKERS)['status'],'passed')
        observer.begin('cached');observer.event(frame,'c_call',launcher.launch)
        observer.event(frame,'c_exception',launcher.launch);observer.end(0.1)
        p['p800_route']=observer.record()
        self.assertEqual(Adapter().route(p,MARKERS)['status'],'partial')

    def test_unrecognized_dependency_text_never_becomes_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(Adapter().diagnose('backend not registered; SPLIT_K; dependency missing',Path(tmp)))

    def test_known_compiler_failure_needs_exception_type_and_real_stack(self):
        error='out of resource: uni_sram PassManager::run failed'
        trace='''Traceback (most recent call last):
  File "/env/flag_gems/runtime/backend/_kunlunxin/ops/mm.py", line 182, in mm
  File "/env/triton/backends/xpu/compiler.py", line 367, in make_ttxir
triton.runtime.errors.OutOfResources: out of resource: uni_sram PassManager::run failed
'''
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write(root/'probe-error.json',{'type':'OutOfResources','error':error})
            (root/'probe.log').write_text(error)
            self.assertIsNone(Adapter().diagnose(error,root))
            (root/'probe.log').write_text(trace)
            self.assertIn('dependency-blocked',Adapter().diagnose(error,root))
            write(root/'probe-error.json',{'type':'RuntimeError','error':error})
            self.assertIsNone(Adapter().diagnose(error,root))


class ExecutionIdentityTests(unittest.TestCase):
    def test_versions_fail_before_runtime_import(self):
        lock=json.loads(PROFILE.read_text());packages={k:{'version':v} for k,v in lock['packages'].items()}
        bootstrap.validate_identity(lock,lock['python'],'/externally-managed/python',packages,'3.6.0',True)
        for kind in ('python','package','triton'):
            pp=copy.deepcopy(packages)
            if kind=='package':pp['torch']['version']='wrong'
            with self.assertRaises(RuntimeError):
                bootstrap.validate_identity(lock,'wrong' if kind=='python' else lock['python'],'/local/python',pp,
                                            'wrong' if kind=='triton' else '3.6.0',True)

    def test_local_visibility_preserved_and_cpu_worker_absolute_command(self):
        adapter=Adapter()
        self.assertNotIn('CUDA_VISIBLE_DEVICES',adapter.environment(7,'probe',True))
        self.assertNotIn('XPU_VISIBLE_DEVICES',adapter.environment(7,'reference',True))
        self.assertEqual(adapter.worker_command(True)[0],sys.executable)
        self.assertIn('-S',adapter.worker_command(False))
        self.assertTrue(adapter.worker_command(False)[0].startswith('/root/miniconda/'))

    def test_cpu_docker_has_no_devices_and_uses_explicit_entrypoint(self):
        args=cli.parser().parse_args(['run','--vendor','kunlunxin','--runtime-profile','p800-xpytorch29','--device-ids','4'])
        for phase in bootstrap.CPU_PHASES:
            with tempfile.TemporaryDirectory() as tmp, patch.object(cli.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'')) as run:
                pool=cli.Workers(Path(tmp),Adapter(),args,'sha256:test','test')
                pool.start({'device_id':4,'oplib':'nativetorch'},phase)
                cmd=run.call_args.args[0]
                self.assertNotIn('--device',cmd);self.assertIn('--entrypoint',cmd)
                self.assertIn('/operation/vendors/kunlunxin/bootstrap.py',cmd)
                self.assertIn('-S',cmd)

    def test_snapshot_has_configs_and_transitive_static_dependencies(self):
        adapter=Adapter()
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'')):
            root=Path(tmp);cli.snapshot(root,adapter)
            manifest=read(root/'source-manifest.json')
            for name in ('source/vendors/kunlunxin/bootstrap.py','source/benchmarks/abs/case_config.yaml',
                         'base/vendors/registry.py','base/vendors/ascend/provider.py',
                         'base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/stack.lock.yaml'):
                self.assertIn(name,manifest)
            for p in adapter.snapshot_paths():
                self.assertEqual((root/p.relative_to(ROOT.parent)).read_bytes(),p.read_bytes())

    def test_phase_identity_drift_fails_and_shared_performance_stays_partial(self):
        fields=('vendor_runtime','packages','case_sha256','input_sha256','registration','runtime_config')
        probe={k:{'fixture':1} for k in fields}
        for key in fields:
            item={'status':'passed','measurement':copy.deepcopy(probe),'runtime_config':{}}
            item['measurement'][key]={'fixture':2}
            Adapter().qualify_result(item,probe)
            self.assertEqual(item['status'],'failed');self.assertIn(key,item['phase_consistency']['mismatches'])
        item={'status':'passed','measurement':copy.deepcopy(probe),'runtime_config':{'sharing_policy':'shared-authorized'}}
        Adapter().qualify_result(item,probe)
        self.assertEqual(item['status'],'partial');self.assertEqual(item['phase_consistency']['status'],'passed')

    def test_failed_and_na_tasks_retain_profile_fingerprint(self):
        args=cli.parser().parse_args(['run','--vendor','kunlunxin','--runtime-profile','p800-xpytorch29',
                                     '--device-ids','0','--case','mm','--dtype','FP16','INT32'])
        adapter,plan=cli.resolve(args)
        self.assertEqual(len({t['runtime_config']['source_sha256'] for t in plan['tasks']}),1)
        self.assertEqual([t['applicable'] for t in plan['tasks']],[True,False])

    def test_replay_records_current_source_without_mutating_original(self):
        args=cli.parser().parse_args(['diagnose','--source-task','fixture','--allow-shared-device'])
        old={'case':'abs','device_id':0,'runtime_config':{'source_sha256':'old'}}
        new=copy.deepcopy(old);adapter=Adapter();adapter.prepare_replay(new,args)
        self.assertEqual(old['runtime_config']['source_sha256'],'old')
        self.assertEqual(new['runtime_config']['source_sha256'],adapter.source_fingerprint())
        self.assertEqual(new['runtime_config']['sharing_policy'],'shared-authorized')

    def test_source_fingerprint_is_independent_of_snapshot_directory_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);adapter=Adapter()
            with patch.object(cli.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'')):
                cli.snapshot(root,adapter)
            script='import sys;sys.path.insert(0,sys.argv[1]);from vendors.kunlunxin.p800 import Adapter;print(Adapter().source_fingerprint())'
            out=subprocess.check_output([sys.executable,'-B','-S','-c',script,str(root/'source')],text=True)
            self.assertEqual(out.strip(),adapter.source_fingerprint())


class AggregationTests(unittest.TestCase):
    def source(self, root, config=None, status='passed', **extra):
        root.mkdir()
        write(root/'summary.json',{'schema_version':2,'protocol':'operation-v2','status':status,
            'actual_image_id':'same','vendor':'kunlunxin','execution':'docker','preflight':{'host':'fixture'},
            'tasks':[{'case':'abs','dtype':'FP32','oplib':'flaggems','device_id':0,'status':status,
                      'runtime_config':config or {'physical_uuid':'a','profile_sha256':'p','source_sha256':'s'}}],**extra})
        seal(root)

    def test_three_batches_merge_and_different_fingerprints_stay_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)
            for i,status in enumerate(('failed','passed','not-run')):self.source(r/str(i),status=status)
            merge_runs([r/str(i) for i in range(3)],r/'merged')
            self.assertEqual(read(r/'merged/summary.json')['tasks'][0]['status'],'passed')
            sources=[]
            for key in ('physical_uuid','profile_sha256','source_sha256','protocol'):
                config={'physical_uuid':'a','profile_sha256':'p','source_sha256':'s'};config[key]='different'
                source=r/key;self.source(source,config);sources.append(source)
            merge_runs([r/'1',*sources],r/'isolated')
            self.assertEqual(len(read(r/'isolated/summary.json')['tasks']),5)

    def test_missing_host_and_orchestration_errors_cannot_aggregate_as_passed(self):
        for extra in ({'preflight':{}}, {'error':'fixture'}, {'lease_release_error':'fixture'},
                      {'cleanup_error':'fixture'}, {'postflight_error':'fixture'}):
            with tempfile.TemporaryDirectory() as tmp:
                r=Path(tmp);self.source(r/'source',**extra)
                with self.assertRaises(ValueError):merge_runs([r/'source'],r/'output')
                self.assertFalse((r/'output').exists())


if __name__ == '__main__': unittest.main()
