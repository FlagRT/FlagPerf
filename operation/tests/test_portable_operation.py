# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Public CLI, lifecycle and real workload contracts."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime import catalog, cli
from runtime.evidence import write, seal, report
from vendors.ascend.adapter import Adapter


class PlanningTests(unittest.TestCase):
    def test_default_all_cases_single_route_and_native_types(self):
        tasks = catalog.expand()
        self.assertEqual(len(tasks), 52)
        self.assertEqual(len({t['case'] for t in tasks}), 52)
        self.assertEqual({t['oplib'] for t in tasks}, {'nativetorch'})
        self.assertTrue(all(t['applicable'] for t in tasks))
        self.assertEqual(next(t['dtype'] for t in tasks if t['case']=='all'), 'INT64')
        self.assertEqual(next(t['dtype'] for t in tasks if t['case']=='bitwise_and'), 'INT32')

    def test_literal_all_case_is_not_suite(self):
        self.assertEqual([t['case'] for t in catalog.expand(['all'])], ['all'])

    def test_matrix_inapplicable_entries_and_size_sweep(self):
        tasks = catalog.expand(['mm'], ['FP32','FP16'], 'both', sizes=['M=7,N=9,K=11','M=13,N=17,K=19'])
        self.assertEqual(len(tasks), 8)
        self.assertEqual({t['case_config']['M'] for t in tasks}, {7,13})
        self.assertFalse(catalog.expand(['bitwise_and'], ['FP32'])[0]['applicable'])
        with self.assertRaises(ValueError): catalog.expand(['mm'], sizes=['unknown=3'])

    def test_invalid_ids_and_peak_rejected(self):
        for value in ('1,1','-1','3-1'):
            with self.assertRaises(ValueError): cli.ids(value)
        args = cli.parser().parse_args(['run','--vendor','nvidia','--device-ids','0','--image','test','--spectflops','nan'])
        with self.assertRaises(ValueError): cli.resolve(args)

    def test_dry_run_has_no_host_or_runtime_side_effects(self):
        with patch.object(cli.subprocess,'run',side_effect=AssertionError('process')), \
             patch.object(cli.subprocess,'check_output',side_effect=AssertionError('process')), \
             patch.object(cli.Lease,'acquire',side_effect=AssertionError('lease')), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(['run','--vendor','nvidia','--device-ids','2','--image','test','--case','mm','--dry-run']),0)
        self.assertEqual(json.loads(out.getvalue())['tasks'][0]['worker_device'],0)

    def test_runtime_is_vendor_neutral(self):
        for path in (ROOT/'runtime').glob('*.py'):
            source = path.read_text()
            for token in ('ASCEND_RT_VISIBLE_DEVICES','flagos cpu_fallback','/dev/davinci','torch_fl','vendors.ascend'):
                self.assertNotIn(token, source, path.name)

    def test_all_cases_expose_shared_construction_without_implicit_device(self):
        for name in catalog.names():
            source = (ROOT/'benchmarks'/name/'main.py').read_text()
            tree = ast.parse(source)
            build = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_case')
            self.assertNotIn('.to(0)',ast.unparse(build))
            self.assertIsInstance(build.body[-1], ast.Return)


class LifecycleTests(unittest.TestCase):
    def test_schema_one_report_remains_readable_without_old_executor(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / 'summary.json', {'schema_version': 1, 'status': 'failed', 'paths': {}})
            write(root / 'artifacts.json', {'schema_version': 1, 'files': {
                'summary.json': hashlib.sha256((root / 'summary.json').read_bytes()).hexdigest(),
            }})
            report(root)
            first = (root / 'report.md').read_bytes()
            report(root)
            self.assertEqual(first, (root / 'report.md').read_bytes())
            write(root / 'summary.json', {})
            with self.assertRaises(ValueError):
                report(root)

    def test_lease_conflict_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = cli.Lease(Path(tmp),[1], 'first');second = cli.Lease(Path(tmp),[1],'second')
            first.acquire()
            with self.assertRaises(RuntimeError): second.acquire()
            first.release();second.acquire();second.release()

    def test_cleanup_failure_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = cli.parser().parse_args(['run','--vendor','ascend','--device-ids','14'])
            pool = cli.Workers(Path(tmp),Adapter(),args,'image','run')
            pool.active[(14,'nativetorch','probe')] = {'name':'exact-owned-container','proc':None,'control':Path(tmp)}
            result = cli.subprocess.CompletedProcess([],1,stdout='',stderr='daemon unavailable')
            with patch.object(cli.subprocess,'run',return_value=result) as run:
                with self.assertRaises(cli.CleanupError): pool.close()
                self.assertEqual(run.call_args.args[0],['docker','rm','-f','exact-owned-container'])
            self.assertTrue(pool.active)

    def test_route_fallback_only_in_target_interval(self):
        adapter = Adapter()
        probe = {'device':'flagos:0','output_devices':['flagos:0'],'calls':[]}
        log = '[flagos cpu_fallback] preparation\nOPERATION_TARGET_BEGIN\n[flagos dispatch] abs -> ascend\nOPERATION_TARGET_END'
        self.assertEqual(adapter.route(probe,log)['status'],'passed')
        self.assertEqual(adapter.route(probe,log.replace('abs -> ascend','[flagos cpu_fallback] abs'))['status'],'failed')
        self.assertEqual(adapter.route(probe,'')['status'],'partial')

    def test_report_rebuild_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root/'summary.json', {'schema_version':2,'status':'partial','tasks':[]})
            seal(root);report(root);one=(root/'report.md').read_bytes();report(root)
            self.assertEqual(one,(root/'report.md').read_bytes())
            (root/'summary.json').write_text('{}')
            with self.assertRaises(ValueError): report(root)


class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.torch = torch

    def test_every_case_cpu_construction_and_dtype(self):
        from runtime.worker import build, tensors
        for task in catalog.expand():
            task.update(vendor='ascend', seed=2026)
            fn, args, bp, count = build(task)
            self.assertTrue(tensors(args),task['case'])
            self.assertTrue(all(t.device.type=='cpu' for t in tensors(args)))
            self.assertGreater(count(1),0)
            if task['case'].startswith('bitwise_'):
                self.assertEqual(tensors(args)[0].dtype,self.torch.int32)

    def test_reference_rejects_wrong_gradients_and_nan(self):
        from runtime.worker import compare
        t=self.torch
        self.assertEqual(compare((t.ones(3), (t.zeros(3),)),(t.ones(3),(t.ones(3),)),'FP32')['status'],'failed')
        self.assertEqual(compare(t.tensor([float('nan')]),t.tensor([1.]),'FP32')['status'],'failed')

    def test_dropout_checks_scaling_and_gradient(self):
        from runtime.worker import dropout_check, gradient_weights
        t=self.torch
        x=t.ones(10000); mask=t.arange(10000)%5!=0;out=mask.float()/0.8
        self.assertEqual(dropout_check((out,(out * gradient_weights(out),)),(x,),True,'FP32')['status'],'passed')
        self.assertEqual(dropout_check((out,(t.zeros_like(out),)),(x,),True,'FP32')['status'],'failed')



class AdditionalContractTests(unittest.TestCase):
    def test_smoke_reduces_element_unit(self):
        daily = catalog.workload('abs','daily')
        smoke = catalog.workload('abs','smoke')
        self.assertLess(smoke['ELEMENT_UNIT'], daily['ELEMENT_UNIT'])
        self.assertLess(smoke['ITERS'], daily['ITERS'])

    def test_auxiliary_allocation_does_not_prove_target_route(self):
        log = 'OPERATION_TARGET_BEGIN\n[flagos dispatch] empty_like -> ascend\nOPERATION_TARGET_END'
        self.assertEqual(Adapter().route({'output_devices':['flagos:0'], 'calls':[]},log)['status'],'partial')

    def test_launch_timeout_retains_owned_container_for_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = cli.parser().parse_args(['run','--vendor','nvidia','--device-ids','0','--image','test'])
            from vendors import get_vendor
            pool = cli.Workers(Path(tmp),get_vendor('nvidia'),args,'image','test')
            with patch.object(cli.subprocess,'run',side_effect=cli.subprocess.TimeoutExpired('docker',60)):
                with self.assertRaises(cli.subprocess.TimeoutExpired):
                    pool.start({'device_id':0,'oplib':'nativetorch'},'probe')
            self.assertEqual(len(pool.active),1)

    def test_soft_target_exceedance_is_not_failure(self):
        from runtime.evidence import read
        from vendors import get_vendor
        with tempfile.TemporaryDirectory() as tmp:
            args = cli.parser().parse_args(['run','--vendor','nvidia','--execution','local','--device-ids','0','--case','abs','--result-root',tmp])
            adapter,plan = cli.resolve(args)
            def phase(root,task,mode):
                if mode == 'probe':
                    write(root/'probe.json',{});(root/'probe.log').write_text('')
                    write(root/'correctness.json',{'status':'passed'})
                elif mode == 'measure':
                    write(root/'measurement.json',{'correctness':{'status':'passed'}})
                    (root/'measure.log').write_text('')
            with patch.object(cli.Workers,'phase',side_effect=phase),patch.object(cli.Workers,'close'), \
                 patch.object(cli.Lease,'acquire'),patch.object(cli.Lease,'release'), \
                 patch.object(cli,'snapshot'),patch.object(adapter,'route',return_value={'status':'passed'}), \
                 patch.object(cli.time,'monotonic',side_effect=[0,0,125,126]),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.execute(args,adapter,plan),0)
            summary=read(next(Path(tmp).glob('*/summary.json')))
            self.assertEqual(summary['tasks'][0]['status'],'passed')
            self.assertTrue(summary['tasks'][0]['soft_target_exceeded'])


class DtypeConstructionTests(unittest.TestCase):
    def test_every_applicable_dtype_can_construct_cpu_inputs(self):
        from runtime.worker import build, tensors
        import torch
        torch.set_num_threads(1)
        for task in catalog.expand(dtype=[*catalog.FLOATS,*catalog.INTEGERS,'INT64'],profile='smoke'):
            if not task['applicable']: continue
            task.update(vendor='ascend',seed=2026)
            with self.subTest(case=task['case'],dtype=task['dtype']):
                fn,inputs,bp,count=build(task)
                self.assertTrue(tensors(inputs))
                if task['case'].startswith('bitwise_'):
                    self.assertEqual(str(tensors(inputs)[0].dtype),{'INT32':'torch.int32','INT16':'torch.int16','BOOL':'torch.bool'}[task['dtype']])


class BackwardProtocolTests(unittest.TestCase):
    def test_nonuniform_gradient_detects_softmax_backward(self):
        import torch
        from runtime.worker import invocation
        x=torch.tensor([[0.2,0.7,-0.1]],dtype=torch.float64,requires_grad=True)
        out,grad=invocation(torch.nn.Softmax(dim=1),(x,),True,nonzero=True)
        self.assertGreater(float(grad[0].abs().max()),0.01)

    def test_timing_reuses_zero_gradient_per_case(self):
        import torch
        from runtime.worker import invocation
        x=torch.ones(3,requires_grad=True);cache=[]
        with patch.object(torch,'zeros_like',wraps=torch.zeros_like) as zeros:
            for _ in range(3): invocation(torch.sigmoid,(x,),True,grad_cache=cache)
            self.assertEqual(zeros.call_count,1)

    def test_dropout_zero_inputs_do_not_imply_dropped_mask(self):
        import torch
        from runtime.worker import dropout_check,gradient_weights
        x=torch.ones(10000);x[0]=0
        keep=torch.arange(10000)%5!=0;keep[0]=True
        out=x*keep/0.8;grad=gradient_weights(out)*keep/0.8
        self.assertEqual(dropout_check((out,(grad,)),(x,),True,'FP32')['status'],'passed')

    def test_kunlunxin_shape_override_membership_is_portable(self):
        from runtime.worker import build
        for task in catalog.expand(profile='smoke'):
            task.update(vendor='kunlunxin',seed=2026)
            build(task)


class AggregateTests(unittest.TestCase):
    def source(self,root,status,image='same-image',size=1,seed=2026):
        root.mkdir()
        write(root/'summary.json',{'schema_version':2,'status':status,'actual_image_id':image,
              'vendor':'nvidia','execution':'docker','preflight':{'host':'test-host'},'tasks':[
              {'case':'abs','dtype':'FP32','oplib':'nativetorch','device_id':0,'seed':seed,'status':status,'case_config':{'Melements':size}}]})
        seal(root)

    def test_distinct_seeds_remain_distinct_workloads(self):
        from runtime.evidence import merge_runs,read
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.source(root/'a','passed',seed=1);self.source(root/'b','passed',seed=2)
            merge_runs([root/'a',root/'b'],root/'seeds')
            self.assertEqual(len(read(root/'seeds/summary.json')['tasks']),2)

    def test_latest_attempt_wins_but_unstarted_does_not_erase(self):
        from runtime.evidence import merge_runs,read
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,status in [('old','failed'),('new','passed'),('unstarted','not-run')]:
                self.source(root/name,status)
            merge_runs([root/'old',root/'new',root/'unstarted'],root/'merged')
            self.assertEqual(read(root/'merged/summary.json')['tasks'][0]['status'],'passed')
            (root/'new/summary.json').write_text('{}')
            with self.assertRaises(ValueError): report(root/'merged')

    def test_different_sizes_stay_separate_and_image_mismatch_rejected(self):
        from runtime.evidence import merge_runs,read
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.source(root/'a','passed',size=1);self.source(root/'b','passed',size=2)
            merge_runs([root/'a',root/'b'],root/'sizes')
            self.assertEqual(len(read(root/'sizes/summary.json')['tasks']),2)
            self.source(root/'other','passed',image='different-image')
            with self.assertRaises(ValueError):merge_runs([root/'a',root/'other'],root/'bad')


if __name__ == '__main__': unittest.main()
