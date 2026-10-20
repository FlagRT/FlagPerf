# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Negative controls for ATen-to-device correlation; no Torch dependency."""
import copy
from pathlib import Path
import sys
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import nullcontext
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vendors.kunlunxin.dispatch import analyze_dispatch
from vendors.kunlunxin.routing import route


def fixture(op='aten::mm'):
    rows=[{'index':0,'op':op,'inputs':['cuda:0'],'outputs':['cuda:0'],
           'device_input':True,'device_output':True,'cpu_transfer':False}]
    events=[
        {'ph':'X','cat':'user_annotation','name':'p800_target_first_0','pid':1,'tid':2,'ts':100,'dur':50},
        {'ph':'X','cat':'user_annotation','name':'p800_dispatch_0','pid':1,'tid':2,'ts':105,'dur':20},
        {'ph':'X','cat':'cuda_runtime','name':'cudaLaunchKernel','pid':1,'tid':2,'ts':110,'dur':2,'args':{'correlation':9}},
        {'ph':'s','cat':'ac2g','pid':1,'tid':2,'ts':110,'id':9},
        {'ph':'X','cat':'kernel','name':'device_matmul','pid':0,'tid':3,'ts':10000,'dur':3,'args':{'correlation':9,'device':0}},
        {'ph':'f','cat':'ac2g','pid':0,'tid':3,'ts':10000,'id':9}]
    return events, rows


class DispatchTests(unittest.TestCase):
    def test_capture_export_failure_preserves_original_target_exception(self):
        from vendors.kunlunxin.dispatch import capture
        class OriginalError(RuntimeError):pass
        class Exporter:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def export_chrome_trace(self,path):raise ValueError('export failed')
        fake=SimpleNamespace(profiler=SimpleNamespace(profile=lambda **kw:Exporter(),
                 ProfilerActivity=SimpleNamespace(CPU=1,CUDA=2),record_function=lambda name:nullcontext()))
        modules={'torch':fake,'torch.utils._python_dispatch':SimpleNamespace(TorchDispatchMode=type('Mode',(),{'__enter__':lambda self:self,'__exit__':lambda self,*args:False})),
                 'runtime.worker':SimpleNamespace(tensors=lambda value:[])}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,modules):
            observer=SimpleNamespace(case='abs',dtype='FP32',dispatch={})
            with self.assertRaisesRegex(OriginalError,'target failed'):
                with capture(Path(tmp),SimpleNamespace(),observer,'first'):
                    raise OriginalError('target failed')
            import json
            saved=json.loads((Path(tmp)/'route-first-capture-error.json').read_text())
            self.assertTrue(saved['target_error_preserved'])

    def test_actual_computation_is_correlated_not_name_matched(self):
        events,rows=fixture()
        result=analyze_dispatch(events,rows)
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['computations'][0]['correlations'],[9])

    def test_copy_metadata_only_unrelated_kernel_cpu_transfer_and_malformed_fail_closed(self):
        for kind in ('copy-only','metadata-only','wrong-thread','missing-op','helper-only',
                     'wrong-device','cpu-transfer','no-device-output','ambiguous-marker','malformed','incomplete-row','no-flow'):
            ev,rows=fixture()
            if kind=='copy-only':rows[0]['op']='aten::copy_'
            if kind=='metadata-only':rows[0]['op']='aten::view'
            if kind=='wrong-thread':ev[1]['tid']=9
            if kind=='missing-op':ev.pop(1)
            if kind=='helper-only':
                rows.append(dict(rows[0],index=1,op='aten::addmm'))
                ev.append(dict(ev[1],name='p800_dispatch_1',ts=130,dur=10))
            if kind=='wrong-device':ev[4]['args']['device']=1
            if kind=='cpu-transfer':rows[0]['cpu_transfer']=True
            if kind=='no-device-output':rows[0]['device_output']=False
            if kind=='ambiguous-marker':ev.append(copy.deepcopy(ev[1]))
            if kind=='malformed':ev[1]['dur']='nan'
            if kind=='incomplete-row':rows[0]={}
            if kind=='no-flow':ev.pop()
            with self.subTest(kind=kind):
                self.assertIn(analyze_dispatch(ev,rows)['status'],('partial','failed'))

    def test_other_unobserved_computation_cannot_hide_beside_a_valid_one(self):
        ev,rows=fixture()
        extra=copy.deepcopy(ev[2:])
        for e in extra:
            if e.get('cat')=='cuda_runtime':e['ts']=135;e['args']['correlation']=10
            elif e.get('cat')=='ac2g':
                e['id']=10;e['ts']=135 if e['ph']=='s' else 10010
            else:e['ts']=10010;e['args']['correlation']=10
        ev.extend(extra)
        result=analyze_dispatch(ev,rows)
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['unattributed_computation_correlations'],[10])

    def test_each_device_computation_including_backward_needs_its_own_kernel(self):
        ev,rows=fixture()
        rows.append(dict(rows[0],index=1,op='aten::threshold_backward'))
        ev.append(dict(ev[1],name='p800_dispatch_1',ts=130,dur=10))
        self.assertEqual(analyze_dispatch(ev,rows)['status'],'partial')

    def test_opaque_linear_needs_own_cpp_event_and_fc_kernel_not_view_or_copy(self):
        for kind in ('valid','no-record','wrong-case','wrong-kernel','wrong-thread','duplicate','outside','nested-copy'):
            ev,rows=fixture('aten::view')
            ev[1].update(cat='cpu_op', name='LinearFunction')
            ev[4]['name']='void xblas_xpu3::fc_one_loop<Args>'
            if kind=='no-record':ev.pop(1)
            if kind=='wrong-kernel':ev[4]['name']='copy_kernel'
            if kind=='wrong-thread':ev[1]['tid']=9
            if kind=='duplicate':ev.append(copy.deepcopy(ev[1]))
            if kind=='outside':ev[1]['ts']=0
            if kind=='nested-copy':ev.append(dict(ev[1],name='aten::copy_',ts=109,dur=5))
            result=analyze_dispatch(ev,rows,case='mm' if kind=='wrong-case' else 'linear', dtype='FP32')
            self.assertEqual(result['status'],'passed' if kind=='valid' else 'partial',kind)

    def test_opaque_bf16_linear_requires_complete_reviewed_conversion_and_fc_chain(self):
        names=['void xpu3::findmax1d_cluster<bfloat16>(args)',
               'hydra_internal_kernel::cast_te(bfloat16 const* args)',
               'void xpu3::findmax1d_cluster<bfloat16>(args)',
               'hydra_internal_kernel::cast_te(bfloat16 const* args)',
               'void xblas_xpu3::fc_bf16_rn_one_loop<Args>']
        for kind in ('valid','missing-conversion','helper-only','wrong-dtype','wrong-order'):
            ev,rows=fixture('aten::view');ev=ev[:2];ev[1].update(cat='cpu_op',name='LinearFunction',dur=35)
            order=list(names)
            if kind=='missing-conversion':order.pop(0)
            if kind=='helper-only':order.pop()
            if kind=='wrong-order':order.reverse()
            for i,name in enumerate(order):
                extra=fixture()[0][2:]
                for e in extra:
                    if e['cat']=='cuda_runtime':e.update(ts=110+i*4);e['args']['correlation']=9+i
                    elif e['cat']=='ac2g':e.update(id=9+i,ts=110+i*4 if e['ph']=='s' else 10000+i*4)
                    else:e.update(ts=10000+i*4,name=name);e['args']['correlation']=9+i
                ev.extend(extra)
            result=analyze_dispatch(ev,rows,case='linear',dtype='FP16' if kind=='wrong-dtype' else 'BF16')
            self.assertEqual(result['status'],'passed' if kind=='valid' else 'partial',kind)

    def test_malformed_boolean_metadata_cannot_prove_execution(self):
        for key in ('device_input','device_output','cpu_transfer'):
            for value in (None, 'false', 0, 1):
                ev,rows=fixture();rows[0][key]=value
                self.assertEqual(analyze_dispatch(ev,rows)['status'],'partial')

    def test_partial_dispatch_cannot_use_legacy_whitelist_to_pass(self):
        from test_p800_execution import probe_fixture, MARKERS
        probe=probe_fixture()
        probe['p800_route']['dispatch']={'first':{'status':'partial'},'cached':{'status':'partial'}}
        result=route(probe,MARKERS)
        self.assertEqual(result['status'],'partial')
        self.assertFalse(result['cached_launch_proven'])

    def test_route_uses_both_current_windows_and_preserves_cpu_failure(self):
        ev,rows=fixture();valid=analyze_dispatch(ev,rows)
        probe={'device':'cuda:0','output_devices':['cuda:0'],'p800_route':{
            'protocol':'p800-xpu-launch-v1','oplib':'nativetorch','backward':True,
            'dispatch':{'first':valid,'cached':copy.deepcopy(valid)}}}
        log='OPERATION_TARGET_BEGIN\nOPERATION_TARGET_END'
        self.assertEqual(route(probe,log)['status'],'passed')
        probe['p800_route']['dispatch']['cached']['status']='partial'
        self.assertEqual(route(probe,log)['status'],'partial')
        probe['p800_route']['dispatch']['cached']['target_cpu_fallback']=True
        self.assertEqual(route(probe,log)['status'],'failed')

if __name__=='__main__':unittest.main()
