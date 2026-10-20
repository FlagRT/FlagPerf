# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Strict P800 CUPTI attribution; CPU-only and malformed traces never pass."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vendors.kunlunxin import profiling
from vendors.kunlunxin.p800 import Adapter
from runtime import cli
from runtime.evidence import write, read


def events():
    result=[]
    for i in range(5):
        # Separate host and device clock values deliberately: containment is not attribution.
        host=1000+i*100;dev=100000+i*20;corr=10+i
        result += [
            {'ph':'X','cat':'user_annotation','name':'operation_target_'+str(i),'pid':1,'tid':2,'ts':host,'dur':90},
            {'ph':'X','cat':'cuda_runtime','name':'cudaLaunchKernel','pid':1,'tid':2,'ts':host+10,'dur':4,'args':{'correlation':corr}},
            {'ph':'s','cat':'ac2g','id':corr,'pid':1,'tid':2,'ts':host+10},
            {'ph':'X','cat':'kernel','name':'target_kernel','pid':0,'tid':7,'ts':dev,'dur':2,'args':{'device':0,'correlation':corr,'stream':7,'est. achieved occupancy %':-391}},
            {'ph':'f','cat':'ac2g','id':corr,'pid':0,'tid':7,'ts':dev},
        ]
    return result


class P800ProfilerTests(unittest.TestCase):
    def test_explicit_correlation_and_flows_join_distinct_clocks(self):
        result=profiling.analyze_events(events())
        self.assertEqual(result['status'],'measured')
        self.assertEqual(result['kernel_count'],5)
        self.assertEqual([x['device_union_us'] for x in result['calls']],[2.0]*5)
        self.assertNotIn('est. achieved occupancy %',str(result['kernels']))

    def test_cpu_only_trace_does_not_prove_device_execution(self):
        ev=[e for e in events() if e.get('cat') not in ('kernel','ac2g')]
        self.assertEqual(profiling.analyze_events(ev)['status'],'partial')

    def test_missing_kernel_reports_unmatched_host_launch_without_weakening_gate(self):
        ev=events()
        ev=[e for e in ev if not (e.get('cat')=='kernel' and e.get('args',{}).get('correlation')==12)]
        result=profiling.analyze_events(ev)
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['missing_calls'],[2])
        self.assertEqual(result['missing_correlations'],[12])
        self.assertEqual(result['launches_without_device_kernel'],[{'call':2,'correlation':12}])

    def test_native_counter_stub_success_without_output_write_is_not_supported(self):
        probe={'cuptiDeviceGetNumMetrics':[{'returncode':0,'before':0x13579bdf,'after':0x13579bdf},
                                           {'returncode':0,'before':0x2468ace0,'after':0x2468ace0}],
               'cuptiDeviceGetNumEventDomains':[{'returncode':0,'before':0x13579bdf,'after':0x13579bdf},
                                                {'returncode':0,'before':0x2468ace0,'after':0x2468ace0}]}
        result=profiling.counter_interface_probe(probe)
        self.assertEqual(result['status'],'not-supported')
        self.assertTrue(result['evidence']['output_unchanged'])
        probe['cuptiDeviceGetNumMetrics'][0]['after']=0
        self.assertEqual(profiling.counter_interface_probe(probe)['status'],'unknown')
        self.assertEqual(profiling.counter_interface_probe({})['status'],'unknown')

    def test_counter_probe_does_not_load_an_unmapped_library(self):
        with patch.object(profiling.Path,'read_text',return_value=''), patch.object(profiling.ctypes,'CDLL') as load:
            result=profiling.probe_active_counter_interface(3)
        load.assert_not_called()
        self.assertEqual(result['status'],'unknown')

    def test_counter_probe_records_selected_device_library_and_unchanged_outputs(self):
        with patch.object(profiling.Path,'read_text',return_value='0-1 r-xp 0 00:00 0 /runtime/libcupti.so.12'), \
             patch.object(profiling.ctypes,'CDLL') as load, patch.object(profiling,'sha',return_value='library-digest'):
            library=load.return_value
            library.cuptiDeviceGetNumMetrics.return_value=0
            library.cuptiDeviceGetNumEventDomains.return_value=0
            result=profiling.probe_active_counter_interface(3)
        self.assertEqual(result['status'],'not-supported')
        self.assertEqual(result['device'],3)
        self.assertEqual(result['library_sha256'],'library-digest')
        self.assertTrue(load.call_args.kwargs['mode'] & 4)
        self.assertEqual(library.cuptiDeviceGetNumMetrics.call_args.args[0],3)
        library.cuptiActivityGetNumDroppedRecords.assert_not_called()

    def test_missing_ambiguous_wrong_thread_device_and_nan_fail_closed(self):
        for kind in ('source','finish','duplicate-launch','duplicate-source','duplicate-range',
                     'wrong-thread','wrong-device','missing-call','nan','negative','outside','unmatched-kernel','duplicate-kernel','boolean-device',
                     'negative-launch','zero-launch','malformed-args','malformed-event'):
            ev=events()
            if kind=='source':ev.pop(2)
            if kind=='finish':ev.pop(4)
            if kind=='duplicate-launch':ev.append(copy.deepcopy(ev[1]))
            if kind=='duplicate-source':ev.append(copy.deepcopy(ev[2]))
            if kind=='duplicate-range':ev.append(copy.deepcopy(ev[0]))
            if kind=='wrong-thread':ev[1]['tid']=123
            if kind=='wrong-device':ev[3]['args']['device']=1
            if kind=='missing-call':ev=ev[:20]
            if kind=='nan':ev[3]['dur']=float('nan')
            if kind=='negative':ev[3]['dur']=-2
            if kind=='outside':ev[1]['ts']=5
            if kind=='unmatched-kernel':ev[3]['args']['correlation']=100
            if kind=='duplicate-kernel':ev.append(copy.deepcopy(ev[3]))
            if kind=='boolean-device':ev[3]['args']['device']=False
            if kind=='negative-launch':ev[1]['dur']=-1
            if kind=='zero-launch':ev[1]['dur']=0
            if kind=='malformed-args':ev[3]['args']=None
            if kind=='malformed-event':ev.append(None)
            with self.subTest(kind=kind):self.assertEqual(profiling.analyze_events(ev)['status'],'partial')

    def test_overlap_union_is_not_sum_or_span(self):
        from decimal import Decimal as D
        self.assertEqual(profiling.union_duration([(D(0),D(3)),(D(2),D(4)),(D(6),D(7))]),5)

    def test_backward_thread_requires_forward_backward_flow_and_sequence(self):
        for missing in (None, 'flow', 'sequence', 'source-thread', 'duplicate-finish', 'duplicate-source'):
            ev=events()
            for i in range(5):
                host=1000+i*100
                ev[i*5+1]['tid']=9;ev[i*5+2]['tid']=9
                ev += [
                    {'ph':'X','cat':'cpu_op','name':'forward','pid':1,'tid':2 if missing!='source-thread' else 3,'ts':host+1,'dur':3,'args':{'Sequence number':i}},
                    {'ph':'X','cat':'cpu_op','name':'backward','pid':1,'tid':9,'ts':host+6,'dur':20,'args':{'Sequence number':i if missing!='sequence' else -1}},
                    {'ph':'s','cat':'fwdbwd','id':i,'pid':1,'tid':2,'ts':host+1},
                ]
                if missing!='flow':ev.append({'ph':'f','cat':'fwdbwd','id':i,'pid':1,'tid':9,'ts':host+6})
                if missing=='duplicate-finish':ev.append(copy.deepcopy(ev[-1]))
                if missing=='duplicate-source':ev.append(copy.deepcopy(ev[-2]))
            self.assertEqual(profiling.analyze_events(ev)['status'],'measured' if missing is None else 'partial',missing)

    def test_modes_plan_and_environment_isolation(self):
        for mode in ('off','timeline','full'):
            args=cli.parser().parse_args(['run','--vendor','kunlunxin','--runtime-profile','p800-xpytorch29',
                                         '--device-ids','7','--case','abs','--profiling',mode])
            _,plan=cli.resolve(args)
            self.assertEqual(plan['tasks'][0]['profiling_mode'],mode)
        adapter=Adapter()
        for phase in ('reference','measure','memory','trace'):
            self.assertEqual(adapter.environment(7,phase)['XPU_CUPTI_ENABLE_DEVICE'],'')
        for phase in ('probe','profiling','profile-target'):
            self.assertEqual(adapter.environment(7,phase)['XPU_CUPTI_ENABLE_DEVICE'],'0')
            self.assertEqual(adapter.environment(7,phase,True)['XPU_CUPTI_ENABLE_DEVICE'],'7')

    def test_failed_capture_stops_and_full_never_fabricates_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'inputs.pt').write_bytes(b'fixture');write(root/'task.json',{})
            with patch.object(profiling.subprocess,'Popen') as start:
                proc=start.return_value;proc.returncode=7;proc.poll.return_value=7
                profiling.collect(root,{'profiling_mode':'full','watchdog_s':30})
            result=read(root/'profiling.json')
            self.assertEqual(result['status'],'partial')
            self.assertTrue(result['execution_error'])
            self.assertEqual(len(result['groups']),7)
            self.assertEqual([g['status'] for g in result['groups'][1:]],['not-supported']*6)
            self.assertNotIn('start_new_session',start.call_args.kwargs)

    def test_capture_timeout_terminates_and_reaps_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write(root/'task.json',{})
            with patch.object(profiling.subprocess,'Popen') as start:
                proc=start.return_value;proc.returncode=-15;proc.poll.return_value=None
                proc.wait.side_effect=[profiling.subprocess.TimeoutExpired('capture',1),-15]
                profiling.collect(root,{'profiling_mode':'timeline','watchdog_s':1})
                proc.terminate.assert_called_once();self.assertEqual(proc.wait.call_count,2)
            self.assertTrue(read(root/'profiling.json')['execution_error'])

    def test_collector_requires_same_input_and_runtime_identity(self):
        from runtime.evidence import sha
        for drift in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'inputs.pt').write_bytes(b'fixture');write(root/'task.json',{})
                identity={k:'same' for k in ('packages','case_sha256','registration','runtime_config','device','input_devices')}
                identity['input_sha256']=sha(root/'inputs.pt')
                identity['vendor_runtime']={k:'same' for k in ('framework_observation','packages','executable','hook_hashes','profile_sha256')}
                write(root/'measurement.json',identity)
                def capture(*args,**kwargs):
                    folder=root/'profiling/timeline'
                    write(folder/'trace.json',{'traceEvents':events()})
                    write(folder/'profile-window.json',{'device':0,'calls':5})
                    record=copy.deepcopy(identity)
                    if drift:record['input_sha256']='different'
                    write(folder/'profile-target-identity.json',record)
                    return 0
                with patch.object(profiling.subprocess,'Popen') as start:
                    proc=start.return_value;proc.returncode=0;proc.poll.return_value=0;proc.wait.side_effect=capture
                    profiling.collect(root,{'profiling_mode':'timeline','watchdog_s':30})
                result=read(root/'profiling.json')
                self.assertEqual(result['status'],'partial' if drift else 'measured')
                self.assertEqual(result['phase_consistency']['status'],'failed' if drift else 'passed')
                if drift:
                    self.assertNotIn('calls',result['groups'][0]['timeline'])
                    self.assertNotIn('kernels',result['groups'][0]['timeline'])
                    self.assertTrue((root/'profiling/timeline/trace.json').is_file())


if __name__=='__main__':unittest.main()
