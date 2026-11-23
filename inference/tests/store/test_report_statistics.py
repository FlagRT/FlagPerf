"""Independent numeric and scope checks for user-facing report statistics."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from flagprof.analyzers.workspace import Limits
from flagprof.report.comparison import conditions,change
from flagprof.report.performance import summarize
from flagprof.report.model import opened_model
from flagprof.report.curves import memory_envelope
from flagprof.tensor_data import metrics


class ReportStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def test_exact_quantiles_scale_and_near_zero_external_path(self):
        result=metrics([0.,1.,2.,3.],[0.,2.,4.,6.],work=self.root,limits=Limits(cache_bytes=64))
        for key,expected in {'50':1.5,'90':2.7,'99':2.97}.items():self.assertAlmostEqual(result['absolute_error_quantiles'][key],expected)
        self.assertAlmostEqual(result['reference_rms'],math.sqrt(3.5))
        self.assertEqual(result['normalized_rmse'],1.)
        self.assertEqual(result['near_zero_reference_fraction'],.25)
        self.assertEqual(list(self.root.iterdir()),[])
        zero=metrics([0.,float('nan')],[1.,0.])
        self.assertIsNone(zero['normalized_rmse']);self.assertEqual(zero['error_quantile_elements'],1)
        self.assertEqual(zero['normalized_rmse_reason'],'zero_reference_scale')

    def test_declaration_does_not_relax_model_semantics_or_window(self):
        identity=dict(model_files={'x':'abc'},inputs_sha256='a',samples=['x'],dtype='float32',attention='eager',
                      software={'torch':'a'},numerics={},device='cpu',pooling='mean')
        window=dict(warmup=1,forwards=2,repeats=1,profiler='on',dispatch='off',window_kind='source')
        a=dict(workload_identity=identity,window_identity=window)
        b=dict(workload_identity=dict(identity,device='npu',software={'torch':'b'}),window_identity=window)
        self.assertEqual(conditions(a,b)['status'],'not_comparable')
        declared=dict(differences=['device','software.torch'])
        self.assertEqual(conditions(a,b,declared)['status'],'comparable')
        b['workload_identity']['pooling']='last'
        self.assertEqual(conditions(a,b,declared)['status'],'not_comparable')
        with self.assertRaises(ValueError):conditions(a,b,dict(differences=['pooling']))
        self.assertIsNone(change(0,1,True)['relative_change']);self.assertEqual(change(0,1,True)['delta'],1)

    def runner(self,name,values,execution=None):
        path=self.root/(name+'.json')
        path.write_text(json.dumps(dict(instrumentation='none',oracle_identity=dict(model_files={'x':'a'},samples=['a','b'],
            batches=[['a','b']],dtype='float32'),runs=[dict(warmup=5,latencies_ns=values)])))
        if execution:
            receipt=self.root/(name+'-receipt.json');receipt.write_text(json.dumps(dict(schema='runner-evidence/1',runner=path.name,execution=execution)))
            return receipt
        return path

    def test_warmup_duplicate_inputs_repeat_provenance_and_spikes(self):
        a=self.runner('a',[10]*19+[60]);b=self.runner('b',[10]*19+[60])
        result=summarize([a,b]);row=result['groups'][0]
        self.assertEqual(row['observations'],20);self.assertEqual(row['spike_count'],1)
        self.assertEqual(row['spike_screening'],'triggered');self.assertEqual(row['independent_repeats'],0)
        self.assertIsNone(row['repeat_cv']);self.assertEqual(row['sample_count'],40)
        inputs=[self.runner(str(i),[10+i]*20,dict(boot_id='boot',pid=i+1,start_ticks=1)) for i in range(3)]
        row=summarize(inputs)['groups'][0]
        self.assertEqual(row['independent_repeats'],3);self.assertAlmostEqual(row['repeat_cv'],1/11)

    def test_generation_warmup_and_decode_steps_have_distinct_denominators(self):
        path=self.root/'generation.json';step=lambda n,t:dict(generation_step=n,phase='prefill' if n==0 else 'decode',latency_ns=t,active_samples=[True])
        request=dict(sample_ids=['s'],input_tokens=4,generated_tokens=3,latency_ns=100,steps=[step(0,50),step(1,20),step(2,30)])
        path.write_text(json.dumps(dict(schema='generation-runner/1',oracle_identity={'samples':['s']},instrumentation='none',
            requests=[dict(request,warmup=True),dict(request,warmup=False)])))
        groups=summarize([path])['groups'];by_scope={g['scope']:g for g in groups}
        self.assertEqual(by_scope['request']['observations'],1)
        self.assertEqual(by_scope['request']['generated_tokens_per_second'],3e7)
        self.assertEqual(by_scope['request_decode']['mean_ns'],50)
        self.assertEqual(by_scope['request_decode']['generated_tokens_per_second'],4e7)

    def test_full_range_peak_and_null_survive_curve_reduction(self):
        points=(dict(dev='d',component='c',source_time_ns=i,values={'allocated_bytes':999999 if i==9001 else None if i==4500 else i%17}) for i in range(10000))
        with opened_model() as model:
            count=memory_envelope(model,points);rows=list(model.rows('memory.series'))
            self.assertEqual(count,10000);self.assertLess(len(rows),2500)
            self.assertEqual(max(r['values']['allocated_bytes'] or 0 for r in rows),999999)
            self.assertTrue(any(r['values']['allocated_bytes'] is None for r in rows))

    def test_group_request_requires_clock_and_counts_tokens_once(self):
        paths=[]
        for rank in (0,1):
            path=self.root/('rank-'+str(rank)+'.json')
            request=dict(request_id='r',warmup=False,sample_ids=['s'],input_tokens=4,generated_tokens=2,
                tokens=[[1,2]],stop_reasons=['length'],start_ns=100+10*rank,end_ns=200+20*rank,latency_ns=100+10*rank,steps=[])
            path.write_text(json.dumps(dict(schema='generation-runner/1',oracle_identity=dict(samples=['s'],parallelism=dict(world_size=2)),
                rank=rank,instrumentation='none',requests=[request])))
            receipt=self.root/('receipt-'+str(rank)+'.json')
            receipt.write_text(json.dumps(dict(schema='runner-evidence/1',runner=path.name,
                execution=dict(group_execution='execution',boot_id='boot',clock='perf_counter_ns',pid=rank+1,start_ticks=1))))
            paths.append(receipt)
        result=summarize(paths);group=next(g for g in result['groups'] if g['rank']=='group')
        self.assertEqual(group['mean_ns'],120);self.assertEqual(group['generated_tokens'],2)
        value=json.loads(paths[1].read_text());value['execution']['boot_id']='other';paths[1].write_text(json.dumps(value))
        self.assertFalse(any(g['rank']=='group' for g in summarize(paths)['groups']))

    def test_instrumentation_overhead_requires_pair_protocol(self):
        a=self.runner('plain',[10]*20);b=self.runner('recorded',[12]*20)
        value=json.loads(b.read_text());value['instrumentation']='attachment';b.write_text(json.dumps(value))
        self.assertEqual(summarize([a,b])['instrumentation_pairs'],[])
        paths=[]
        for index,path in enumerate((a,b)):
            receipt=self.root/('pair-'+str(index)+'.json')
            receipt.write_text(json.dumps(dict(schema='runner-evidence/1',runner=path.name,
                execution=dict(pair_id='pair',measurement_protocol='synchronized-forward-v1'))))
            paths.append(receipt)
        result=summarize(paths)['instrumentation_pairs']
        self.assertEqual(len(result),1);self.assertAlmostEqual(result[0]['overhead_ratio'],.2)


if __name__=='__main__':unittest.main()
