# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Small independent numeric examples through the public oracle report."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from flagprof.report import generate_oracle
from flagprof.store import create_run,Writer,finalize,StoreView
from flagprof.tensor_data import metrics


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def fixture(self,label,*,delta=0,input_delta=0,config=1,omit=False,source='input',blob=True):
        run=create_run(self.root/label)
        identity=dict(model_files={'model':'same'},oracle_id='fixture',inputs_sha256='same',samples=['s'],batches=[[1]],dtype='float32',attention='eager',
            pooling='fixture',boundaries=['<root>.layer'],probe_alg='p4v1',device='cpu',software={},numerics={},flaggems='off',include=None)
        state=run/'attachments/f/state.json';state.parent.mkdir(parents=True)
        state.write_text(json.dumps(dict(id='f',probes={},oracle_identity=identity)))
        with Writer(run) as writer:
            for name,values in [('input',[1.,2.+input_delta]),('op:aten.mul/0/0',[2.,4.+delta])]:
                if omit and name.startswith('op:'):continue
                key='<root>.layer#0#'+name+'#s';raw=struct.pack('<2f',*values);path=run/'blobs'/('input.bin' if name=='input' else 'output.bin');path.write_bytes(raw)
                row=dict(t='tensor',call_key=key,role='in' if name=='input' else 'out',boundary=name,shape=[2],dtype='torch.float32',
                    stats={},chunks=[],probe=values,probe_alg='p4v1',digest='sha256:'+hashlib.sha256(raw).hexdigest())
                if blob:row['blob']=dict(path=str(path.relative_to(run)),offset=0,nbytes=len(raw),hash=row['digest'],fmt='raw-le:float32:[2]')
                if name.startswith('op:'):row.update(src=['unknown' if source=='unknown' else '<root>.layer#0#'+source+'#s'],ext={'call_config':config if isinstance(config,list) else {'scale':config}})
                writer.emit(row)
        finalize(run);return run

    def report(self,a,b,label):
        out=self.root/label;generate_oracle(a,b,out)
        return json.loads((out/'result.json').read_text())['results']

    def test_introduced_requires_same_actual_inputs_and_configuration(self):
        a=self.fixture('a');b=self.fixture('b',delta=.5)
        result=self.report(a,b,'equal-input')
        row=result['oracle.op_sensitivity']['data'][0]
        self.assertTrue(row['introduced']);self.assertEqual(row['error_in'],0)
        self.assertFalse(row['digest_equal'])
        c=self.fixture('c',delta=.5,config=2)
        result=self.report(a,c,'changed-config')
        self.assertFalse(result['oracle.op_sensitivity']['data'][0]['introduced'])
        self.assertEqual(result['oracle.op_sensitivity']['status'],'partial')
        self.assertEqual(result['oracle.first_divergence']['status'],'ok')
        d=self.fixture('d',delta=.5,source='unknown')
        unknown=self.report(a,d,'wrong-source')['oracle.op_sensitivity']
        self.assertFalse(unknown['data'][0]['introduced']);self.assertEqual(unknown['coverage']['alignment']['source_unknown'],1)
        e=self.fixture('e',delta=1,input_delta=.5)
        row=self.report(a,e,'upstream-error')['oracle.op_sensitivity']['data'][0]
        self.assertFalse(row['introduced']);self.assertAlmostEqual(row['amplification'],1.)
        f=self.fixture('f',delta=.5,config=[dict(name='eps',value=[dict(p='0',k='other',type='fixture')])])
        unknown=self.report(f,f,'unknown-config')['oracle.op_sensitivity'];row=unknown['data'][0]
        self.assertFalse(row['introduced']);self.assertEqual(row['call_config_status'],'unknown')
        self.assertEqual(unknown['coverage']['alignment']['source_unknown'],0)
        self.assertEqual(unknown['coverage']['alignment']['call_config_unknown'],1)

    def test_missing_pair_and_sampled_output_remain_explicit(self):
        a=self.fixture('a');b=self.fixture('b',omit=True)
        result=self.report(a,b,'missing')
        self.assertEqual(result['oracle.first_divergence']['coverage']['alignment']['only_ref'],1)
        c=self.fixture('c',delta=.5,blob=False)
        row=self.report(a,c,'sampled')['oracle.op_sensitivity']['data'][0]
        self.assertEqual(row['level'],'sampled');self.assertFalse(row['introduced'])

    def test_hand_computed_metrics(self):
        import math
        r=metrics(iter([1.,2.]),iter([1.,4.]))
        self.assertEqual(r['mse'],2.);self.assertEqual(r['mae'],1.);self.assertEqual(r['max_abs'],2.)
        self.assertEqual(r['relative_mean'],.5)
        self.assertEqual(r['relative_max'],1.)
        self.assertAlmostEqual(r['relative_l2'],2/math.sqrt(5))
        self.assertAlmostEqual(r['cosine_similarity'],9/math.sqrt(85))
        r=metrics([0.,float('nan'),float('inf')],[1.,0.,float('-inf')])
        self.assertEqual(r['valid_elements'],1);self.assertIsNone(r['cosine_similarity'])
        self.assertEqual(r['special']['ref']['nan'],1)
        self.assertEqual(r['relative_mean'],1e12)
        self.assertEqual(r['relative_max'],1e12);self.assertEqual(r['nonfinite_pair_mismatches'],2)

    def test_identity_difference_requires_declaration_and_inputs_cannot_be_declared(self):
        a=self.fixture('a');b=self.fixture('b')
        path=b/'attachments/f/state.json';state=json.loads(path.read_text())
        state['oracle_identity']['software']={'torch':'other'};path.write_text(json.dumps(state))
        result=self.report(a,b,'undeclared')
        self.assertEqual(result['oracle.identity_diff']['notes'],['undeclared_difference'])
        out=self.root/'declared';generate_oracle(a,b,out,declare={'differences':['software']})
        self.assertEqual(json.loads((out/'result.json').read_text())['results']['oracle.identity_diff']['status'],'ok')
        state['oracle_identity']['inputs_sha256']='other';path.write_text(json.dumps(state))
        out=self.root/'wrong-input';generate_oracle(a,b,out,declare={'differences':['software','inputs_sha256']})
        self.assertEqual(json.loads((out/'result.json').read_text())['results']['oracle.identity_diff']['notes'],['identity_mismatch'])

    def test_public_verify_detects_blob_corruption(self):
        run=self.fixture('a')
        state=run/'attachments/f/state.json';v=json.loads(state.read_text());v['tensor']={'events':2};state.write_text(json.dumps(v))
        with StoreView(run) as view:self.assertEqual(view.tensor_status(verify=True)['verified_blobs'],2)
        (run/'blobs/input.bin').write_bytes(b'broken!!')
        with StoreView(run) as view:
            with self.assertRaisesRegex(ValueError,'hash mismatch'):view.tensor_status(verify=True)

    def bit_pair(self,dtype,left,right,*,stats=False):
        from flagprof.tensor_data import FLOATS,compare
        width,code,_=FLOATS[dtype];events=[]
        for label,bits in [('ref',left),('cmp',right)]:
            raw=struct.pack('<'+str(len(bits))+code,*bits);path=self.root/(label+'.bin');path.write_bytes(raw)
            event=dict(dtype='torch.'+dtype,shape=[len(bits)],digest=hashlib.sha256(raw).hexdigest(),
                blob=dict(path=path.name,offset=0,nbytes=len(raw)),stats={})
            if stats:
                from flagprof.tensor_data import values
                vals=list(values(self.root,event,'full'))
                event['stats']=dict(count=len(vals),nan=0,inf=0,min=min(vals) if vals else None,
                    max=max(vals) if vals else None,l2=sum(v*v for v in vals)**.5)
            events.append(event)
        return events,compare(self.root,events[0],self.root,events[1])

    def test_paired_extremes_and_single_extremes_remain_distinct_from_nonfinite(self):
        examples=[('float32',0x7f7fffff,0xff7fffff,0x3f800000,0x40000000),
                  ('bfloat16',0x7f7f,0xff7f,0x3f80,0x4000),('float16',0x7bff,0xfbff,0x3c00,0x4000)]
        for dtype,hi,lo,one,two in examples:
            with self.subTest(dtype=dtype):
                _,r=self.bit_pair(dtype,[hi,lo,one],[hi,lo,two])
                self.assertEqual(r['paired_sentinel'],2);self.assertEqual(r['valid_elements'],1)
                self.assertEqual(r['max_abs'],1);self.assertEqual(r['max_abs_index'],2)
                self.assertEqual(r['cosine_similarity'],1);self.assertEqual(r['differing_fraction'],1)
                self.assertEqual(r['nonfinite_pair_mismatches'],0)
                _,r=self.bit_pair(dtype,[hi,lo,one],[one,lo,two])
                self.assertEqual(r['finite_extreme_pair_mismatches'],1)
                self.assertEqual(r['nonfinite_pair_mismatches'],0);self.assertEqual(r['valid_elements'],2)
        _,r=self.bit_pair('float32',[0x7f7fffff,0x7fc00000,0x3f800000],[0x7f7fffff,0,0x40000000])
        self.assertEqual(r['paired_sentinel'],1);self.assertEqual(r['special']['ref']['nan'],1)
        self.assertEqual(r['nonfinite_pair_mismatches'],1);self.assertIsNone(r['cosine_similarity'])
        _,r=self.bit_pair('float32',[0x7f7fffff],[0x7f7fffff],stats=True)
        self.assertEqual(r['valid_elements'],0);self.assertIsNone(r['max_abs'])
        self.assertIsNone(r['differing_fraction']);self.assertEqual(r['cosine_undefined_reason'],'no_valid_elements')

    def test_ulp_literal_buckets_cross_zero_and_first_maximum(self):
        steps=[1,2,3,4,5,16,17,256,257]
        _,r=self.bit_pair('float32',[0x3f800000]*len(steps),[0x3f800000+n for n in steps])
        self.assertEqual(r['ulp_histogram'],{'1':1,'2':1,'3-4':2,'5-16':2,'17-256':2,'>256':1})
        self.assertEqual(r['ulp_max'],257);self.assertEqual(r['max_abs_ulp'],257)
        self.assertEqual(r['max_abs_index'],8);self.assertEqual(r['differing_elements'],9)
        _,r=self.bit_pair('float32',[1,0,0x80000000],[0x80000001,0x80000000,0])
        self.assertEqual(r['ulp_histogram']['2'],1);self.assertEqual(r['differing_elements'],1)
        self.assertEqual(r['differing_fraction'],1/3)
        for dtype,base in [('bfloat16',0x3f80),('float16',0x3c00),('float64',0x3ff0000000000000)]:
            _,r=self.bit_pair(dtype,[base],[base+1]);self.assertEqual(r['ulp_max'],1)
        # Both deltas are 2**-23; the first is one ULP and the second two ULP.
        _,r=self.bit_pair('float32',[0x3f800000,0x3f000000],[0x3f800001,0x3f000002])
        self.assertEqual(r['max_abs_index'],0);self.assertEqual(r['max_abs_ulp'],1)

    def test_equal_digest_fast_path_requires_stats_excluding_extremes(self):
        from unittest.mock import patch
        from flagprof.tensor_data import compare
        events,r=self.bit_pair('float32',[0x3f800000],[0x3f800000],stats=True)
        with patch('flagprof.tensor_data.values',side_effect=AssertionError('unexpected blob scan')):
            r=compare(self.root,events[0],self.root,events[1])
        self.assertEqual(r['ulp_max'],0);self.assertEqual(r['max_abs_index'],0)
        events[0]['stats']={};events[1]['stats']={}
        with patch('flagprof.tensor_data.values',side_effect=AssertionError('required blob scan')):
            with self.assertRaisesRegex(AssertionError,'required blob scan'):compare(self.root,events[0],self.root,events[1])
        events,r=self.bit_pair('float32',[],[],stats=True)
        self.assertIsNone(r['differing_fraction']);self.assertEqual(r['cosine_undefined_reason'],'empty_tensor')

    def test_layout_is_separate_and_scalar_configuration_is_preserved(self):
        from copy import deepcopy
        config=[dict(name='x',value=[dict(p='0',k='tensor',shape=[2],dtype='torch.float32',layout='torch.strided',contiguous=True)]),
                dict(name='eps',value=[dict(p='0',k='float',v=1.)])]
        other=deepcopy(config);other[0]['value'][0].update(contiguous=False,stride=[2])
        a=self.fixture('a',config=config);b=self.fixture('b',delta=.5,config=other)
        op=self.report(a,b,'layout')['oracle.op_sensitivity']
        self.assertEqual(op['status'],'ok');self.assertTrue(op['data'][0]['layout_differs'])
        self.assertTrue(op['data'][0]['introduced'])
        other[1]['value'][0]['v']=2.
        c=self.fixture('c',delta=.5,config=other)
        op=self.report(a,c,'eps')['oracle.op_sensitivity'];self.assertFalse(op['data'][0]['introduced'])
        self.assertEqual(op['coverage']['alignment']['call_config_mismatch'],1)
        from flagprof.analyzers.oracle import call_config
        self.assertNotEqual(call_config({'ext':{'call_config':{'stride':1}}}),call_config({'ext':{'call_config':{'stride':2}}}))

    def test_layer_maximum_provenance_and_population_histogram(self):
        template=self.fixture('template')
        identity=json.loads((template/'attachments/f/state.json').read_text())
        def run(label,candidate):
            target=create_run(self.root/label);state=target/'attachments/f/state.json';state.parent.mkdir(parents=True);state.write_text(json.dumps(identity))
            with Writer(target) as writer:
                for sample,x,y in [('large',6176.,6208.),('small',1.,1.125),('tie',6176.,6208.)]:
                    raw=struct.pack('<H',struct.unpack('<I',struct.pack('<f',y if candidate else x))[0]>>16);path=target/'blobs'/(sample+'.bin');path.write_bytes(raw)
                    writer.emit(dict(t='tensor',call_key='<root>.layer#0#layer#'+sample,role='out',boundary='layer',
                        shape=[1],dtype='torch.bfloat16',stats={},chunks=[],probe=[y if candidate else x],probe_alg='p4v1',digest='sha256:'+hashlib.sha256(raw).hexdigest(),
                        blob=dict(path=str(path.relative_to(target)),offset=0,nbytes=2,hash='sha256:'+hashlib.sha256(raw).hexdigest(),fmt='raw-le:bfloat16:[1]')))
            finalize(target);return target
        result=self.report(run('ref',False),run('cmp',True),'population')
        layer=result['oracle.layer']['data'][0]
        self.assertEqual(layer['worst']['max_abs'],dict(value=32.,sample='large',ulp=1,index=0))
        self.assertEqual(layer['ulp_histogram']['1'],2);self.assertEqual(layer['ulp_histogram']['5-16'],1)
        self.assertEqual(layer['valid_elements'],3);self.assertEqual(layer['differing_elements'],3)
        self.assertEqual(layer['differing_fraction'],1)
        document=(self.root/'population/report.html').read_text()
        self.assertLess(document.index('id="oracle.identity_diff"'),document.index('id="oracle.first_divergence"'))
        self.assertNotIn('relative_mean',document.split('<section id="oracle.layer">')[1].split('</section>')[0])

    def test_horizontal_cells_keep_legacy_field_mapping(self):
        import io
        from tests.store.html_cells import Cells
        from flagprof.report import html_rows
        rows=[dict(exclusive_ns=350,name='a'),dict(exclusive_ns=700,name='b')]
        out=io.StringIO();html_rows(out,'perf.op',iter(rows),2)
        parser=Cells();parser.feed(out.getvalue())
        self.assertEqual(parser.rows['perf.op'],2)
        for i,value in enumerate((350,700)):
            self.assertEqual(parser.values[('perf.op',i,'exclusive_ns')],['ns',[str(value)]])
        old=Cells();old.feed('<table data-panel="perf.op" data-row="1"><tr><td data-field="exclusive_ns" data-unit="ns">700</td></tr></table>')
        self.assertEqual(old.values[('perf.op',1,'exclusive_ns')],['ns',['700']])


if __name__=='__main__':unittest.main()
