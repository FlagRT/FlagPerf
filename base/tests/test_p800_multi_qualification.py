"""Exercise five-run cross-rank and shared-resource acceptance gates."""
import hashlib,json,tempfile,unittest
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from base.tests.test_p800_day6 import metric_record,correctness_record,VALID
from qualification import summarize

class MultiQualificationTests(unittest.TestCase):
 def fixture(self,parent,shared=False):
  case='interconnect-MPI_intraserver:P800';config=VALID[case];roots=[]
  for i in range(5):
   root=parent/str(i);roots.append(root)
   for d in ['control','artifacts','benchmark-monitor']:(root/d).mkdir(parents=True)
   assets=json.dumps({'merged_config':config}).encode();ah=hashlib.sha256(assets).hexdigest()
   bindings=[{'resource_key':f'kunlunxin/device-{r}'} for r in range(2)]
   context={'case':case,'run_id':str(i),'nproc_per_node':2,'case_assets_sha256':ah,
            'host':{'foreign_occupancy_observed':shared,'devices':[{'uncorrectable_ecc_counts':[0,0]}]*2}}
   raw=json.dumps(context).encode();ch=hashlib.sha256(raw).hexdigest();metrics=[]
   for r in range(2):
    m=metric_record(case,r);c=correctness_record(case,r)
    for x in [m,c]:x.update(run_id=str(i),context_sha256=ch,binding=bindings[r])
    m['case_assets_sha256']=ah;metrics.append(m)
    (root/f'artifacts/metric-rank-{r}.json').write_text(json.dumps(m))
    (root/f'artifacts/correctness-rank-{r}.json').write_text(json.dumps(c))
   summary={k:'passed' for k in ['status','execution_status','measurement_status','correctness_status','measurement_evidence_status','monitoring_status','postflight_status','cleanup_status']}
   summary.update(run_id=str(i),lease_released=True,qualification={'mode':'qualification'},runtime={'image_id':'fixed'},device_bindings=bindings)
   docs={'summary.json':summary,'code-identity.json':{'source_sha256':{'driver':'fixed'}},
         'benchmark-result.json':{'status':'passed','metrics':metrics,'expected_ranks':[0,1],'observed_ranks':[0,1],'missing_ranks':[]},
         'benchmark-monitor/summary.json':{'status':'passed','primary_sample_counts_by_target':{b['resource_key']:20 for b in bindings}}}
   for f,x in docs.items():(root/f).write_text(json.dumps(x))
   (root/'control/host-context.json').write_bytes(raw);(root/'control/case-assets.json').write_bytes(assets)
  return roots
 def test_two_rank_values_are_not_summed(self):
  with tempfile.TemporaryDirectory() as d:
   result=summarize(self.fixture(Path(d)))
   self.assertEqual(result['status'],'passed');self.assertEqual(result['rank_count'],2)
   self.assertEqual(result['runs'][0]['value'],min(result['runs'][0]['rank_values']))
 def test_shared_group_never_becomes_qualified(self):
  with tempfile.TemporaryDirectory() as d:
   roots=self.fixture(Path(d),True)
   with self.assertRaises(ValueError):summarize(roots)
   result=summarize(roots,allow_shared=True)
   self.assertEqual(result['status'],'exploratory');self.assertEqual(result['qualification_status'],'not-qualified')
 def test_missing_or_duplicate_rank_is_rejected(self):
  for mode in ['missing','duplicate','no-content-checks']:
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
    roots=self.fixture(Path(d));p=roots[0]/'benchmark-result.json';x=json.loads(p.read_text())
    if mode=='missing':x['metrics'].pop()
    elif mode=='duplicate':x['metrics'][1]=x['metrics'][0]
    else:
     c=roots[0]/'artifacts/correctness-rank-1.json';v=json.loads(c.read_text());v['checks']=[];c.write_text(json.dumps(v))
    p.write_text(json.dumps(x))
    with self.assertRaises((ValueError,RuntimeError)):summarize(roots)
if __name__=='__main__':unittest.main()
