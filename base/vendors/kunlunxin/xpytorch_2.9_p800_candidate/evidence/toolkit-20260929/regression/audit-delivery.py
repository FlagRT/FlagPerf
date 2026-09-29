"""Audit completed runs and export a bounded, explicitly scoped evidence bundle."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('--p2p-run', required=True)
parser.add_argument('--integration-run', required=True)
args = parser.parse_args()
repo = Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
root = repo/'base/result/p800-toolkit-20260929'
sys.path[:0] = [str(repo/'base'), str(repo)]
from toolkits._common.kunlunxin.P800.contract import CASES, metric_from_native
from toolkits._common.kunlunxin.P800.evidence import sha256, validate_index, write_json, index
from toolkits._common.kunlunxin.P800.report import generate_and_record

selected = ['20260929T041155Z-e100eb5e','20260929T042026Z-009e799d',args.p2p_run,args.integration_run]
audits = []
coverage = defaultdict(list)
groups = []

def json_file(p):
    return json.loads(p.read_text())

def file_hashes(p):
    return {f.relative_to(p).as_posix():sha256(f) for f in p.rglob('*') if f.is_file() and not f.is_symlink()}

for run in selected:
    p = root/run
    before = json_file(p/'summary.json')
    assert before['status'] != 'running', run
    assert before['cleanup_status'] == 'passed' and before['postflight_status'] == 'passed' and before['lease_released'] is True, run
    assert before['measurement_status'] == 'passed', run
    assert before['validated'] is False and before['release_stage'] == 'candidate', run
    assert not validate_index(p), (run,validate_index(p))
    immutable = {f.relative_to(p).as_posix():sha256(f) for f in (p/'toolkit-evidence').rglob('*') if f.is_file()}
    generate_and_record(p)
    first = file_hashes(p)
    generate_and_record(p)
    assert first == file_hashes(p), ('non-deterministic-report',run)
    assert immutable == {f.relative_to(p).as_posix():sha256(f) for f in (p/'toolkit-evidence').rglob('*') if f.is_file()}, ('mutated-measurement',run)
    after = json_file(p/'summary.json')
    assert {k:v for k,v in before.items() if k!='report_generation'} == {k:v for k,v in after.items() if k!='report_generation'}
    assert not validate_index(p)
    charts=list((p/'report-assets').glob('*.svg'))
    for chart in charts: ET.parse(chart)
    manifest = json_file(p/'toolkit-evidence/manifest.json')
    assert 1 not in manifest['selected_physical_device_ids']
    checked = 0
    for case, result in manifest['cases'].items():
        assert result['measurement_status'] == 'passed'
        coverage[case].append({'run_id':run,'measurement':result['measurement_status'],'monitoring':result['monitoring_status'],
                               'target_count':len(result['targets']),'settings':manifest['settings']})
        repeats=defaultdict(list)
        for target in result['targets']:
            point=target['point']
            assert point['source'] != 1 and point.get('destination') != 1
            assert target['measurement_status']=='passed'
            for command in target['commands']:
                assert command['returncode']==0
                for key in ('stdout','stderr'):
                    raw=p/'toolkit-evidence'/command[key]['path']
                    assert sha256(raw)==command[key]['sha256']
            if point['mode'] != 'capacity-query':
                native=json_file(p/'toolkit-evidence/cases'/case/target['target']/'samples.json')
                metric=target['metrics'][0]
                recomputed=metric_from_native(native,point,case,metric['source'],manifest['settings']['matrix_size'],manifest['settings']['samples'])
                assert math.isclose(recomputed['value'],metric['value'],rel_tol=1e-12)
                assert recomputed['unit']==metric['unit']
                repeats[json.dumps(point,sort_keys=True)].append(metric['value'])
            checked+=1
        for point, values in repeats.items():
            cv=statistics.pstdev(values)/abs(statistics.mean(values)) if len(values)>1 else None
            groups.append({'run_id':run,'case':case,'point':json.loads(point),'repeat_count':len(values),'values':values,
                           'median':statistics.median(values),'cv':cv,
                           'stability':'insufficient-repeats' if len(values)<5 else 'unstable' if cv>.05 else 'stable-observed'})
    audits.append({'run_id':run,'targets_verified':checked,'monitoring_status':manifest['monitoring_status'],
                   'report_regeneration':'byte-identical','experiment_status_preserved':True,'raw_evidence_unchanged':True,
                   'sha256_index':'passed','svg_xml_valid':len(charts),'summary_sha256':sha256(p/'summary.json'),
                   'index_sha256':sha256(p/'sha256-index.json')})
assert set(coverage)==set(CASES)
candidate=json_file(repo/'base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/image-manifest.json')
assert candidate['validated'] is False

bundle=repo/'base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/toolkit-20260929'
assert not bundle.exists(), 'refuse to overwrite an existing delivery bundle'
bundle.mkdir(parents=True)
write_json(bundle/'verification.json',{'schema_version':1,'runs':audits,'case_coverage':len(coverage),
    'formal_qualification':False,'physical_card_1_excluded':True,'candidate_validated':False})
write_json(bundle/'support-matrix.json',{'schema_version':1,'status':'functionally-verified','formal_qualification':False,
    'diagnosis_status':'not-supported','case_coverage':dict(coverage),
    'limits':['shared-card exploratory data','host synchronized API timing','INT8 fc_fusion FP32 output',
              'pinned async chunked payload','D2D workload differs from reference','Base two/eight-card gates unchanged']})
write_json(bundle/'repeat-groups.json',groups)
write_json(bundle/'qualification-record.json',{'schema_version':1,'kind':'toolkit-engineering-record','status':'scope-limited',
    'release_stage':'candidate','validated':False,'base_qualification_updated':False,'covered_cases':list(CASES),
    'validated_scope':'native Toolkit functionality, correctness, selected target monitoring, reports and evidence only',
    'not_qualified':['idle-resource performance','vendor threshold diagnosis','Base 8-card AllReduce','XTDK Kernel sample suite'],
    'source_files':{f:sha256(repo/f) for f in json_file(Path('/tmp/p800-toolkit-changed-files.json'))}})

all_runs=[]
for p in sorted(root.iterdir()):
    if not (p/'summary.json').is_file():continue
    s=json_file(p/'summary.json')
    assert s['status']!='running', p.name
    records=bundle/'run-records'/p.name
    records.mkdir(parents=True)
    for name in ('summary.json','code-identity.json','image-identity.json','sha256-index.json',
                 'toolkit-evidence/manifest.json','toolkit-evidence/provenance.json','toolkit-evidence/runtime-bindings.json'):
        source=p/name
        if source.is_file():
            dest=records/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,dest)
    all_runs.append({'run_id':p.name,'absolute_path':str(p),'status':s.get('status'),'measurement_status':s.get('measurement_status'),
                     'monitoring_status':s.get('monitoring_status'),'cleanup_status':s.get('cleanup_status'),
                     'postflight_status':s.get('postflight_status'),'lease_released':s.get('lease_released'),
                     'full_raw_evidence_mirrored':p.name==args.integration_run})
write_json(bundle/'run-index.json',all_runs)
shutil.copytree('/tmp/p800-toolkit-final-checks',bundle/'regression')
shutil.copyfile('/tmp/p800-toolkit-pre-fix-source.tar.gz',bundle/'pre-fix-source.tar.gz')
shutil.copyfile(repo/'base/toolkits/_common/kunlunxin/P800/toolkit-contract.json',bundle/'toolkit-contract.json')
# One complete small run is mirrored; long-run raw samples remain at explicit remote paths.
shutil.copytree(root/args.integration_run,bundle/'complete-integration-run')
(bundle/'README.md').write_text('''# P800 Toolkit engineering evidence — 2026-09-29

Twelve native Toolkit cases are functionally verified. This is shared-card exploratory evidence;
it does not promote the candidate or qualify the Base two/eight-card gates. Physical card 1 is excluded.

`support-matrix.json` maps each case to actual runs. `verification.json` records raw metric checks,
monitoring, immutable measurement evidence, byte-identical report regeneration and SHA-256 audits.
`repeat-groups.json` retains every repeat, including unstable groups. `qualification-record.json`
declares the exact engineering scope and hashes the delivered sources.

`complete-integration-run/` contains a full compact run with working relative evidence/report links.
`run-records/` contains original-byte metadata snapshots of all runs, including historical failures.
Its copied indexes describe the FULL remote runs, not the subset mirrored in that directory;
do not validate a full-run index against a metadata subset. See `run-index.json` for absolute
remote paths. The bundle root index describes only this bundle and includes nested run indexes.

The 04:21 P2P exploratory run used an early metadata counter reporting one logical work unit
for a bidirectional sample, although the implementation executed two actual peer copies.
The counter was corrected to two native submissions and a new five-repeat group was run.
The earlier raw artifact is retained without rewriting its measurements or status.

Official fc_effciency/perf_regression still need the matching XBLAS unittest. Vendor threshold
diagnosis is not-supported. Pinned 512 MiB async copies use explicit 1 MiB native chunks.
INT8 uses native fc_fusion with FP32 output. These scopes are not equivalent hardware peak claims.
''',encoding='utf-8')
# Include nested indexes here; the run-level helper intentionally omits files of that name.
files=[f for f in sorted(bundle.rglob('*')) if f.is_file() and f != bundle/'bundle-sha256-index.json']
write_json(bundle/'bundle-sha256-index.json',{'scope':'all regular files in this delivery bundle except this index',
    'files':[{'path':f.relative_to(bundle).as_posix(),'sha256':sha256(f),'bytes':f.stat().st_size} for f in files]})
with tarfile.open('/tmp/p800-toolkit-delivery-evidence.tar.gz','w:gz') as archive:
    archive.add(bundle,arcname='toolkit-20260929')
print(json.dumps({'bundle':str(bundle),'audits':audits,'repeat_groups':len(groups),
    'unstable_groups':sum(g['stability']=='unstable' for g in groups),
    'archive_bytes':Path('/tmp/p800-toolkit-delivery-evidence.tar.gz').stat().st_size},indent=2))
