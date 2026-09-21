import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import sys

ROOT = Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
SOURCE = ROOT / 'base/result/p800-pr3-20260921-1011'
DEST = ROOT / 'base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/day4'
sys.path.insert(0, str(ROOT / 'base'))
sys.path.insert(0, str(ROOT))
from generate_benchmark_report import generate_benchmark_report
from executors.preflight import render_report


def sha(data):
    return hashlib.sha256(data).hexdigest()


def project(value):
    if isinstance(value, list):
        return [project(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: project(item) for key, item in value.items()}
    argv = result.get('argv', [])
    if 'stdout' in result and 'xpu-smi' in argv and '-m' in argv:
        result['stdout'] = ''.join(line for line in result['stdout'].splitlines(keepends=True)
                                  if len(shlex.split(line)) == 32 and shlex.split(line)[1] == '6')
        result['projection'] = 'selected physical card 6 only; original bytes remain in operational evidence'
    if 'stdout' in result and 'xpu-smi' in argv and '-q' in argv:
        result['stdout'] = re.sub(r'(?m)^(\s*Process ID\s*:\s*)\d+\s*$', r'\g<1>0', result['stdout'])
        result['stdout'] = re.sub(r'(?m)^(\s*Process Name\s*:\s*).+$', r'\g<1>[redacted]', result['stdout'])
    if 'stdout' in result and 'fuser' in argv and result['stdout'].strip():
        result['stdout'] = '[redacted nonempty device handle list]\n'
        result['projection'] = 'process identifiers redacted; original exit/occupancy state retained'
    if 'stdout' in result and 'docker' in argv and 'image' in argv and 'inspect' in argv and result.get('returncode') == 0:
        result['stdout'] = json.dumps([{key: row.get(key) for key in ('Id', 'RepoDigests', 'Architecture', 'Created')}
                                       for row in json.loads(result['stdout'])], indent=2) + '\n'
        result['projection'] = 'image identity fields only; full original inspect stays on development host'
    if 'Config' in result and 'HostConfig' in result:
        allowed = {'CUDA_VISIBLE_DEVICES', 'XPU_VISIBLE_DEVICES', 'XPU_EVENT_KL3_ENABLE', 'USE_FLAGGEMS',
                   'PYTHONDONTWRITEBYTECODE', 'PYTHONNOUSERSITE', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR',
                   'PATH', 'LD_LIBRARY_PATH', 'CONDA_DEFAULT_ENV', 'CONDA_PREFIX'}
        result['Config']['Env'] = [item for item in result['Config'].get('Env', []) if item.split('=', 1)[0] in allowed]
        result['Config']['Labels'] = {key: item for key, item in result['Config'].get('Labels', {}).items() if key == 'flagperf.run_id'}
        result['projection'] = 'container environment/labels restricted to execution-relevant fields'
    return result


DEST.mkdir(exist_ok=False)
provenance = []
for source in sorted(SOURCE.rglob('*')):
    if not source.is_file():
        continue
    relative = source.relative_to(SOURCE)
    destination = DEST / ('attempts' if (SOURCE / relative.parts[0] / 'summary.json').is_file() else 'logs') / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = source.read_bytes()
    content = raw
    projected = False
    if source.suffix in ('.json', '.jsonl'):
        if source.suffix == '.jsonl':
            original = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
            value = project(original)
            if value != original:
                content = ''.join(json.dumps(item, sort_keys=True) + '\n' for item in value).encode()
        else:
            original = json.loads(raw)
            value = project(original)
            if value != original:
                content = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
        projected = content != raw
    destination.write_bytes(content)
    provenance.append({'source': str(source.relative_to(ROOT)), 'destination': str(destination.relative_to(DEST)),
                       'source_sha256': sha(raw), 'archive_sha256': sha(content), 'byte_identical': not projected})

for source in (Path('/tmp/p800-pr3-run.py'), Path('/tmp/p800-pr3-verify.py'), Path('/tmp/p800-pr3-audit.py'), Path(__file__)):
    destination = DEST / 'scripts' / source.name
    destination.parent.mkdir(exist_ok=True)
    shutil.copyfile(source, destination)
    provenance.append({'source': str(source), 'destination': str(destination.relative_to(DEST)),
                       'source_sha256': sha(source.read_bytes()), 'archive_sha256': sha(destination.read_bytes()), 'byte_identical': True})

baseline = ROOT / 'base/result/p800-pr3-20260920/baseline-base.log'
shutil.copyfile(baseline, DEST / 'logs/baseline-base.log')
provenance.append({'source': str(baseline.relative_to(ROOT)), 'destination': 'logs/baseline-base.log',
                   'source_sha256': sha(baseline.read_bytes()), 'archive_sha256': sha(baseline.read_bytes()), 'byte_identical': True})

for summary_path in sorted((DEST / 'attempts').glob('*/summary.json')):
    root = summary_path.parent
    for monitor_path in (root / 'monitor/summary.json', root / 'benchmark-monitor/summary.json'):
        if not monitor_path.exists():
            continue
        record = json.loads(monitor_path.read_text())
        source_record = json.loads(monitor_path.read_text())
        for key in ('raw_samples', 'parsed_samples'):
            if key in record:
                path = root / record[key]['path']
                record[key].update(sha256=sha(path.read_bytes()), bytes=path.stat().st_size)
        if record != source_record:
            shutil.copyfile(monitor_path, monitor_path.with_name('summary.source.json'))
            monitor_path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    for report in root.glob('report*.md'):
        shutil.copyfile(report, report.with_name(report.stem + '.source.md'))
    summary = json.loads(summary_path.read_text())
    if summary.get('kind') == 'benchmark-preflight':
        render_report(root)
    elif summary.get('kind') == 'benchmark' or (root / 'benchmark-result.json').exists():
        generate_benchmark_report(root)

for entry in provenance:
    destination = DEST / entry['destination']
    entry['archive_sha256'] = sha(destination.read_bytes())
    entry['byte_identical'] = entry['source_sha256'] == entry['archive_sha256']
(DEST / 'provenance.json').write_text(json.dumps({'schema_version': 1, 'files': provenance,
    'projection_policy': 'Target-only machine inventory; process identifiers/names redacted; container env and image inspect narrowed. Monitor source hashes retained separately; reports regenerated for projected references. Experimental states unchanged.'}, indent=2) + '\n')
index = {str(path.relative_to(DEST)): sha(path.read_bytes()) for path in sorted(DEST.rglob('*')) if path.is_file()}
(DEST / 'sha256-index.json').write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')
print(json.dumps({'destination': str(DEST), 'files': len(index), 'source_files': len(provenance)}))
