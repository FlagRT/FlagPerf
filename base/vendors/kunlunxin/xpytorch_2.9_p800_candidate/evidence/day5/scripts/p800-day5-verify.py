import hashlib
import json
from pathlib import Path

ROOT = Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
DEST = ROOT / 'base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/day5'


def sha(data):
    return hashlib.sha256(data).hexdigest()


index = json.loads((DEST / 'sha256-index.json').read_text())
mismatches, missing = [], []
for relative, expected in index.items():
    path = DEST / relative
    if not path.is_file():
        missing.append(relative)
    elif sha(path.read_bytes()) != expected:
        mismatches.append(relative)

provenance = json.loads((DEST / 'provenance.json').read_text())
source_mismatch, archive_mismatch = [], []
for entry in provenance['files']:
    source = Path(entry['source'])
    if not source.is_absolute():
        source = ROOT / entry['source']
    destination = DEST / entry['destination']
    if source.is_file() and sha(source.read_bytes()) != entry['source_sha256']:
        source_mismatch.append(entry['source'])
    if destination.is_file() and sha(destination.read_bytes()) != entry['archive_sha256']:
        archive_mismatch.append(entry['destination'])

# Leak scan for third-party identities in committed evidence.
# This scanner and its own output necessarily contain the needle list itself.
needles = ('chenyunlong', 'qiyiyan', 'zhenghaojia', 'hliu553', 'xliu969', 'lianzhongyou',
           'cgu135', 'jliu171', 'xianyiyuan', 'dev-zkm', 'x-benchmark', 'model_runner')
excluded = {DEST / 'scripts/p800-day5-verify.py', DEST / 'verification.json'}
leaks = []
for path in DEST.rglob('*'):
    if not path.is_file() or path.stat().st_size > 4 * 1024 * 1024 or path in excluded:
        continue
    if path.suffix in ('.jsonl',):
        continue
    try:
        text = path.read_text(errors='ignore')
    except OSError:
        continue
    hits = [needle for needle in needles if needle in text]
    if hits:
        leaks.append({'file': str(path.relative_to(DEST)), 'hits': hits})

result = {'index_entries': len(index), 'index_missing': missing[:10], 'index_mismatches': mismatches[:10],
          'provenance_entries': len(provenance['files']), 'source_mismatches': source_mismatch[:10],
          'archive_mismatches': archive_mismatch[:10], 'third_party_leaks': leaks[:20],
          'status': 'passed' if not (missing or mismatches or source_mismatch or archive_mismatch or leaks) else 'failed'}
(DEST / 'verification.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, indent=2)[:2000])
