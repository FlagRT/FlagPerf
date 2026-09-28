"""Package a provenance-labelled projection; never edit raw or historical runs."""
from pathlib import Path
import ast
import hashlib
import json
import re
import shutil
ROOT=Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
OUT=ROOT/'base/result/p800-day7-20260924'
DEST=ROOT/'base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/day7'
DEST.mkdir(parents=True,exist_ok=True)
needles=[]
policy=OUT/'staged/redaction-policy.json'
if policy.exists(): needles.extend(json.loads(policy.read_text()))
# Reuse historical redaction policy without copying third-party identifiers.
old=ROOT/'base/result/p800-day6-20260923/redact_day6.py'
if old.exists():
 for node in ast.walk(ast.parse(old.read_text())):
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NEEDLES' for t in node.targets):
   needles.extend(ast.literal_eval(node.value))
private_home=re.compile(r'/home/(?!kzhang519/)[A-Za-z0-9_.-]+/')
secret_patterns=[re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
                 re.compile(r'(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}'),
                 re.compile(r'Bearer [A-Za-z0-9_.-]{30,}')]
entries=[]; leaks=[]
for source in sorted(OUT.rglob('*')):
 if not source.is_file(): continue
 rel=source.relative_to(OUT)
 if any(part in ('__pycache__','staged') for part in rel.parts): continue
 if source.suffix in ('.pyc','.tar','.gz'): continue
 data=source.read_bytes(); projected=data
 try:
  text=data.decode('utf-8'); cleaned=text
  for needle in needles: cleaned=cleaned.replace(needle,'[third-party-redacted]')
  cleaned=private_home.sub('/home/[third-party-redacted]/',cleaned)
  if cleaned!=text: projected=cleaned.encode('utf-8')
  if any(pattern.search(cleaned) for pattern in secret_patterns): leaks.append(str(rel))
 except UnicodeDecodeError:
  raise RuntimeError('unexpected binary requires explicit review: '+str(rel))
 target=DEST/rel; target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(projected)
 entries.append(dict(path=str(rel),source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(data).hexdigest(),
                     archive_sha256=hashlib.sha256(projected).hexdigest(),projection='third-party identifier redaction' if data!=projected else 'byte-identical'))
if leaks: raise RuntimeError('credential-pattern matches require review: '+repr(leaks))
(DEST/'provenance.json').write_text(json.dumps(dict(schema_version=1,files=entries, policy='Raw sources and Day1-Day6 unchanged; redacted projections labelled individually; metrics and run statuses are not normalized'),indent=2)+'\n')
index={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.rglob('*'))
       if p.is_file() and str(p.relative_to(DEST)) not in ('sha256-index.json','verification.json')}
(DEST/'sha256-index.json').write_text(json.dumps(index,indent=2)+'\n')
mismatch=[n for n,h in index.items() if hashlib.sha256((DEST/n).read_bytes()).hexdigest()!=h]
source_mismatch=[e['source'] for e in entries if hashlib.sha256((ROOT/e['source']).read_bytes()).hexdigest()!=e['source_sha256']]
verification=dict(status='passed' if not mismatch and not source_mismatch and not leaks else 'failed',index_entries=len(index), provenance_entries=len(entries),redacted_files=sum(e['projection']!='byte-identical' for e in entries), index_mismatches=mismatch,source_mismatches=source_mismatch,credential_pattern_matches=leaks, qualification_status='see qualification/overview.json; package integrity does not imply every experiment passed')
(DEST/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification,indent=2))
