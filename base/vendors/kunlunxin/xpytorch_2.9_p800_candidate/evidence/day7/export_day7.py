"""Export committed Day7 evidence and its scoped source/document delta."""
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
OUT=ROOT/'base/result/p800-day7-20260924'
BASE='ca9967d9'
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
paths=subprocess.check_output(['git','diff','--name-only','--diff-filter=ACMR',BASE,head],cwd=ROOT,text=True).splitlines()
evidence='base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/'
assert all(not p.startswith(evidence+'day6/') for p in paths),'User Day6 changes must not be committed'
paths=[p for p in paths if not p.startswith(evidence)]
assert paths
for p in paths:
 assert (ROOT/p).read_bytes()==subprocess.check_output(['git','show',head+':'+p],cwd=ROOT),p
archives=[]
for name,description,args in [
 ('remote-delivery.tar.gz','Day7 complete sanitized evidence archive',['tar','-czf',str(OUT/'remote-delivery.tar.gz'),evidence+'day7']),
 ('remote-delta.tar.gz','Source and document delta from ca9967d9; excludes evidence',['git','archive','--format=tar.gz','--output='+str(OUT/'remote-delta.tar.gz'),head,*paths])]:
 target=OUT/name;assert not target.exists(),target
 subprocess.run(args,cwd=ROOT,check=True)
 archives.append(dict(file=name,description=description,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size))
records=subprocess.check_output(['git','log','--reverse','--format=%H%x09%s',BASE+'..'+head],cwd=ROOT,text=True).splitlines()
commits=dict(baseline=BASE,head=head,commits=[dict(hash=x.split(chr(9),1)[0],subject=x.split(chr(9),1)[1]) for x in records],pushed=False,delta_files=paths)
(OUT/'staged/remote-commits.json').write_text(json.dumps(commits,indent=2)+'\n')
(OUT/'staged/archive-manifest.json').write_text(json.dumps(dict(archives=archives),indent=2)+'\n')
print(json.dumps(dict(archives=archives,commits=commits['commits'],delta_files=len(paths)),indent=2))

