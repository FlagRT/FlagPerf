"""Fresh two-rank regression after correcting rank-local synchronization."""
import json
from pathlib import Path
from run_day7 import ROOT, OUT, run

configdir=OUT/'communication-configs'
configdir.mkdir(exist_ok=True)
def config(case,m,iters,mode='qualification',fault='none'):
    p=configdir/(case+'-m'+str(m)+'-'+mode+'-'+fault+'.yaml')
    data=dict(Melements=m,WARMUP=10,ITERS=iters,SEED=519,DTYPE='float32',
              DIST_BACKEND='cpu:gloo,cuda:flagcx',IMPLEMENTATION='native-xpytorch',MODE=mode,FAULT_MODE=fault)
    text=''.join(str(k)+': '+str(v)+'\n' for k,v in data.items())
    if p.exists() and p.read_text()!=text:
        raise RuntimeError('frozen communication config differs')
    p.write_text(text)
    return p

for card in ('5','6'):
    if run('preflight-comm-card'+card,cards=card):
        raise RuntimeError('card unavailable: '+card)
# The standalone preflight CLI is single-card; each two-rank benchmark run
# performs the joint host checks, leases and in-container binding itself.
for short in ('MPI','P2P'):
    case='interconnect-'+short+'_intraserver'
    smoke=config(case,1,3,'smoke')
    if run('comm-'+short+'-smoke',case,smoke,monitor='off',cards='5,6',timeout=180):
        raise RuntimeError('two-card smoke failed')
    for m,iters in ((1,160000),(4,48000),(16,12000),(64,3000)):
        cfg=config(case,m,iters)
        for repeat in range(1,6 if m==64 else 2):
            if run('comm-'+short+'-m'+str(m)+'-q'+str(repeat),case,cfg,cards='5,6'):
                raise RuntimeError('communication qualification failed')
    for repeat in range(1,4):
        for monitor in ('off','on'):
            if run('comm-'+short+'-ab-'+monitor+'-'+str(repeat),case,config(case,64,3000),monitor=monitor,cards='5,6'):
                raise RuntimeError('communication AB failed')
    timeout_cfg=ROOT/'base/benchmarks'/case/'kunlunxin/P800/case_config.timeout.yaml'
    rc=run('comm-'+short+'-timeout',case,timeout_cfg,monitor='off',cards='5,6',timeout=60)
    if rc==0:
        raise RuntimeError('timeout injection unexpectedly succeeded')
print('DAY7-COMMUNICATION-COMPLETE',flush=True)
