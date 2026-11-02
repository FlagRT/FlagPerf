"""Streaming tensor values and descriptive errors, using only the standard library."""
import hashlib
import math
from pathlib import Path
import struct


FLOATS={
    'float64': (64, 'Q', float.fromhex('0x1.fffffffffffffp+1023')),
    'float32': (32, 'I', float.fromhex('0x1.fffffep+127')),
    'float16': (16, 'H', 65504.),
    'bfloat16': (16, 'H', float.fromhex('0x1.fep+127')),
}
ULP_BUCKETS=('1','2','3-4','5-16','17-256','>256')


FORMATS={'float64':'d','float32':'f','float16':'e','bfloat16':'H','int64':'q','int32':'i','int16':'h','int8':'b','uint8':'B','bool':'?'}


def blob_path(run,blob):
    root=Path(run).resolve();path=(root/blob['path']).resolve()
    if not path.is_relative_to(root):raise ValueError('blob path outside Store')
    return path


def values(run,event,level,*,with_bits=False):
    if level!='full':
        for value in event.get('probe',[]):
            yield float(value.replace('+inf','inf')) if isinstance(value,str) else value
        return
    blob=event['blob'];dtype=event['dtype'].removeprefix('torch.')
    code=FORMATS.get(dtype)
    if code is None:raise ValueError('unsupported blob dtype: '+dtype)
    unit=struct.calcsize('<'+code);remaining=blob['nbytes']
    if remaining!=math.prod(event['shape'])*unit:raise ValueError('blob shape/length mismatch')
    with blob_path(run,blob).open('rb') as stream:
        stream.seek(blob['offset'])
        while remaining:
            raw=stream.read(min(65536//unit*unit,remaining))
            if not raw or len(raw)%unit:raise ValueError('truncated tensor blob')
            remaining-=len(raw)
            decoded=struct.iter_unpack('<'+code,raw)
            if with_bits:
                for (value,),(bits,) in zip(decoded,struct.iter_unpack('<'+FLOATS[dtype][1],raw)):
                    number=struct.unpack('<f',struct.pack('<I',value<<16))[0] if dtype=='bfloat16' else value
                    yield number,bits
            else:
                for value, in decoded:
                    yield struct.unpack('<f',struct.pack('<I',value<<16))[0] if dtype=='bfloat16' else value


def verify_blob(run,event):
    blob=event['blob'];digest=hashlib.sha256();remaining=blob['nbytes']
    if remaining<0 or blob['offset']<0:raise ValueError('invalid blob range')
    with blob_path(run,blob).open('rb') as stream:
        stream.seek(blob['offset'])
        while remaining:
            raw=stream.read(min(65536,remaining))
            if not raw:raise ValueError('truncated tensor blob')
            remaining-=len(raw);digest.update(raw)
    if 'sha256:'+digest.hexdigest()!=blob['hash']:raise ValueError('tensor blob hash mismatch: '+event['call_key'])


def _ulp_distance(a: int, b: int, width: int) -> int:
    bias=1<<(width-1)
    def ordered(bits):
        magnitude=bits&(bias-1)
        return bias-magnitude if bits&bias else bias+magnitude
    return abs(ordered(a)-ordered(b))


def _ulp_bucket(distance: int) -> str:
    if distance<=2:return str(distance)
    if distance<=4:return '3-4'
    if distance<=16:return '5-16'
    if distance<=256:return '17-256'
    return '>256'


def metrics(reference,candidate,*,dtype=None,with_bits=False):
    """One pass over pairs; ULP is descriptive and never an accuracy verdict."""
    from itertools import zip_longest
    missing=object();n=count=0;absolute=square=relative=aa=bb=ab=0.;max_abs=max_relative=0.
    nonfinite_mismatches=nonfinite_pairs=paired_sentinel=extreme_mismatches=0
    spec=FLOATS.get(dtype);maximum=spec[2] if spec else None
    histogram=dict.fromkeys(ULP_BUCKETS,0);differing=0;ulp_max=0;max_abs_ulp=max_abs_index=None
    special={side:dict(nan=0,positive_inf=0,negative_inf=0) for side in ('ref','cmp')}
    for index,(a,b) in enumerate(zip_longest(reference,candidate,fillvalue=missing)):
        if a is missing or b is missing:raise ValueError('tensor population mismatch')
        x,xb=a if with_bits else (a,None);y,yb=b if with_bits else (b,None)
        count+=1
        for side,value in [('ref',x),('cmp',y)]:
            if math.isnan(value):special[side]['nan']+=1
            elif math.isinf(value):special[side]['positive_inf' if value>0 else 'negative_inf']+=1
        if not math.isfinite(x) or not math.isfinite(y):
            category=lambda v:'nan' if math.isnan(v) else '+inf' if v==math.inf else '-inf' if v==-math.inf else 'finite'
            nonfinite_mismatches+=category(x)!=category(y);nonfinite_pairs+=1
            continue
        if maximum is not None and x==y and abs(x)==maximum:
            paired_sentinel+=1
            continue
        if maximum is not None and x!=y and (abs(x)==maximum or abs(y)==maximum):extreme_mismatches+=1
        delta=abs(y-x);rel=delta/max(abs(x),1e-12)
        distance=_ulp_distance(xb,yb,spec[0]) if with_bits else None
        if with_bits and distance:
            differing+=1;histogram[_ulp_bucket(distance)]+=1;ulp_max=max(ulp_max,distance)
        if max_abs_index is None or delta>max_abs:
            max_abs=delta;max_abs_index=index;max_abs_ulp=distance
        n+=1;absolute+=delta;square+=delta*delta;relative+=rel
        max_relative=max(max_relative,rel);aa+=x*x;bb+=y*y;ab+=x*y
    def finite(v):return v if v is None or math.isfinite(v) else None
    cosine=ab/math.sqrt(aa)/math.sqrt(bb) if not nonfinite_pairs and n and aa>0 and bb>0 else None
    reason='nonfinite_elements' if nonfinite_pairs else 'empty_tensor' if not count else 'no_valid_elements' if not n else 'zero_norm' if aa==0 or bb==0 else 'norm_overflow' if not math.isfinite(aa*bb) else None
    result=dict(cosine_undefined_reason=reason,nonfinite_pair_mismatches=nonfinite_mismatches,
        paired_sentinel=paired_sentinel,finite_extreme_pair_mismatches=extreme_mismatches,
        mse=finite(square/n) if n else None,mae=finite(absolute/n) if n else None,
        max_abs=max_abs if n else None,max_abs_index=max_abs_index,
        relative_mean=finite(relative/n) if n else None,relative_max=finite(max_relative) if n else None,
        cosine_similarity=finite(cosine),relative_l2=finite(math.sqrt(square)/max(math.sqrt(aa),1e-12)) if n else None,
        valid_elements=n,total_elements=count,statistics_scope='excluding_paired_sentinels' if paired_sentinel else 'all_elements' if n==count else 'finite_pairs_only',special=special)
    if with_bits:
        result.update(differing_elements=differing,differing_fraction=differing/n if n else None,
            ulp_histogram=histogram,ulp_max=ulp_max if n else None,max_abs_ulp=max_abs_ulp)
    return result


def compare(ref_run,a,cmp_run,b):
    if a['shape']!=b['shape'] or a['dtype']!=b['dtype']:return dict(status='misaligned_shape',level=None)
    level='full' if 'blob' in a and 'blob' in b else 'sampled' if a.get('probe_alg')==b.get('probe_alg') and a.get('probe') and b.get('probe') else 'aggregate'
    dtype=a['dtype'].removeprefix('torch.');spec=FLOATS.get(dtype);with_bits=level=='full' and spec is not None
    result=dict(status='ok',level=level,digest_equal=bool(a.get('digest') and a['digest']==b.get('digest')))
    if level=='full' and result['digest_equal']:
        count=math.prod(a['shape']);stats=a.get('stats',{});other=b.get('stats',{})
        finite_stats=stats.get('count')==count and other.get('count')==count and not any(s.get('nan') or s.get('inf') for s in (stats,other))
        no_extremes=not spec or not count or all(s.get('min') is not None and s.get('max') is not None and -spec[2]<s['min']<=s['max']<spec[2] for s in (stats,other))
        if finite_stats and no_extremes:
            for run,event in ((ref_run,a),(cmp_run,b)):
                blob=event['blob']
                if blob['offset']<0 or blob_path(run,blob).stat().st_size<blob['offset']+blob['nbytes']:raise ValueError('invalid tensor blob range')
            zero=stats.get('l2')==0
            result.update(mse=0. if count else None,mae=0. if count else None,max_abs=0. if count else None,
                relative_mean=0. if count else None,relative_max=0. if count else None,relative_l2=0. if count else None,
                cosine_similarity=1. if count and not zero else None,cosine_undefined_reason='empty_tensor' if not count else 'zero_norm' if zero else None,
                valid_elements=count,total_elements=count,statistics_scope='all_elements',nonfinite_pair_mismatches=0,
                paired_sentinel=0,finite_extreme_pair_mismatches=0,max_abs_index=0 if count else None,
                special={side:dict(nan=0,positive_inf=0,negative_inf=0) for side in ('ref','cmp')})
            if with_bits:result.update(differing_elements=0,differing_fraction=0. if count else None,
                ulp_histogram=dict.fromkeys(ULP_BUCKETS,0),ulp_max=0 if count else None,max_abs_ulp=0 if count else None)
            return result
    if level!='aggregate':result.update(metrics(values(ref_run,a,level,with_bits=with_bits),values(cmp_run,b,level,with_bits=with_bits),dtype=dtype,with_bits=with_bits))
    if not with_bits:result['ulp_unavailable_reason']='requires_full_float_pair'
    return result
