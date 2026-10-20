# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Independent P800 XPU CUPTI capture; never used as clean measurement."""
from decimal import Decimal
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from runtime.evidence import read, write, sha
from runtime.progress import publish_profile

MARKER = 'operation_target_'
COUNTER_GROUPS = ('ArithmeticUtilization', 'PipeUtilization', 'Memory',
                  'MemoryL0', 'MemoryUB', 'ResourceConflictRatio')


def number(value):
    if isinstance(value, bool):
        raise ValueError('boolean timestamp')
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError('non-finite timestamp')
    return result


def union_duration(intervals):
    end = None
    total = Decimal(0)
    for lo, hi in sorted(intervals):
        total += max(Decimal(0), hi - max(lo, end if end is not None else lo))
        end = max(hi, end if end is not None else hi)
    return float(total)


def analyze_events(events, device=0, *, count=5, marker=MARKER):
    """Require host launch + explicit ac2g flow + matching device correlation.

    Host windows select host launches only. Device timestamps are never compared
    with host timestamps, and CUDA-shaped occupancy fields are not promoted.
    """
    partial = lambda reason: {'status': 'partial', 'reason': reason, 'kernel_count': None}
    try:
        if not isinstance(events, list) or any(not isinstance(e, dict) for e in events):
            return partial('traceEvents must be a list of event objects')
        ranges = []
        for i in range(count):
            matches = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'user_annotation'
                       and e.get('name') == marker + str(i)]
            if len(matches) != 1:
                return partial('requires unique complete target ranges')
            e = matches[0];lo = number(e['ts']);dur = number(e['dur'])
            if dur <= 0:
                return partial('invalid target duration')
            ranges.append((e, lo, lo + dur))
        if any(ranges[i][2] > ranges[i+1][1] for i in range(count-1)):
            return partial('overlapping or unordered target ranges')
        # Autograd may launch on a separate host thread. Require an explicit
        # fwdbwd flow from a CPU forward op inside the target to the containing
        # backward CPU op; temporal overlap alone is insufficient.
        backward_ranges = []
        for finish in events:
            if finish.get('cat') != 'fwdbwd' or finish.get('ph') != 'f':continue
            sources = [e for e in events if e.get('cat') == 'fwdbwd' and e.get('ph') == 's'
                       and e.get('id') == finish.get('id') and e.get('pid') == finish.get('pid')]
            finishes = [e for e in events if e.get('cat') == 'fwdbwd' and e.get('ph') == 'f'
                        and e.get('id') == finish.get('id') and e.get('pid') == finish.get('pid')]
            if finish.get('id') is None or len(sources) != 1 or len(finishes) != 1:continue
            source = sources[0]
            forwards = [e for e in events if e.get('cat') == 'cpu_op' and e.get('ph') == 'X'
                        and (e.get('pid'),e.get('tid'),number(e['ts'])) == (source.get('pid'),source.get('tid'),number(source['ts']))]
            backwards = [e for e in events if e.get('cat') == 'cpu_op' and e.get('ph') == 'X'
                         and (e.get('pid'),e.get('tid'),number(e['ts'])) == (finish.get('pid'),finish.get('tid'),number(finish['ts']))]
            if len(forwards) != 1 or len(backwards) != 1:continue
            f,b = forwards[0],backwards[0]
            sequence = f.get('args',{}).get('Sequence number')
            backward_sequence = b.get('args',{}).get('Sequence number')
            if type(sequence) is not int or type(backward_sequence) is not int or sequence != backward_sequence:continue
            for i,(r,lo,hi) in enumerate(ranges):
                if ((f.get('pid'),f.get('tid')) == (r.get('pid'),r.get('tid'))
                        and lo <= number(f['ts']) <= number(f['ts'])+number(f['dur']) <= hi
                        and lo <= number(b['ts']) <= number(b['ts'])+number(b['dur']) <= hi):
                    backward_ranges.append((i,b,number(b['ts']),number(b['ts'])+number(b['dur'])))
        launches = {}
        for e in events:
            if e.get('ph') != 'X' or e.get('cat') != 'cuda_runtime' or e.get('name') != 'cudaLaunchKernel':
                continue
            lo = number(e['ts']);duration = number(e['dur']);hi = lo + duration
            if duration <= 0:
                return partial('invalid host launch duration')
            matches = [i for i,(r,a,b) in enumerate(ranges)
                       if (e.get('pid'),e.get('tid')) == (r.get('pid'),r.get('tid')) and a <= lo <= hi <= b]
            matches += [i for i,r,a,b in backward_ranges if (e.get('pid'),e.get('tid')) == (r.get('pid'),r.get('tid')) and a <= lo <= hi <= b]
            matches = sorted(set(matches))
            if len(matches) != 1:
                return partial('launch outside a unique target range')
            corr = e.get('args',{}).get('correlation')
            if not isinstance(corr, int) or isinstance(corr, bool) or corr in launches:
                return partial('missing or ambiguous launch correlation')
            flows = [f for f in events if f.get('cat') == 'ac2g' and f.get('ph') == 's'
                     and f.get('id') == corr and (f.get('pid'),f.get('tid')) == (e.get('pid'),e.get('tid'))
                     and number(f['ts']) == lo]
            if len(flows) != 1:
                return partial('missing or ambiguous host-to-device flow source')
            launches[corr] = (matches[0], e)
        kernels = []
        seen = set()
        for e in events:
            if e.get('ph') != 'X' or e.get('cat') != 'kernel':
                continue
            args = e.get('args', {});corr = args.get('correlation')
            if type(corr) is not int or corr not in launches or type(args.get('device')) is not int or args.get('device') != device:
                return partial('unattributed kernel or wrong device')
            lo = number(e['ts']);dur = number(e['dur'])
            if dur <= 0:
                return partial('invalid kernel duration')
            flows = [f for f in events if f.get('cat') == 'ac2g' and f.get('ph') == 'f'
                     and f.get('id') == corr and (f.get('pid'),f.get('tid')) == (e.get('pid'),e.get('tid'))
                     and number(f['ts']) == lo]
            if len(flows) != 1:
                return partial('missing or ambiguous host-to-device flow destination')
            key = (corr, e.get('pid'), e.get('tid'), lo)
            if key in seen:
                return partial('duplicate correlated device event')
            seen.add(key)
            call, host = launches[corr]
            kernels.append({'name': e.get('name'), 'call': call, 'start': lo, 'duration': dur,
                            'correlation': corr, 'device': device, 'stream': args.get('stream'),
                            'pid': e.get('pid'), 'tid': e.get('tid')})
        if {k['call'] for k in kernels} != set(range(count)) or {k['correlation'] for k in kernels} != set(launches):
            return partial('not every target call and launch has a correlated device kernel')
        origin = min(k['start'] for k in kernels)
        calls = []
        for i in range(count):
            kk = [k for k in kernels if k['call'] == i]
            spans = [(k['start'],k['start']+k['duration']) for k in kk]
            calls.append({'kernel_count': len(kk), 'kernel_sum_us': float(sum(k['duration'] for k in kk)),
                          'device_union_us': union_duration(spans),
                          'device_span_us': float(max(b for a,b in spans)-min(a for a,b in spans))})
        for k in kernels:
            k['ts_us'] = float(k.pop('start')-origin)
            k['duration_us'] = float(k.pop('duration'))
        return {'status':'measured', 'kernel_count':len(kernels), 'calls':calls, 'kernels':kernels,
                'clock_origin_us':str(origin), 'timestamp_scope':'device microseconds relative to clock_origin_us',
                'boundary':'XPU CUPTI kernel events joined by correlation and ac2g flow; cross-thread backward launches additionally require fwdbwd flow and matching sequence within five synchronized ranges; not proof of all semantic components or absence of CPU fallback',
                'discarded_fields':'CUDA-compatible grid/block/occupancy/register/SM metadata is not validated for P800 and is not a hardware counter result'}
    except (KeyError, TypeError, ValueError, ArithmeticError, AttributeError) as exc:
        return partial('invalid trace: ' + str(exc))


def target_window(adapter, root, task, call):
    import torch
    if os.environ.get('XPU_CUPTI_ENABLE_DEVICE') != str(adapter.device.index):
        raise RuntimeError('profiler device does not match the UUID-bound logical device')
    for _ in range(task['case_config']['WARMUP']):
        out = call();del out
    adapter.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as prof:
        for i in range(5):
            with torch.profiler.record_function(MARKER+str(i)):
                out = call()
                adapter.synchronize()
                del out
    prof.export_chrome_trace(str(root/'trace.json'))
    write(root/'profile-window.json', {'calls':5, 'marker':MARKER, 'device':adapter.device.index,
          'boundary':'independent synchronized replays after warmup; original invocation and saved inputs; not clean timing',
          'environment':{k:os.environ.get(k) for k in ('XPU_CUPTI_ENABLE_DEVICE','XPU_ENABLE_PROFILER_TRACING')}})


def collect(root, task):
    began = time.monotonic()
    folder = root/'profiling/timeline'
    folder.mkdir(parents=True, exist_ok=False)
    for name in ('inputs.pt','reference.pt','correctness.json','task.json'):
        if (root/name).is_file():shutil.copyfile(root/name,folder/name)
    command = [sys.executable,'-B','-S',str(Path(__file__).with_name('bootstrap.py')),
               '--root',str(folder),'--phase','profile-target']
    write(folder/'command.json',{'argv':command})
    result = {'protocol':'p800-cupti-profile-v1','mode':task['profiling_mode'],'status':'partial',
              'collector_sha256':sha(Path(__file__)), 'groups':[],
              'boundary':'independent device timeline; CUDA-compatible occupancy fields are not trusted P800 counters; route and exclusive performance gates remain independent'}
    entry = {'group':'timeline','status':'partial','directory':'profiling/timeline'}
    publish_profile(root,'timeline',1,1,'capturing',0)
    # Keep the child in its parent's process group: local watchdog/SIGTERM and
    # Docker cleanup then cover it too; no detached descendants.
    with (folder/'capture.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            proc.wait(timeout=max(1,task.get('watchdog_s',300)-20))
        except subprocess.TimeoutExpired:
            entry['reason']='profile target watchdog exceeded'
            result['execution_error']=True
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=3)
    entry['returncode']=proc.returncode
    if proc.returncode or (folder/'profile-target-error.json').exists():
        entry.setdefault('reason','profile target execution failed; inspect capture.log')
        result['execution_error']=True
    elif not all((folder/n).is_file() for n in ('trace.json','profile-window.json','profile-target-identity.json')):
        entry['reason']='missing completed target capture or identity'
    else:
        identity=read(folder/'profile-target-identity.json')
        measured=read(root/'measurement.json')
        keys=('packages','case_sha256','input_sha256','registration','runtime_config','device','input_devices')
        mismatches=[k for k in keys if k not in identity or k not in measured or identity[k]!=measured[k]]
        # Profiler environment differs intentionally; physical binding, hooks and
        # interpreter must still match the clean worker exactly.
        runtime_keys=('framework_observation','packages','executable','hook_hashes','profile_sha256')
        mismatches += ['vendor_runtime.'+k for k in runtime_keys if identity.get('vendor_runtime',{}).get(k) != measured.get('vendor_runtime',{}).get(k)
                       or k not in identity.get('vendor_runtime',{})]
        if identity.get('input_sha256') != sha(root/'inputs.pt'):mismatches.append('saved_input')
        result['phase_consistency']={'status':'failed' if mismatches else 'passed','mismatches':mismatches}
        result['identity']=identity
        trace=read(folder/'trace.json')
        timeline=analyze_events(trace.get('traceEvents',[]),read(folder/'profile-window.json')['device'])
        timeline['trace']='trace.json'
        timeline['trace_sha256']=sha(folder/'trace.json')
        result['tool']={k:trace.get(k) for k in ('schemaVersion','cupti_version','cuda_runtime_version','cuda_driver_version')}
        result['tool']['boundary']='reported compatibility API versions; actual P800 stack identity is recorded separately'
        entry.update(status=timeline['status'],timeline=timeline)
        if mismatches:
            # Preserve the raw capture on disk, but do not expose timing values
            # from another identity as this task's metrics, including in reports.
            entry.update(status='partial',reason='profile replay identity differs from clean measurement',
                         timeline={'status':'partial','reason':'profile replay identity differs from clean measurement',
                                   'trace':'trace.json','trace_sha256':timeline['trace_sha256']})
    result['groups'].append(entry)
    if task['profiling_mode']=='full':
        for name in COUNTER_GROUPS:
            result['groups'].append({'group':name,'status':'not-supported',
                                    'reason':'no validated P800 hardware counter collector or semantic mapping; CUDA trace metadata is not a replacement'})
        result['reason']='timeline attempted; six hardware counter groups remain unsupported'
    result['status']='measured' if all(g['status']=='measured' for g in result['groups']) else 'partial'
    result['elapsed_s']=time.monotonic()-began
    write(root/'profiling.json',result)
    publish_profile(root,'timeline',1,1,entry['status'],result['elapsed_s'])
