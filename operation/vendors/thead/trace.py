"""Strict PPU host launch / flow / device event correlation."""
from decimal import Decimal
from collections import defaultdict


def is_host_launch(event):
    """Return true for reviewed PPU CUDA-compat kernel launch records.

    Native ATen uses the CUDA runtime API while FlagGems on PPU commonly emits
    the driver API's ``cuLaunchKernelEx``. Native library kernels may instead
    emit ``cuLaunchKernel``. All three must still satisfy the same
    correlation and flow checks below; accepting the driver event does not
    weaken the routing gate.
    """
    return ((event.get('cat'), event.get('name')) in {
        ('cuda_runtime', 'cudaLaunchKernel'),
        ('cuda_driver', 'cuLaunchKernel'),
        ('cuda_driver', 'cuLaunchKernelEx'),
    })


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

def analyze_events(events, device=0, *, count=5, marker='operation_target_'):
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
        # Cold autotuning can emit thousands of launches in a large trace.
        # Index candidate flows once; retain every duplicate and still match
        # the exact timestamp below, rather than scanning all events per launch.
        device_flows = defaultdict(list)
        backward_flows = defaultdict(list)
        for e in events:
            if e.get('ph') not in ('s', 'f'):
                continue
            if e.get('cat') == 'ac2g':
                device_flows[(e.get('ph'), e.get('id'), e.get('pid'), e.get('tid'))].append(e)
            elif e.get('cat') == 'fwdbwd':
                backward_flows[(e.get('ph'), e.get('id'), e.get('pid'))].append(e)
        cpu_ops = defaultdict(list)
        if any(e.get('cat') == 'fwdbwd' and e.get('ph') == 'f' for e in events):
            for e in events:
                if e.get('cat') == 'cpu_op' and e.get('ph') == 'X':
                    cpu_ops[(e.get('pid'), e.get('tid'), number(e['ts']))].append(e)
        # Autograd may launch on a separate host thread. Require an explicit
        # fwdbwd flow from a CPU forward op inside the target to the containing
        # backward CPU op; temporal overlap alone is insufficient.
        backward_ranges = []
        for finish in events:
            if finish.get('cat') != 'fwdbwd' or finish.get('ph') != 'f':continue
            sources = backward_flows[('s', finish.get('id'), finish.get('pid'))]
            finishes = backward_flows[('f', finish.get('id'), finish.get('pid'))]
            if finish.get('id') is None or len(sources) != 1 or len(finishes) != 1:continue
            source = sources[0]
            forwards = cpu_ops[(source.get('pid'), source.get('tid'), number(source['ts']))]
            backwards = cpu_ops[(finish.get('pid'), finish.get('tid'), number(finish['ts']))]
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
            if e.get('ph') != 'X' or not is_host_launch(e):
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
            flows = [f for f in device_flows[('s', corr, e.get('pid'), e.get('tid'))]
                     if number(f['ts']) == lo]
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
            flows = [f for f in device_flows[('f', corr, e.get('pid'), e.get('tid'))]
                     if number(f['ts']) == lo]
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
        observed_calls = {k['call'] for k in kernels}
        observed_correlations = {k['correlation'] for k in kernels}
        if observed_calls != set(range(count)) or observed_correlations != set(launches):
            result = partial('not every target call and launch has a correlated device kernel')
            result['expected_calls'] = list(range(count))
            result['observed_calls'] = sorted(observed_calls)
            result['missing_calls'] = sorted(set(range(count)) - observed_calls)
            result['expected_correlations'] = sorted(launches)
            result['observed_correlations'] = sorted(observed_correlations)
            result['missing_correlations'] = sorted(set(launches) - observed_correlations)
            result['launches_without_device_kernel'] = [
                {'call': launches[corr][0], 'correlation': corr}
                for corr in sorted(set(launches) - observed_correlations)]
            return result
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
                'boundary':'PPU CUDA-compatible profiler kernel events joined by correlation and ac2g flow; cross-thread backward launches additionally require fwdbwd flow and matching sequence within five synchronized ranges; not proof of all semantic components or absence of CPU fallback',
                'discarded_fields':'CUDA-compatible grid/block/occupancy/register/SM metadata is not validated for PPU and is not a hardware counter result'}
    except (KeyError, TypeError, ValueError, ArithmeticError, AttributeError) as exc:
        return partial('invalid trace: ' + str(exc))
