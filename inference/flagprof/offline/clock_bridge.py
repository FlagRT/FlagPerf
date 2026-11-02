"""Project RAW clock to an already exported trace domain (derived evidence only)."""
from decimal import Decimal, localcontext

REQUEST_RULE = 'generation_request_span_after_initial_sync_through_final_sync/v1'


def locator(stored):
    e = stored.record
    return dict(run_id=stored.run_id, writer_id=stored.writer_id,
                shard_index=stored.shard_index, seq=e.get('seq'), ts=e.get('ts'))


def collect(stored, state):
    e = stored.record
    row = dict(source=locator(stored), writer_id=stored.writer_id, clk=e.get('clk'),
               pid=e.get('pid'), rank=e.get('rank'), start_ns=e.get('ts'))
    if e['t'] == 'meta' and e.get('kind') == 'clock':
        state['clock_samples'].append(dict(row, raw_before_ns=e['data']['mono_ns'],
                                          raw_after_ns=e['ts'], wall_ns=e['data']['wall_ns']))
    elif e['t'] == 'span' and e.get('attrs', {}).get('kind') == 'generation_request':
        state['requests'].append(dict(row, end_ns=e['ts'] + e['dur'], corr=e['corr'], **e['attrs']))
    elif e['t'] == 'span' and e.get('name') == 'generation_step_complete':
        state['request_steps'].append(dict(row, end_ns=e['ts'] + e['dur'],
                                           parent=e.get('parent'), **e['attrs']))


def bridges(inputs):
    """Vendor adapters supply interpreted clock parameters; no vendor I/O here."""
    out = []
    for writer in inputs.get('project_writers', []):
        identity = dict(source_run_id=inputs['source_run_id'], capture=inputs.get('capture_occurrence'),
                        writer_id=writer['writer_id'], rank=writer['rank'],
                        proc_key=writer['proc_key'], clk=writer['clk'])
        samples = [s for s in inputs.get('clock_samples', [])
                   if s['writer_id'] == writer['writer_id'] and s['clk'] == writer['clk']]
        for spec in inputs.get('clock', {}).get('project_transforms', []):
            b = dict(spec, schema='clock-bridge/1', identity=identity,
                     id=str(writer['writer_id']) + ':' + str(spec.get('dev', 'trace')),
                     status='available', error_bound_ns=None, error_bound_status='unavailable',
                     validation_scope='source_formula; independent marker checks belong to acceptance evidence')
            if writer['proc_key'] != inputs.get('capture_process'):
                b.update(status='unavailable', reason='capture_process_identity_missing_or_different')
            if spec['method'] == 'raw_wall_piecewise':
                points = sorted(samples, key=lambda s: s['raw_before_ns'])
                valid = len(points) >= 2 and all(s['raw_before_ns'] <= s['raw_after_ns'] for s in points)
                valid = valid and all(a['raw_after_ns'] < z['raw_before_ns'] and a['wall_ns'] < z['wall_ns']
                                      for a, z in zip(points, points[1:]))
                if not valid:
                    b.update(status='unavailable', reason='missing_or_nonmonotonic_clock_samples')
                else:
                    b.update(points=points,
                             raw_start_ns=(points[0]['raw_before_ns']+points[0]['raw_after_ns']+1)//2,
                             raw_end_ns=(points[-1]['raw_before_ns']+points[-1]['raw_after_ns'])//2,
                             sampling_bracket_max_ns=max(s['raw_after_ns']-s['raw_before_ns'] for s in points),
                             interpolation_assumption='piecewise affine RAW to wall between sampled brackets; no extrapolation')
            out.append(b)
    return out


def map_time(bridge, raw_ns):
    """Round once at the exported nanosecond boundary; never touch device events."""
    with localcontext() as ctx:
        ctx.prec = 50
        if bridge['method'] == 'raw_offset':
            # CANN trans_into_local_time quantizes microseconds to 0.001 us.
            return int((Decimal(raw_ns)/1000 + Decimal(bridge['offset_us'])).quantize(Decimal('.001'))*1000)
        for a, b in zip(bridge['points'], bridge['points'][1:]):
            x0 = a['raw_before_ns']+a['raw_after_ns']
            x1 = b['raw_before_ns']+b['raw_after_ns']
            if x0 <= 2*raw_ns <= x1:
                wall = Decimal(a['wall_ns']) + Decimal(2*raw_ns-x0)*(b['wall_ns']-a['wall_ns'])/(x1-x0)
                return int((wall-bridge['base_time_ns']).to_integral_value())
    raise ValueError('timestamp outside clock anchor range')


def request_windows(inputs, capture_windows):
    all_bridges = bridges(inputs); windows = []; mappings = []
    for request in inputs.get('requests', []):
        identity = dict(request_id=request['request_id'], writer_id=request['writer_id'],
                        rank=request['rank'], pid=request['pid'], clk=request['clk'])
        candidates = [b for b in all_bridges if b['status'] == 'available'
                      and b['identity']['writer_id'] == request['writer_id']
                      and b['identity']['clk'] == request['clk']
                      and b['identity']['rank'] == request['rank']
                      and b['identity']['proc_key']['pid'] == request['pid']
                      and request['clk'] == 'host_mono:'+b['identity']['proc_key']['boot_id']]
        if not candidates:
            mappings.append(dict(identity=identity, status='unavailable', reason='matching_clock_bridge_missing', source=request['source']))
            continue
        steps = sorted((s for s in inputs.get('request_steps', [])
                        if all(s.get(k) == request.get(k) for k in ('request_id','writer_id','rank','pid','clk'))
                        and s.get('parent') == request['corr']), key=lambda s: s['generation_step'])
        complete = bool(inputs.get('capture_normal_close') and steps and steps[-1].get('done')
                        and [s['generation_step'] for s in steps] == list(range(len(steps)))
                        and all(request['start_ns'] <= s['start_ns'] <= s['end_ns'] <= request['end_ns'] for s in steps))
        for bridge in candidates:
            a = max(request['start_ns'], bridge['raw_start_ns'])
            z = min(request['end_ns'], bridge['raw_end_ns'])
            mapping = dict(identity=identity, bridge_id=bridge['id'], source=request['source'],
                           original_start_ns=request['start_ns'], original_end_ns=request['end_ns'],
                           request_complete=complete, completion_source=steps[-1]['source'] if steps else None,
                           boundary_rule=REQUEST_RULE, status='unavailable', reason='outside_clock_anchor_range')
            mappings.append(mapping)
            if z <= a: continue
            left, right = map_time(bridge, a), map_time(bridge, z)
            mapping.update(mapped_start_ns=left, mapped_end_ns=right,
                           mapped_raw_start_ns=a, mapped_raw_end_ns=z, reason='capture_window_missing')
            covers = [w for w in capture_windows if w['kind'] == 'official_capture'
                      and (bridge.get('dev') is None or w.get('dev') in (None, bridge['dev']))]
            for cover in covers:
                start, end = max(left, cover['start_ns']), min(right, cover['end_ns'])
                if end <= start: continue
                full = complete and (a,z) == (request['start_ns'],request['end_ns']) and (start,end) == (left,right)
                status = 'complete' if full else 'partial'
                reason = None if full else 'request_incomplete' if not complete else 'capture_or_anchor_clipped'
                mapping.update(status=status, reason=reason)
                windows.append(dict(kind='project_request', request_id=request['request_id'],
                    rank=request['rank'], dev=cover.get('dev', bridge.get('dev')), start_ns=start, end_ns=end,
                    coverage_status=status, boundary_rule=REQUEST_RULE, bridge_id=bridge['id'],
                    boundary_source=dict(request=request['source'], completion=mapping['completion_source'], capture=cover['boundary_source']),
                    mapping=dict(mapping), window_identity=dict(inputs.get('window_identity',{}),
                        window_kind='project_request', boundary_rule=REQUEST_RULE, coverage_status=status,
                        samples=request.get('samples'),generation_mode=request.get('mode'),generated_steps=len(steps))))
    return all_bridges, mappings, windows
