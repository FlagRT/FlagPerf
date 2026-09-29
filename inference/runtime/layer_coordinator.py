# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Execute isolated layer sampling workers and preserve partial evidence."""
import os
from runtime.common import read_json, write_json
from analysis.layer import ranks, summarize_layer, overhead

def collect(cfg, root, result, contexts, prepared, policy, sides, progress, invoke):
    layer = {'schema_version': 1, 'selection': cfg['performance']['layers'], 'runs': {},
             'world_size': len(cfg['runtime']['devices']) if cfg['runtime'].get('parallelism') == 'tp' else 1,
             'profiles': {}, 'routes': {}, 'failure_count': 0}
    result['layer'] = layer
    for phase in ('layer_timing', 'layer_profile', 'layer_memory'):
        layer['runs'][phase] = {}
        for repeat in range(cfg['performance']['repeats']):
            for side in sides if repeat % 2 == 0 else reversed(sides):
                label = f'repeat-{repeat:02}'
                ctx = contexts[side] if contexts else {'cfg': cfg, 'prepared': prepared,
                        'include': policy['include'] if side == 'on' else None}
                destination = root/'layer'/phase/side/label
                progress.phase = f'{phase}/{side}/{label}'
                groups = len(cfg['performance']['layers']) if phase == 'layer_memory' else 1
                value = invoke(ctx['cfg'], root, phase, destination, ctx['prepared'], ctx['include'],
                               repeat_index=repeat, timeout=cfg['runtime']['timeout_seconds']*groups)
                layer['runs'][phase].setdefault(side, {})[label] = value
                if value['status'] != 'completed':
                    layer['failure_count'] += 1
                    result.update(status='failed' if phase == 'layer_timing' else 'partial',
                                  error=f'{phase}/{side}/{label}: {value.get("error", "incomplete")}')
                    if phase == 'layer_timing':
                        layer['timing'] = summarize_layer(layer)
                        write_json(root/'layer'/'summary.json', layer)
                        return
                    continue
                if contexts:
                    from runtime.stack import assert_observation
                    assert_observation(ctx, value)
                if phase == 'layer_profile':
                    from analysis.layer_trace import analyze, associate_collectives
                    profile = {}
                    for rank, state in ranks(value).items():
                        folder = destination/('rank-'+rank) if layer['world_size'] > 1 else destination
                        try:
                            detail = analyze(folder, state)
                            if layer['world_size'] > 1:
                                from analysis.communication import analyze_rank
                                detail['communication'] = analyze_rank(folder, int(rank))
                                associate_collectives(detail)
                        except Exception as error:
                            detail = {'status': 'partial', 'reason': str(error), 'rows': [], 'events': [], 'coverage': {}}
                        write_json(folder/'attribution.json', detail)
                        # Event map remains in its own file; summary carries layer rows and coverage.
                        profile[rank] = {k: v for k, v in detail.items() if k not in ('events','auxiliary_events')}
                        profile[rank]['evidence'] = str((folder/'attribution.json').relative_to(root))
                        if detail['status'] != 'completed' or detail.get('communication', {}).get('status', 'completed') != 'completed':
                            result['status'] = 'partial'
                    layer['profiles'].setdefault(side, {})[label] = profile
                    if layer['world_size'] > 1:
                        alias = root/'profiles'/side/label
                        alias.parent.mkdir(parents=True, exist_ok=True)
                        alias.symlink_to(os.path.relpath(destination, alias.parent), target_is_directory=True)
                        from analysis.communication import analyze as analyze_communication
                        communication = analyze_communication(destination, layer['world_size'])
                        result.setdefault('profiles', {}).setdefault(side, {})[label] = {
                            'status': communication['status'], 'communication': communication}
    layer['timing'] = summarize_layer(layer)
    layer['overhead'] = overhead(result)
    from runtime.coordinator import route_summary
    for side, audit in result.get('component_audits', result.get('audits', {})).items():
        native_path = root/'audit'/side/'native'/'result.json'
        native = read_json(native_path) if native_path.is_file() else audit
        layer['routes'][side] = {}
        for rank, state in ranks(audit).items():
            native_rank = ranks(native).get(rank, {})
            selected_policy = contexts[side].get('policy') if contexts else policy
            rows = {}
            for name, row in state.get('layer_routes', {}).items():
                rows[name] = dict(row, hardware_kernel_fallback_ratio=None)
                if selected_policy:
                    rows[name]['classification'] = route_summary(
                        {'candidates': native_rank.get('candidates', [])},
                        {'inventory': row['inventory'], 'route': {'actual_function_calls': row['actual_function_calls']}},
                        selected_policy)
            layer['routes'][side][rank] = rows
    if layer['timing']['status'] != 'completed': result['status'] = 'partial'
    layer['status'] = result['status']
    write_json(root/'layer'/'summary.json', layer)
