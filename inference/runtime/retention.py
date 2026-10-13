"""One post-analysis retention policy; sealed references and inputs stay complete."""
from pathlib import Path
import re
from urllib.parse import unquote

from runtime.common import read_json, write_json, file_hash
from runtime.artifacts import Records


def _inside(root, path):
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return False
    return path.resolve().is_relative_to(root.resolve())


def _states(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _states(child)
    elif isinstance(value, (list, Records)):
        for child in value:
            yield from _states(child)


def _targets(root):
    """Only worker-registered outputs and run-owned raw/cache directories."""
    targets = set()
    workers = {}
    from runtime.capture_index import paths as indexed_paths
    indexed = indexed_paths(root)
    for path in indexed[0] if indexed is not None else root.rglob('result.json'):
        if path == root/'result.json' or not _inside(root, path):
            continue
        parts = path.relative_to(root).parts
        if parts[0] in ('cache', 'source-code') or 'profiler' in parts or 'profiler-recovery' in parts:
            continue
        state = read_json(path)
        names = state.get('artifacts', [])
        for name in names:
            member = Path(name)
            if (member.is_absolute() or '..' in member.parts or member.suffix != '.pt' or
                    path.parent == root/'prepared' or path.parent.is_relative_to(root/'prepared')):
                raise ValueError('unsafe registered retention target: '+str(name))
            tensor = path.parent/member
            if not _inside(root, tensor) or not tensor.is_file():
                raise ValueError('missing or unsafe registered tensor: '+str(tensor))
            targets.add(tensor)
        if any(row.get('artifacts') for row in _states(state)):
            workers[path] = state
    raw_paths = (path for directory in indexed[1] for path in directory.rglob('*')) if indexed is not None else root.rglob('*')
    for path in raw_paths:
        if not _inside(root, path):
            continue
        parts = path.relative_to(root).parts
        if (parts[0] == 'cache' or 'profiler' in parts or 'profiler-recovery' in parts):
            if path.is_file():
                targets.add(path)
    return sorted(targets), workers


def _project_artifacts(state, folder, root, removed):
    names = state.get('artifacts', [])
    discarded = [name for name in names if (folder/name).relative_to(root).as_posix() in removed]
    if discarded:
        state.update(artifacts=[name for name in names if name not in discarded],
                     discarded_artifacts=discarded, tensor_retention='metrics_only')
    for rank, row in state.get('ranks', {}).items():
        _project_artifacts(row, folder/('rank-'+str(rank)), root, removed)


def _availability(root):
    tensors = False
    traces = False
    from runtime.capture_index import paths as indexed_paths
    indexed = indexed_paths(root)
    if indexed is not None:
        from itertools import chain
        candidates = chain(indexed[0], (path for directory in indexed[1] if directory != root/'cache' for path in directory.rglob('*')))
    else:
        candidates = root.rglob('*')
    for path in candidates:
        if not _inside(root, path) or not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if 'profiler' in parts or 'profiler-recovery' in parts:
            traces = True
        if (path.name == 'result.json' and path != root/'result.json' and
                parts[0] not in ('cache', 'source-code') and
                'profiler' not in parts and 'profiler-recovery' not in parts):
            try:
                state = read_json(path)
            except (ValueError, OSError):
                continue
            tensors |= any((path.parent/name).is_file() and _inside(root, path.parent/name)
                           for name in state.get('artifacts', [])
                           if not Path(name).is_absolute() and '..' not in Path(name).parts)
    return tensors, traces


def _save(root, result, ledger, command):
    result['retention'] = ledger
    write_json(root/'retention.json', ledger)
    if command == 'accuracy':
        status = ('metrics_only' if ledger['mode'] == 'compact' else
                  'kept_requested' if ledger['reason'] == 'requested' else
                  'kept_anomalies' if ledger['reason'] == 'numerical anomalies' else 'kept_failure')
        if ledger['status'] == 'cleanup_partial':
            status = 'cleanup_partial'
        compatibility = {'status': status, 'reason': ledger['reason'],
                         'files_removed': len(ledger['removed']),
                         'bytes_removed': sum(row['bytes'] for row in ledger['removed'])}
        if result.get('oracle', {}).get('role') == 'consumer':
            result['oracle']['retention'] = compatibility
        write_json(root/'tensor-retention.json', dict(compatibility, files=ledger['removed']))
    write_json(root/'result.json', result)
    if (root/'report-source.json').is_file():
        source = read_json(root/'report-source.json')
        source['result_sha256'] = file_hash(root/'result.json')
        write_json(root/'report-source.json', source)


def finalize_retention(cfg, root, result, *, command=None):
    root = Path(root)
    command = command or result.get('command')
    if command not in ('accuracy', 'performance', 'oracle'):
        return
    previous = result.get('retention', {})
    if previous.get('mode') == 'compact' and previous.get('status') == 'complete':
        return
    producer = (command == 'oracle' or cfg.get('_performance_capture') or
                result.get('oracle', {}).get('role') == 'producer')
    mode = 'accuracy' if command == 'oracle' else command
    keep = cfg.get(mode, {}).get('keep_tensors' if mode == 'accuracy' else 'keep_evidence', False)
    reason = ('reference producer' if producer else 'requested' if keep else
              'execution incomplete' if result.get('status') != 'completed' else
              'analysis incomplete' if result.get('analysis_status') != 'completed' else
              'numerical anomalies' if any(row.get('numerical_anomalies') for row in _states(result)) else
              'measurement quality requires review' if result.get('measurement_quality', {}).get('status')
                  in ('needs_review', 'invalid_evidence') else None)
    tensors, traces = _availability(root) if reason else (False, False)
    ledger = {'mode': 'full' if reason else 'compact', 'status': 'complete',
              'reason': reason or 'completed capture, analysis and report', 'removed': [],
              'reanalysis_available': traces, 'tensors_available': tensors}
    if reason:
        _save(root, result, ledger, command)
        if (root/'report.md').is_file():
            from runtime.reporting import append_retention
            append_retention(root, result)
        return
    # Production validates all forward workers before reporting. Repeat accuracy
    # validation here so altered/missing checks never permit destructive cleanup.
    try:
        if cfg.get('_forward_checks_required'):
            forwards = [row for row in _states(result) if 'forward_evidence' in row]
            if not forwards or any(not row.get('forward_evidence_validated') for row in forwards):
                raise ValueError('forward checks are incomplete or unvalidated')
            if command == 'accuracy':
                from runtime.preview_evidence import validate_forward_checks
                expected_ranks = ({str(r) for r in range(len(cfg['runtime']['devices']))}
                                  if cfg['runtime'].get('parallelism') == 'tp' else {'single'})
                for side, state in result['paths'].items():
                    ranks = state.get('ranks') or {'single': state}
                    if set(ranks) != expected_ranks:
                        raise ValueError('accuracy rank coverage is incomplete')
                    for rank, row in ranks.items():
                        folder = root/side if rank == 'single' else root/side/('rank-'+rank)
                        validate_forward_checks(cfg, folder, row, root/'prepared')
        targets, workers = _targets(root)
        entries = [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
                    'sha256': file_hash(p)} for p in targets]
    except (Exception, KeyboardInterrupt):
        tensors, traces = _availability(root)
        ledger.update(mode='full', status='kept_failure', reason='retention evidence incomplete',
                      reanalysis_available=traces, tensors_available=tensors)
        _save(root, result, ledger, command)
        raise
    ledger.update(status='registered', planned=entries)
    # Render the final presentation before deletion; even a second renderer
    # failure leaves every original tensor, trace and cache in place.
    markdown = {p: p.read_bytes() for p in root.glob('*.md')}
    _save(root, result, ledger, command)
    try:
        from runtime.reporting import render
        render(root, result)
        update_retained_links(root, entries)
    except (Exception, KeyboardInterrupt):
        for path, content in markdown.items():
            path.write_bytes(content)
        tensors, traces = _availability(root)
        ledger.update(mode='full', status='kept_failure', reason='final compact report failed',
                      reanalysis_available=traces, tensors_available=tensors)
        ledger.pop('planned', None)
        _save(root, result, ledger, command)
        raise
    try:
        for path, entry in zip(targets, entries):
            if not _inside(root, path) or file_hash(path) != entry['sha256']:
                raise ValueError('retention target changed: '+entry['path'])
            path.unlink()
            ledger['removed'].append(entry)
            # The registered list survives a killed process; intermediate
            # checkpoints bound journal write amplification for large caches.
            if len(ledger['removed']) % 32 == 0:
                write_json(root/'retention.json', ledger)
    except BaseException:
        ledger.update(mode='partial', status='cleanup_partial', reason='cleanup interrupted',
                      reanalysis_available=False, tensors_available=False)
        _save(root, result, ledger, command)
        raise
    try:
        removed = {entry['path'] for entry in ledger['removed']}
        for path, state in workers.items():
            _project_artifacts(state, path.parent, root, removed)
            write_json(path, state)
        for state in _states(result):
            if state.get('artifacts'):
                state['discarded_artifacts'] = state['artifacts']
                state['artifacts'] = []
                state['tensor_retention'] = 'metrics_only'
        # Update TP parent files after rank projections.
        for side, state in result.get('paths', {}).items():
            if command == 'accuracy' and state.get('ranks'):
                write_json(root/side/'result.json', state)
        ledger.update(status='complete')
        ledger.pop('planned', None)
        _save(root, result, ledger, command)
    except BaseException:
        ledger.update(mode='partial', status='cleanup_partial', reason='cleanup metadata incomplete',
                      reanalysis_available=False, tensors_available=False)
        _save(root, result, ledger, command)
        raise


def update_retained_links(root, entries=None):
    """Suppress removed-file links, including aliases inside a TP run."""
    root = Path(root)
    ledger = read_json(root/'retention.json') if (root/'retention.json').is_file() else {}
    entries = ledger.get('removed', []) if entries is None else entries
    removed = {(root/row['path']).resolve() for row in entries}
    if not removed:
        return
    for path in root.glob('*.md'):
        def replace(match):
            target = unquote(match.group(2).split('#')[0])
            if not target or '://' in target or target.startswith('#'):
                return match.group(0)
            resolved = (root/target).resolve()
            if not resolved.is_relative_to(root.resolve()):
                return match.group(0)
            directory_removed = any(p.is_relative_to(resolved) for p in removed) and (
                not resolved.exists() or (resolved.is_dir() and not any(
                    p.resolve() not in removed for p in resolved.rglob('*') if p.is_file())))
            if resolved in removed or directory_removed:
                return match.group(1)+'（原始证据已清理，参见 retention.json）'
            return match.group(0)
        path.write_text(re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace, path.read_text()))
