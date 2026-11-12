# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Adapted-model registry for the training tasks.

Reads ``inference/targets/model_registry.json`` so users can omit
``--pooling`` / ``--rerank-score`` when they name an already-adapted model.
Lookup matches the model directory name exactly, then the longest matching
prefix rule. An explicit CLI flag always wins and the origin of the chosen
value is returned so the training protocol can record it faithfully.
"""
import json
from pathlib import Path

POOLS = ('last', 'cls', 'mean')
SCORES = ('causal_yesno', 'sequence_logit')
TASKS = ('embedding', 'rerank')
REGISTRY_PATH = Path(__file__).resolve().parents[2] / 'targets' / 'model_registry.json'


def load_registry(path=None):
    """Parse and validate the registry file; raises ValueError on bad entries."""
    stream = Path(path) if path else REGISTRY_PATH
    data = json.loads(stream.read_text(encoding='utf-8'))
    if data.get('schema') != 'model-registry-v1':
        raise ValueError('registry schema must be model-registry-v1')
    for name, entry in data.get('entries', {}).items():
        _validate_entry(entry, context='entry %r' % name)
    for rule in data.get('prefixes', []):
        if not rule.get('prefix'):
            raise ValueError('prefix rules need a non-empty prefix')
        _validate_entry(rule, context='prefix %r' % rule.get('prefix'))
    return data


def _validate_entry(entry, *, context):
    task = entry.get('task')
    if task not in TASKS:
        raise ValueError('%s: unsupported task %r' % (context, task))
    if task == 'embedding' and entry.get('pooling') not in POOLS:
        raise ValueError('%s: embedding entries need pooling in %s' % (context, ', '.join(POOLS)))
    if task == 'rerank' and entry.get('rerank_score') not in SCORES:
        raise ValueError('%s: rerank entries need rerank_score in %s' % (context, ', '.join(SCORES)))


def lookup(source, registry=None):
    """Return the registry record for a model path or None when not adapted.

    The directory name is matched exactly first; prefix rules are applied
    longest-first afterwards. `hf:` prefixes are accepted and ignored.
    """
    data = registry if registry is not None else load_registry()
    name = str(source).removeprefix('hf:').rstrip('/').rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
    entry = data.get('entries', {}).get(name)
    if entry:
        return dict(entry, matched=name)
    for rule in sorted(data.get('prefixes', []), key=lambda r: len(r['prefix']), reverse=True):
        if name.startswith(rule['prefix']):
            return dict(rule, matched=rule['prefix'])
    return None


def resolve(value, source, *, kind, registry=None):
    """Explicit CLI value wins, then the registry default, else None.

    kind is 'pooling' or 'rerank_score'; returns (value, origin) with origin
    'cli' or 'registry' (or (None, None) so callers can raise a helpful error).
    """
    if value:
        return value, 'cli'
    entry = lookup(source, registry)
    if entry and entry.get('task') in ('embedding', 'rerank') and entry.get(kind):
        return entry[kind], 'registry'
    return None, None
