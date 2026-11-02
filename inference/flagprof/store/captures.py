"""Read-only capture inspection shared by sessions, views and finalization."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any
from .manifest import process_alive


def inspect_capture(directory: str | Path, *, owner_alive: bool | None = None) -> dict[str, Any]:
    data = json.loads((Path(directory) / 'session.json').read_bytes())
    alive = process_alive(data.get('process')) if owner_alive is None else owner_alive
    for window in data['windows']:
        window['effective_state'] = (('lost' if alive is False else 'active' if alive else 'unknown')
                                     if window['state'] in ('collecting', 'exporting') else window['state'])
    return data


def capture_summary(data: dict) -> dict:
    windows = []
    for window in data['windows']:
        state = window['effective_state']
        windows.append(dict(capture_id=window['capture_id'], state=state,
            reasons=window.get('validity', {}).get('reasons', []),
            log_check=window.get('log_check', dict(status='not_applicable' if state=='lost' else 'unavailable',
                reason='window lost' if state=='lost' else 'legacy or not observed'))))
    return dict(session_id=data['session_id'], status=data.get('status','unavailable'), windows=windows,
                log_tail=data.get('log_tail', dict(status='unavailable', reason='legacy or not closed')))


def log_counts(captures: list) -> dict:
    return dict(windows=dict(Counter(w.get('log_check',{}).get('status','unavailable')
                                     for c in captures for w in c.get('windows',[]))),
                tails=dict(Counter(c.get('log_tail',{}).get('status','unavailable') for c in captures)))
