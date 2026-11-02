"""Byte intervals for unmodified Chrome JSON; no vendor or linker imports."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

# Match complete JSON strings in C; structural bytes inside strings are ignored.
_TOKEN = re.compile(rb'"(?:[^"\\]|\\.)*"|[\[\]{}:,]')
_SPACE = b' \r\n\t'


def _end(data: bytes, start: int) -> int:
    if data[start] not in b'[{':
        if data[start] == 34:
            match = _TOKEN.match(data, start)
            if match is None:
                raise ValueError('unterminated JSON string')
            return match.end()
        end = start
        while end < len(data) and data[end] not in b',]} \r\n\t':
            end += 1
        return end
    depth = 0
    for match in _TOKEN.finditer(data, start):
        token = match.group()
        if token in (b'[', b'{'):
            depth += 1
        elif token in (b']', b'}'):
            depth -= 1
            if depth == 0:
                return match.end()
    raise ValueError('unterminated JSON value')


def _space(data: bytes, position: int) -> int:
    while position < len(data) and data[position] in _SPACE:
        position += 1
    return position


def build_index(data: bytes, trace_uid: str) -> dict[str, Any]:
    """Input has already passed json.loads; offsets are UTF-8 bytes, not chars."""
    start = _space(data, 0)
    if data[start:start + 1] == b'{':
        cursor = _space(data, start + 1)
        found = None
        while data[cursor:cursor + 1] != b'}':
            end = _end(data, cursor)
            key = json.loads(data[cursor:end])
            cursor = _space(data, end)
            if data[cursor:cursor + 1] != b':':
                raise ValueError('invalid object separator')
            cursor = _space(data, cursor + 1)
            end = _end(data, cursor)
            if key == 'traceEvents':
                if found is not None:
                    raise ValueError('duplicate traceEvents')
                found = cursor
            cursor = _space(data, end)
            if data[cursor:cursor + 1] == b',':
                cursor = _space(data, cursor + 1)
        if found is None:
            raise ValueError('missing traceEvents')
        start = found
    if data[start:start + 1] != b'[':
        raise ValueError('traceEvents must be an array')
    cursor = _space(data, start + 1)
    entries, seen = [], set()
    while data[cursor:cursor + 1] != b']':
        end = _end(data, cursor)
        record = json.loads(data[cursor:end])
        loc = record.get('_linker_loc', len(entries))
        if str(loc) in seen:
            raise ValueError('duplicate source loc')
        seen.add(str(loc))
        entries.append(dict(loc=loc, ordinal=len(entries), start=cursor, end=end))
        cursor = _space(data, end)
        if data[cursor:cursor + 1] == b',':
            cursor = _space(data, cursor + 1)
    return dict(format_version=1, trace_uid=trace_uid, raw_sha256=hashlib.sha256(data).hexdigest(),
                raw_bytes=len(data), entries=entries)
