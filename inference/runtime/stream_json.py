"""Incremental standard JSON decoding, including strict trailing/truncation checks."""
import json
import re

_WHITESPACE = re.compile(r"[ \t\r\n]*")


class Reader:
    def __init__(self, stream, chunk_bytes=1024*1024):
        self.stream, self.chunk_bytes = stream, chunk_bytes
        self.buffer, self.position, self.eof = '', 0, False
        self.decoder = json.JSONDecoder()

    def fill(self):
        self.buffer = self.buffer[self.position:]
        self.position = 0
        chunk = self.stream.read(self.chunk_bytes)
        self.buffer += chunk
        self.eof = not chunk
        return bool(chunk)

    def peek(self):
        while True:
            self.position = _WHITESPACE.match(self.buffer, self.position).end()
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof or not self.fill():
                return ''

    def take(self, char):
        if self.peek() != char:
            raise ValueError('invalid or truncated JSON; expected ' + char)
        self.position += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
                # Numbers may end at a chunk boundary: fetch their delimiter first.
                if end == len(self.buffer) and not self.eof:
                    self.fill(); continue
                if type(value) in (int, float) and end < len(self.buffer) and self.buffer[end] not in ',]} \t\r\n:':
                    if not self.eof:
                        self.fill(); continue
                    raise ValueError('invalid JSON number')
                self.position = end
                return value
            except json.JSONDecodeError:
                if self.eof or not self.fill():
                    raise ValueError('invalid or truncated JSON value') from None

    def array(self):
        self.take('[')
        if self.peek() != ']':
            while True:
                yield self.value()
                if self.peek() == ']': break
                self.take(',')
        self.take(']')

    def structure(self, store=None, threshold=1024):
        """Decode containers incrementally, spilling long record arrays when requested."""
        kind = self.peek()
        if kind == '{':
            self.take('{'); value = {}
            if self.peek() != '}':
                while True:
                    key = self.value()
                    if not isinstance(key, str): raise ValueError('invalid JSON object key')
                    self.take(':')
                    value[key] = self.structure(store, threshold)
                    if self.peek() == '}': break
                    self.take(',')
            self.take('}'); return value
        if kind == '[':
            self.take('['); value, sink = [], None
            if self.peek() != ']':
                while True:
                    # Record objects are decoded by the C JSON decoder. Large
                    # arrays spill at this level, without Python walks per field.
                    item = self.value()
                    if sink is not None and not isinstance(item, dict):
                        value, sink = list(sink.finish()), None
                    if sink is None:
                        value.append(item)
                        if store is not None and len(value) == threshold and all(isinstance(row, dict) for row in value):
                            from runtime.artifacts import RecordWriter
                            sink = RecordWriter(store); sink.extend(value); value = []
                    else: sink.append(item)
                    if self.peek() == ']': break
                    self.take(',')
            self.take(']')
            return sink.finish() if sink is not None else value
        return self.value()


def read_document(path, store=None):
    with open(path, encoding='utf-8') as stream:
        reader = Reader(stream)
        value = reader.structure(store)
        if reader.peek(): raise ValueError('trailing JSON content')
        return value


def trace_events(stream, metadata=None):
    reader = Reader(stream)
    reader.take('{')
    seen = False
    if reader.peek() != '}':
        while True:
            key = reader.value()
            if not isinstance(key, str): raise ValueError('invalid JSON object key')
            reader.take(':')
            if key == 'traceEvents':
                if seen: raise ValueError('duplicate traceEvents')
                seen = True
                for event in reader.array():
                    if not isinstance(event, dict): raise ValueError('invalid CUDA trace event')
                    yield event
            else:
                value = reader.value()
                if metadata is not None: metadata[key] = value
            if reader.peek() == '}': break
            reader.take(',')
    reader.take('}')
    if reader.peek() or not seen:
        raise ValueError('trailing JSON content or missing traceEvents')
