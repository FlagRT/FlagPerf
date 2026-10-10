"""Select an explicit decoder; mixed or unknown captures remain partial evidence."""
from pathlib import Path
import importlib
from runtime.common import ROOT, read_json
from runtime.adapters import descriptor


def decoder(folder):
    folder = Path(folder)
    capture = folder / 'trace-capture.json'
    if capture.is_file():
        metadata = read_json(capture)
        entry = descriptor('vendors', metadata.get('vendor'))
        if (metadata.get('schema_version') != 1 or not entry.get('decoder') or
                metadata.get('version') not in entry['capabilities'].get('trace_versions', {}).get(metadata.get('format'), [])):
            raise ValueError('unsupported trace format/version')
        for path in sorted((ROOT/'registry/vendors').glob('*.json')):
            if path.stem == metadata['vendor']: continue
            other = descriptor('vendors',path.stem)
            if any(any(folder.glob(pattern)) for pattern in other.get('capture_patterns',[])):
                raise ValueError('mixed vendor trace formats; attribution is partial')
        selected = importlib.import_module(entry['decoder'])
        return selected
    matches = []
    for path in sorted((ROOT / 'registry/vendors').glob('*.json')):
        entry = descriptor('vendors', path.stem)
        if not entry.get('decoder'): continue
        adapter = importlib.import_module(entry['decoder'])
        if adapter.has_capture(folder):
            matches.append(path.stem)
    if len(matches) > 1:
        raise ValueError('mixed vendor trace formats; attribution is partial')
    if len(matches) == 1:
        # Schema-1 runs predate trace-capture.json. Only unambiguous saved formats qualify.
        return importlib.import_module(descriptor('vendors', matches[0])['decoder'])
    raise ValueError('no unambiguous supported vendor trace; attribution is partial')
