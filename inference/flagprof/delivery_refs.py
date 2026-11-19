"""Resolve immutable evidence references in live trees and relocated deliveries."""
import hashlib
import json
from pathlib import Path


def reference_key(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def resolve(owner, value, *, base=None):
    owner = Path(owner).resolve()
    relative = Path(value)
    local = ((Path(base) if base is not None else owner.parent) / relative).resolve()
    for payload in owner.parents:
        receipt = payload.parent / 'RESTORED.json'
        if payload.name != 'payload' or not receipt.is_file():
            continue
        data = json.loads(receipt.read_text())
        source = owner.relative_to(payload.parent).as_posix()
        mapped = data.get('references', {}).get(source, {}).get(reference_key(value))
        if mapped is not None:
            local = (payload.parent / mapped).resolve()
        elif relative.is_absolute() and data.get('schema') != 'flagprof-delivery/2':
            local = (payload / str(relative).lstrip('/')).resolve()
        if not local.is_relative_to(payload):
            raise ValueError('reference outside restored delivery: ' + source)
        break
    return local
