"""Pure production import preparation; writes no files and acquires no writer.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from flagprof.correlate.linker import identity, link
from flagprof.correlate.native import relate, coverage as native_coverage
from flagprof.correlate.coverage import coverage
from flagprof.store.manifest import json_bytes
from flagprof.store.schema import SchemaValidator
from .chrome import adapt, stamp_events
from .inventory import enumerate_raw
from .native_common import ScopeConfig
from .format import load_adapter
from .legacy_fixture import preserve_unbound_communication
from .disposition import UnrepresentableError, enumerate_source, reconcile
from .contracts import assess
from .fingerprint import parser_identity


@dataclass
class PreparedImport:
    trace_uid: str
    sha256: str
    events: list[dict[str, Any]]
    devices: list[dict[str, Any]]
    native: list[dict[str, Any]]
    ledger: list[dict[str, Any]]
    disposition: dict[str, Any]
    inventory: dict[str, Any]
    identities: dict[str, Any]
    coverage: dict[str, Any]
    native_coverage: dict[str, Any]
    normalization: dict[str, Any]
    native_rules_version: str
    scope_config: dict[str, Any]
    unbound_calls: list[dict[str, Any]]
    format_assessment: dict[str, Any]


def source_identity(supplied: Mapping[str, Any] | None, vendor: str) -> dict[str, Any]:
    supplied = supplied or {}
    result = {}
    for layer in ('hardware','collector'):
        value = supplied.get(layer)
        result[layer] = (json.loads(json_bytes(value)) if isinstance(value,Mapping) and (value.get('source') or
                         value.get('status')=='unavailable' and isinstance(value.get('reason'),str) and value['reason'])
                         else dict(status='unknown',reason='no sourced '+layer+' identity'))
    result['parser'] = dict(importer='flagprof',version='t8.1-prepare-v1',rule_family=vendor,
                            exporter=supplied.get('exporter',dict(status='unknown',reason='no sourced exporter version')))
    return result


def prepare_import(source: str | Path | bytes, *, vendor: str, calls: Sequence[dict[str, Any]] | None = None,
                   kind: str | None = None, marker_prefix: str | None = None,
                   identities: Mapping[str, Any] | None = None, scope_config: ScopeConfig | None = None,
                   registry: Mapping[str, Any] | None = None, taxonomy_version: str | None = None) -> PreparedImport:
    adapter = load_adapter(vendor)
    taxonomy = adapter.device_taxonomy(taxonomy_version)
    contents = source if isinstance(source,bytes) else Path(source).read_bytes()
    digest = hashlib.sha256(contents).hexdigest(); uid=digest[:16]
    fixture=json.loads(contents)
    original=fixture if isinstance(fixture,list) else fixture['traceEvents']
    if not isinstance(original,list) or not all(isinstance(e,dict) for e in original):
        raise ValueError('traceEvents must be an array of records')
    calls=json.loads(json_bytes(list(calls if calls is not None else [])))
    effective_kind=kind if kind is not None else 'all'
    prefix=marker_prefix if marker_prefix is not None else None
    if effective_kind not in ('compute','communication','all') or (prefix is not None and (not isinstance(prefix,str) or not prefix)):
        raise ValueError('invalid normalization kind/marker_prefix')
    events=stamp_events(original)
    inventory=enumerate_raw(events,uid,taxonomy=taxonomy)
    dispositions=importlib.import_module('flagprof.vendors.'+vendor+'.disposition')
    native_rules=importlib.import_module('flagprof.vendors.'+vendor+'.native_rules')
    ledger=enumerate_source(events,uid,dispositions.classifier(events,inventory,taxonomy))
    trace=adapt(events,calls,trace_uid=uid,kind=effective_kind,marker_prefix=prefix,inventory=inventory,taxonomy=taxonomy)
    unbound_calls = preserve_unbound_communication(trace, calls)
    if registry is None:
        registry = adapter.default_registry()
    rules=adapter.build_rules(trace)
    result=link(trace,rules,registry)
    output,devices,missing=[],[],{}
    validator=SchemaValidator()
    field_records = adapter.field_context(events)
    originals={str(e['_linker_loc']):o for e,o in zip(events,original)}
    for device in result['devices']:
        try:
            fields=adapter.device_fields(device,field_records)
        except UnrepresentableError as error:
            missing[identity(device)]=error.fields
            continue
        dev,stream=fields.pop('dev'),fields.pop('stream')
        ext=dict(device['ext'],**{'chrome.trace_uid':uid,'chrome.loc':device['loc'],
                                 'chrome.raw_record':originals[str(device['loc'])],'chrome.clock_valid':False})
        event=dict(t='kernel',name=device['name'],dev=dev,stream=stream,ktype=device['category'],
                   ts=None,dur=device['dur'],clk='unmapped:'+uid,link=device['link'],ext=ext)
        if device['category']=='memcpy' and {'kind','bytes'}<=fields.keys():
            event.update(t='memop',kind=fields['kind'],bytes=fields['bytes']);event.pop('name');event.pop('ktype')
        else:
            event.update({k:fields[k] for k in ('grid','block','regs','smem') if k in fields})
        validator.decode(dict(event,seq=0))
        output.append(event);devices.append(device)
    ledger,summary=reconcile(ledger,devices,missing,inventory)
    config=(scope_config or ScopeConfig()).resolved(native_rules.ANNOTATION_CATEGORIES)
    host,direct,native_calls,annotations,modules=native_rules.build_native_rules(trace,config)
    relations=relate(trace,[r for r in rules if r.side=='device'],host,direct,native_calls,annotations,modules,registry)
    counts=Counter(r['category'] for r in inventory)
    raw=dict(categories={c:counts[c] for c in ('compute','comm','memcpy','unknown')},records=inventory)
    source_ids = source_identity(identities, vendor)
    source_ids["parser"]["taxonomy"] = taxonomy.identity()
    source_ids["parser"]["closure"] = parser_identity(vendor,taxonomy.version)
    if hasattr(adapter,'rules_identity'):
        source_ids['parser']['rules']=adapter.rules_identity()
    assessment=assess(events,source_ids,ledger,taxonomy)
    for group in assessment['unknown_groups']:
        for ref in group['examples']:ref['trace_uid']=uid
    return PreparedImport(uid,digest,output,devices,relations,ledger,summary,raw,source_ids,
        coverage(output,raw),native_coverage(devices,relations),
        dict(kind=effective_kind,marker_prefix=prefix,calls=calls),native_rules.VERSION,asdict(config),
        sorted(unbound_calls,key=lambda r: (str(r['trace_uid']),int(r['loc']))),assessment)
