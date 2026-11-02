"""Versioned Kineto device taxonomy used by the CUDA rule family."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from flagprof.vendors.fingerprint import format_identity
from flagprof.vendors.contracts import load_contract, closure, source_times_present, bind_roles, track_role, device_scope, execution_tracks

VERSION = '2'
CATEGORIES = {'kernel': 'compute', 'gpu_memcpy': 'memcpy', 'gpu_memset': 'memcpy'}

class DeviceContext(set):
    """Raw device process hints plus a per-context role index."""

# T9.0 predicate inventory: native call candidates differ from host-track categories.
NATIVE_CALL_CATEGORIES = ('cpu_op',)
# Native annotation selection differs from device-track exemptions.
ANNOTATION_CATEGORIES = ('user_annotation', 'gpu_user_annotation')

@dataclass(frozen=True)
class Taxonomy:
    version: str = VERSION

    def __post_init__(self) -> None:
        if self.version not in ('1','2'):
            raise ValueError('unsupported nvidia taxonomy version: ' + str(self.version))
        object.__setattr__(self,'contract',load_contract(__file__,self.version))

    def identity(self) -> dict[str, Any]:
        return dict(format_identity(self.version, closure(__file__,self.version)), revision=self.contract['revision'])

    def context(self, events: Sequence[Mapping[str, Any]]) -> set:
        context = DeviceContext(e.get('pid') for e in events if e.get('cat') in self.contract['roles']['device_categories'])
        return bind_roles(events, self, context)

    def is_device(self, event: Mapping[str, Any], context: Any) -> bool:
        roles = self.contract['roles']
        if event.get('ph') != 'X' or event.get('cat') in roles['host_categories'] + roles['allowed_non_device_categories']:
            return False
        args = event.get('args', {})
        return event.get('cat') in roles['device_categories'] or event.get('pid') in context or ('device' in args and 'stream' in args)

    def category(self, event: Mapping[str, Any], context: Any) -> str:
        return CATEGORIES.get(event.get('cat'), 'unknown')

    def normalizable(self, event: Mapping[str, Any], context: Any) -> bool:
        return self.is_device(event, context) and track_role(event, context) == 'device' and (self.category(event, context) != 'unknown'
            or (self.contract['open_unknown'] and source_times_present(event)))

    def task_type(self, event: Mapping[str, Any]) -> Any:
        return event.get('args', {}).get('Task Type')

    def host_pid(self, event: Mapping[str, Any], context: Any) -> Any:
        return event.get('pid')

    def is_launch(self, event: Mapping[str, Any]) -> bool:
        return (event.get('ph') == 's' and event.get('cat') == 'ac2g') or (
            event.get('cat') in ('cuda_runtime', 'cuda_driver') and event.get('args', {}).get('correlation') is not None)

    def audit_dependencies(self, events: Sequence[Mapping[str,Any]], context: Any) -> list[dict]:
        names={e.get('pid'):e.get('args',{}).get('name') for e in events if e.get('name')=='process_name'}
        allowed=set(self.contract['roles']['expected_process_names'])
        unexpected=sorted(str(pid) for pid,name in names.items() if name not in allowed)
        return [dict(id='device_process_label',reason='Device process label was not present in the verified export structure',pids=unexpected)] if unexpected else []

    def scope_facts(self, events: Sequence[Mapping[str,Any]], context: Any) -> dict:
        devices=[e for e in events if self.is_device(e,context)]
        hosts={e.get('pid') for e in events if e.get('ph')=='X' and e.get('cat') in self.contract['roles']['host_categories']}
        values=[e.get('args',{}).get('device') for e in devices]
        facts = device_scope(values,
            {e.get('pid') for e in devices},hosts,
            dict(kind='raw_trace',fields=['args.device','pid','cat'],examples=[e.get('_linker_loc') for e in devices[:3]]))
        facts.update(execution_tracks(devices, values, context))
        selected=[e for e in devices if track_role(e,context)=='device']
        mismatch=any('device' in e.get('args',{}) and (type(e.get('pid')) is not int or type(e['args']['device']) is not int or e.get('pid') != e['args']['device']) for e in selected)
        missing=not selected or any('device' not in e.get('args',{}) for e in selected)
        facts['device_pid_matches_device_arg']=False if mismatch else None if missing else True
        return facts
