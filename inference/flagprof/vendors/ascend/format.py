"""Versioned CANN record taxonomy and host-format hooks."""
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence
from flagprof.vendors.fingerprint import format_identity
from flagprof.vendors.contracts import load_contract, closure, source_times_present, bind_roles, track_role, device_scope, execution_tracks

VERSION = '4'
LEGACY_COMPUTE = ('AI_CORE', 'AI_VECTOR_CORE', 'MIX_AIC', 'MIX_AIV', 'AI_CPU')
CURRENT_COMPUTE = ('KERNEL_AICORE', 'KERNEL_AIVEC', 'KERNEL_MIX_AIV', 'KERNEL_AICPU')
COPY = ('SDMA_SQE', 'MEMCPY', 'MEMCPY_ASYNC', 'MEMSET')
PID_OFFSET = 10
CANN_HOST_BITS = (7 << 5) | 31


def queue_key_valid(value: Any) -> bool:
    return type(value) is int and value >= 0


class HostContext(dict):
    """Raw process names plus independent evidence of real host processes."""
    def __init__(self, events: Sequence[Mapping[str, Any]]) -> None:
        super().__init__((e.get('pid'), e.get('args', {}).get('name'))
                         for e in events if e.get('name') == 'process_name')
        self.names: dict[Any, set] = {}
        self.host_records = set()
        for event in events:
            pid = event.get('pid')
            if event.get('name') == 'process_name':
                self.names.setdefault(pid, set()).add(event.get('args', {}).get('name'))
            if event.get('cat') in NATIVE_CALL_CATEGORIES or event.get('name', '').startswith(('Enqueue', 'Dequeue')):
                self.host_records.add(pid)


def decoded_host_pid(event: Mapping[str, Any], context: HostContext) -> Any:
    """Interpret only the sourced CANN host namespace, never raw PID equality.

Official CANN 9.0 TraceViewManager: (pid << 10) | (layer << 5) | device.
CANN host layer=7, device=31. torch_npu Level0 can remove its process_name.
See docs/validation/t8.2/host-identity-provenance.json for exact source hashes.
"""
    pid = event.get('pid')
    names = context.names.get(pid, set())
    encoded_record = 'CANN' in names or (event.get('ph') == 's' and event.get('cat') == 'HostToDevice')
    if not encoded_record:
        return pid
    if (type(pid) is int and pid >= 0 and pid & ((1 << PID_OFFSET) - 1) == CANN_HOST_BITS
            and (not names or names == {'CANN'}) and pid not in context.host_records
            and context.names.get(pid >> PID_OFFSET) == {'Python'}):
        return pid >> PID_OFFSET
    # This internal key cannot equal a raw JSON process ID. Keeping raw pid as
    # the join key would allow a complete competing host chain to steal a flow.
    # The original pid remains untouched in chrome.adapt and in source evidence.
    return ('unresolved-host', pid, event.get('_linker_loc'))

# T9.0 predicate inventory: native call candidates differ from host-track categories.
NATIVE_CALL_CATEGORIES = ('cpu_op',)
# Native annotation selection differs from device-track exemptions.
ANNOTATION_CATEGORIES = ('user_annotation', 'gpu_user_annotation')

@dataclass(frozen=True)
class Taxonomy:
    version: str = VERSION

    def __post_init__(self) -> None:
        if self.version not in ('1', '2', '3', '4'):
            raise ValueError('unsupported ascend taxonomy version: ' + str(self.version))
        object.__setattr__(self,'contract',load_contract(__file__,self.version))

    @property
    def compute(self) -> tuple[str, ...]:
        return LEGACY_COMPUTE + (CURRENT_COMPUTE if self.version != '1' else ())

    def identity(self) -> dict[str, Any]:
        # All hooks, constants and helpers live in this source module.
        return dict(format_identity(self.version, closure(__file__,self.version)), revision=self.contract['revision'])

    def context(self, events: Sequence[Mapping[str, Any]]) -> dict:
        return bind_roles(events, self, HostContext(events))

    def is_device(self, event: Mapping[str, Any], context: Any) -> bool:
        return event.get('ph') == 'X' and context.get(event.get('pid')) in self.contract['roles']['device_process_names']

    def task_type(self, event: Mapping[str, Any]) -> Any:
        return event.get('args', {}).get('Task Type')

    def category(self, event: Mapping[str, Any], context: Any) -> str:
        name = event.get('name', '')
        if context.get(event.get('pid')) in self.contract['roles']['communication_process_names']:
            return 'comm' if name.lower().startswith('hcom_') else 'unknown'
        task = self.task_type(event)
        if task in self.compute:
            return 'comm' if name == 'allreduceAicpuKernel' else 'compute'
        return 'memcpy' if task in COPY else 'unknown'

    def normalizable(self, event: Mapping[str, Any], context: Any) -> bool:
        execution = set(self.contract['roles']['unknown_execution_process_names'])
        if (not self.is_device(event,context) or track_role(event,context)!='device'
                or (context.get(event.get('pid')) in execution and event.get('name') in self.contract['excluded_duplicate_names'])):
            return False
        return self.category(event,context)!='unknown' or (self.contract['open_unknown']
            and bool(context.names.get(event.get('pid'))) and context.names[event.get('pid')] <= execution
            and self.task_type(event) is not None and source_times_present(event))

    def host_pid(self, event: Mapping[str, Any], context: Any) -> Any:
        if self.version in ('3','4'):
            return decoded_host_pid(event, context)
        pid = event.get('pid')
        return pid >> 10 if context.get(pid) == 'CANN' and isinstance(pid, int) else pid

    def is_launch(self, event: Mapping[str, Any]) -> bool:
        return event.get('name') == 'Node@launch' or (event.get('ph') == 's' and event.get('cat') == 'HostToDevice')

    def audit_dependencies(self, events: Sequence[Mapping[str,Any]], context: Any) -> list[dict]:
        issues={}
        for event in events:
            if self.version in ('3','4') and event.get('ph')=='s' and event.get('cat')=='HostToDevice':
                key=decoded_host_pid(event,context)
                if isinstance(key,tuple):
                    issues.setdefault('encoded_host_target',dict(id='encoded_host_target',reason='HostToDevice encoded pid lacks a unique valid Python target',count=0))['count']+=1
            if event.get('name','').startswith(('Enqueue','Dequeue')):
                args=event.get('args',{});value=args.get('correlation_id')
                valid=queue_key_valid(value)
                unexpected=any('correlation' in k.lower() and k!='correlation_id' for k in args)
                if not valid or unexpected:
                    issues.setdefault('queue_key',dict(id='queue_key',reason='Queue keys require correlation_id in the documented queue integer domain; unrelated fields are not aliases',count=0))['count']+=1
        return [issues[k] for k in sorted(issues)]

    def scope_facts(self, events: Sequence[Mapping[str,Any]], context: Any) -> dict:
        labels={e.get('pid'):e.get('args',{}).get('labels','') for e in events if e.get('name')=='process_labels'}
        devices=[e for e in events if self.is_device(e,context)]
        values=[]
        for event in devices:
            matched=re.fullmatch(r'NPU (\d+)',labels.get(event.get('pid'),''))
            values.append(int(matched.group(1)) if matched else None)
        hosts={e.get('pid') for e in events if e.get('ph')=='X' and e.get('cat') in NATIVE_CALL_CATEGORIES}
        facts = device_scope(values,{e.get('pid') for e in devices},hosts,
            dict(kind='raw_trace',fields=['process_labels.args.labels','pid','cat'],examples=[e.get('_linker_loc') for e in devices[:3]]))
        facts.update(execution_tracks(devices, values, context))
        return facts
