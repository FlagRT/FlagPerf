"""Native ownership over source identities, independent of project corr.

Vendor adapters supply already prepared host/direct rules and node classes.
Only proven same-source-thread containment can remove an ancestor. Device
timestamps are never consulted when selecting an owner.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

from .linker import NormalizedTrace, Record, Rule, identity, ref, ref_sort, unique_refs, registry_match


def thread(node: Mapping[str, Any]) -> tuple[Any, ...]:
    return node['trace_uid'], node.get('host_pid'), node.get('tid')


def contains(outer: Mapping[str, Any], inner: Mapping[str, Any], *, strict: bool = False) -> bool:
    if thread(outer) != thread(inner) or None in (outer.get('ts'), inner.get('ts'), outer.get('dur'), inner.get('dur')):
        return False
    a, c = outer['ts'], inner['ts']
    b, d = a + outer['dur'], c + inner['dur']
    return a <= c < b and d <= b and outer['dur'] > 0 and (not strict or (a, b) != (c, d))


def innermost(candidates: Sequence[Record]) -> list[Record]:
    nodes = sorted({identity(n): n for n in candidates}.values(), key=ref_sort)
    if len(nodes) < 2:
        return nodes
    if not all(contains(a, b, strict=True) or contains(b, a, strict=True) for a, b in combinations(nodes, 2)):
        return nodes
    return [n for n in nodes if not any(contains(n, other, strict=True) for other in nodes if identity(n) != identity(other))]


class ScopeIndex:
    """Source-thread interval lookup; no SDK or vendor-format knowledge."""
    def __init__(self, nodes: Sequence[Record]) -> None:
        from bisect import bisect_right
        self.bisect = bisect_right
        groups: dict[tuple, list] = {}
        for n in nodes:
            if n.get('ts') is not None and n.get('dur') is not None:
                groups.setdefault(thread(n), []).append(n)
        self.groups = {}
        for key, ns in groups.items():
            ns.sort(key=lambda n: (n['ts'], n['dur'], ref_sort(n)))
            size=1
            while size<len(ns):size*=2
            maxima=[float('-inf')]*(size*2)
            for i,n in enumerate(ns):maxima[size+i]=n['ts']+n['dur']
            for i in range(size-1,0,-1):maxima[i]=max(maxima[i*2],maxima[i*2+1])
            self.groups[key] = ([n['ts'] for n in ns], ns, maxima, size)

    def enclosing(self, node: Record, *, strict: bool = False) -> list[Record]:
        if node.get('ts') is None:
            return []
        starts, ns, maxima, size = self.groups.get(thread(node), ([], [], [], 0))
        limit = self.bisect(starts, node['ts'])
        result = []
        stack=[(1,0,size)] if size else []
        while stack:
            index,lo,hi=stack.pop()
            if lo>=limit or maxima[index]<node['ts']:continue
            if hi-lo==1:
                candidate=ns[lo]
                if identity(candidate)!=identity(node) and contains(candidate,node,strict=strict):result.append(candidate)
            else:
                middle=(lo+hi)//2
                stack.extend(((index*2,lo,middle),(index*2+1,middle,hi)))
        return sorted(result, key=lambda n: (n['ts'], -n['dur'], ref_sort(n)))


def relate(trace: NormalizedTrace, device_rules: Sequence[Rule], host_rules: Sequence[Rule],
           direct_rules: Sequence[Rule], calls: Sequence[Record], annotations: Sequence[Record],
           module_ids: set[tuple[str, str]], registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    call_ids = {identity(n) for n in calls}
    scope_index = ScopeIndex([*calls, *annotations])
    host = {}
    for launch in trace.launches:
        candidates, evidence, via = [], [launch], set()
        for rule in host_rules:
            hits = rule.matches(launch)
            if hits:
                via.add(rule.via)
            for hit in hits:
                if identity(hit.target) not in call_ids:
                    raise ValueError('native host rule returned a non-call owner')
                candidates.append(hit.target)
                evidence.extend((hit.target, *hit.evidence))
        host[identity(launch)] = (innermost(candidates), evidence, via)
    rows = []
    for event in sorted(trace.devices, key=ref_sort):
        sources, candidates, evidence, via = {}, {}, [event], set()
        direct = {}
        for rule in device_rules:
            if rule.side != 'device':
                continue
            hits = rule.matches(event)
            if hits:
                via.add(rule.via)
            for hit in hits:
                sources[identity(hit.target)] = hit.target
                evidence.extend((hit.target, *hit.evidence))
        unknown = False
        for key, launch in sources.items():
            owners, proof, rules = host[key]
            unknown |= not owners
            candidates.update((identity(o), o) for o in owners)
            evidence.extend(proof)
            via.update(rules)
        for rule in direct_rules:
            hits = rule.matches(event)
            if hits:
                via.add(rule.via)
            for hit in hits:
                if identity(hit.target) not in call_ids:
                    raise ValueError('native direct rule returned a non-call owner')
                direct[identity(hit.target)] = hit.target
                evidence.extend((hit.target, *hit.evidence))
        # Direct evidence is retained separately. A missing required launch
        # never becomes a fabricated complete two-hop chain.
        excluded = registry_match(event, registry)
        count = len(candidates)
        state = 'exempt' if excluded else 'conflicted' if count > 1 or (count == 1 and unknown) else 'owned' if count else 'unowned'
        owner = next(iter(candidates.values())) if state == 'owned' else None
        ancestors = scope_index.enclosing(owner, strict=True) if owner else []
        module_candidates = [n for n in ancestors if identity(n) in module_ids]
        modules = innermost(module_candidates)
        # Also expose annotations around launches, including inner route
        # scopes that are not ancestors of the owner itself.
        contexts = unique_refs([n for launch in sources.values() for n in scope_index.enclosing(launch)
                                if identity(n) not in call_ids])
        evidence.extend(ancestors)
        refs = unique_refs(list(candidates.values()))
        rows.append(dict(trace_uid=event['trace_uid'], loc=event['loc'], category=event['category'],
            state=state, owner=ref(owner) if owner else None, candidates=refs + ([None] if unknown and refs else []),
            launches=unique_refs(list(sources.values())), direct_owners=unique_refs(list(direct.values())),
            unknown_launch_owner=unknown, scope_chain=[ref(n) for n in ancestors],
            scope_chain_ordered=all(contains(a,b,strict=True) for a,b in zip(ancestors,ancestors[1:])),
            module_scope=ref(modules[0]) if len(modules)==1 else None,
            module_candidates=unique_refs(modules), launch_scopes=contexts,
            via=sorted(via), evidence=unique_refs(evidence + contexts),
            reason='excluded' if excluded else 'conflict' if state=='conflicted' else None if owner else 'no_bridge' if sources else 'no_device_key'))
    return rows


def coverage(events: Sequence[Mapping[str, Any]], relations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    byref = {(r['trace_uid'], str(r['loc'])): r for r in relations}
    result = {}
    for category in ('compute','comm','memcpy','unknown'):
        selected = [e for e in events if e['category']==category and e['link']['status']!='not_applicable']
        owned = [e for e in selected if byref[identity(e)]['state']=='owned']
        scoped = [e for e in selected if byref[identity(e)]['module_scope'] is not None]
        total = sum(e.get('dur') or 0 for e in selected)
        result[category] = dict(native_owned_count_ratio=len(owned)/len(selected) if selected else None,
            native_owned_time_ratio=sum(e.get('dur') or 0 for e in owned)/total if total else None,
            scope_count_ratio=len(scoped)/len(selected) if selected else None,
            scope_time_ratio=sum(e.get('dur') or 0 for e in scoped)/total if total else None,
            eligible_count=len(selected), dur_missing=sum(e.get('dur') is None for e in selected))
    return result
