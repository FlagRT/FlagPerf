# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Per-call windows and disjoint summary aggregation."""
from contextlib import nullcontext
import json
import threading
from .base import Probe, clock


def devices(args, kwargs, output):
    import torch
    found = {}
    pending = [(str(i), v) for i, v in enumerate(args)]
    pending.extend(('kw:'+k, v) for k, v in sorted(kwargs.items()))
    pending.append(('out', output))
    while pending:
        path, item = pending.pop()
        if isinstance(item, (torch.Tensor, torch.device)):
            found[path] = str(item.device if isinstance(item, torch.Tensor) else item)
        elif isinstance(item, (tuple, list)):
            pending.extend((path+('.'+str(i) if path.startswith('out') else '['+str(i)+']'), v) for i, v in enumerate(item))
        elif isinstance(item, dict):
            pending.extend((path+'{'+str(k)+'}', v) for k, v in item.items())
    if not found:
        return None
    values = set(found.values())
    return next(iter(values)) if len(values) == 1 else dict(sorted(found.items()))


from flagprof.devices import accelerator



class DispatchProbe(Probe):
    id = 'dispatch'
    produces = ('sig', 'op', 'opsum')
    requires = ('torch',)

    def __init__(self, manager):
        super().__init__(manager)
        self.mode = None
        self.summary = {}
        self.lock = threading.RLock()
        self.window_start = clock()
        self.keys = set()
        self.calls = 0

    def install(self):
        if self.installed:
            return
        from torch.utils._python_dispatch import TorchDispatchMode
        owner = self
        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if not owner.active():
                    return func(*args, **(kwargs or {}))
                return owner.invoke(func, args, kwargs or {})
        self.mode = Mode()
        self.mode.__enter__()
        self.installed = True

    def invoke(self, func, args, kwargs):
        import torch
        from .argtree import describe_args, describe_result
        from ..store.sig import compute_sid
        state = self.context.probe_state()
        root = state['root']
        each = bool(root and root['each'])
        training = state.get('training')
        if training and self.config['dispatch'] == 'each-window':
            # Explicit training scopes select whole optimizer steps, including
            # backward and optimizer calls on the owning context. No inferred
            # thread propagation or change to an unmarked inference workload.
            index = training['step']
            each = self.config['each_start'] <= index < self.config['each_start'] + self.config['each_count']
        if not training and (self.manager.module is None or not self.manager.module.active()) and self.config['dispatch'] == 'each-window':
            each = self.calls < self.config['fallback_calls']
        self.calls += 1
        # Metadata failures downgrade this probe while the user's call still runs once.
        try:
            name = func._schema.name
            overload = func._schema.overload_name
            self.keys.add((name, overload))
            tree = describe_args(args, kwargs)
            sid = compute_sid(name, overload, 'in', tree)
            self.context.writer.register_signature(dict(name=name, overload=overload, role='in', tree=tree, sid=sid))
        except Exception as error:
            self.manager.failure(self.id, error)
            return func(*args, **kwargs)
        from ..components.flaggems import aten_key
        with self.context.probe_op(each,op_key=aten_key(name,overload)) as (fields, frame):
            start = clock()
            marker = (torch.profiler.record_function('flagprof::op#' + str(fields['corr']))
                      if each and not frame['nested'] and self.manager.official else nullcontext())
            output = None
            error = None
            try:
                with marker:
                    output = func(*args, **kwargs)
                return output
            except BaseException as exc:
                error = exc
                raise
            finally:
                dur = clock()-start
                try:
                    dev = devices(args, kwargs, output)
                    route = self.manager.route_decision(name, overload, frame, dev, error)
                    if each:
                        outtree = describe_result(output)
                        osid = compute_sid(name, overload, 'out', outtree)
                        self.context.writer.register_signature(dict(name=name, overload=overload, role='out', tree=outtree, sid=osid))
                        event = dict(t='op', sid=sid, osid=osid, ts=start, dur=dur, fw='torch', **fields)
                        if dev is not None:
                            event['dev'] = dev
                        if error is not None:
                            event['err'] = repr(error)
                        self.context.writer.emit(event)
                        if route:
                            self.context.writer.emit(dict(t='route', corr=fields['corr'], lib='flaggems', **route))
                    else:
                        self.aggregate(sid, dev, frame['nested'], route, start, dur)
                except Exception as exc:
                    self.manager.failure(self.id, exc)

    def aggregate(self, sid, dev, nested, route, start, dur):
        key = (sid, json.dumps(dev, sort_keys=True), nested, route['decision'] if route else None)
        with self.lock:
            row = self.summary.get(key)
            if row is None:
                row = dict(t='opsum', sid=sid, n=0, dur_sum=0, dur_min=dur, dur_max=dur)
                if dev is not None:
                    row['dev'] = dev
                if nested:
                    row['nested'] = True
                if route:
                    row['route_decision'] = route['decision']
                self.summary[key] = row
            row.update(n=row['n']+1, dur_sum=row['dur_sum']+dur,
                       dur_min=min(row['dur_min'], dur), dur_max=max(row['dur_max'], dur))
            if clock()-self.window_start >= self.config['summary_seconds']*1e9:
                self.flush()

    def flush(self):
        with self.lock:
            end = clock()
            for key in sorted(self.summary, key=repr):
                self.context.writer.emit(dict(self.summary[key], win=[self.window_start, end]))
            self.summary.clear()
            self.window_start = end

    def uninstall(self):
        if self.installed:
            from flagprof.tools.op_kinds import build
            self.manager.state['op_kinds'] = build(self.keys)
            self.manager.state['op_kinds_bytes'] = len(json.dumps(self.manager.state['op_kinds']).encode())
            self.flush()
            self.mode.__exit__(None, None, None)
        self.mode = None
        super().uninstall()
