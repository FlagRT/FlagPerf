# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Structural JIT observations; no latency-based cache classification."""
from .base import Probe
import importlib.metadata


class CompileProbe(Probe):
    id = 'compile'
    produces = ('compile',)
    requires = ('triton',)

    def install(self):
        if self.installed:
            return
        import triton
        from triton import knobs
        if not hasattr(knobs, 'compilation') or not hasattr(knobs.compilation, 'listener'):
            self.manager.unavailable(self.id, 'cache_state_indeterminate')
            return
        self.knobs = knobs
        self.previous = knobs.compilation.listener
        try:
            self.version = importlib.metadata.version('flagtree')
            self.backend = 'flagtree'
        except importlib.metadata.PackageNotFoundError:
            self.version, self.backend = triton.__version__, 'triton'
        def listener(**values):
            previous_result = self.previous(**values) if self.previous else None
            self.safe(self.compiled, values)
            return previous_result
        self.listener = listener
        knobs.compilation.listener = listener
        self.installed = True
        self.manager.state['probes'][self.id] = dict(status='available', descriptor=self.descriptor(),
            backend=self.backend, ver=self.version, triton_version=triton.__version__,
            scope='compiler_invocations; persistent cache hit/miss; in-memory-only reuse has no observation',
            not_observable=['compile_failure','cache_invalid','in_memory_reuse'],
            source='triton.knobs.compilation.listener.cache_hit')

    def compiled(self, values):
        if type(values.get('cache_hit')) is not bool:
            raise ValueError('cache_state_indeterminate: listener lacks structural cache_hit')
        metadata, times = values['metadata'], values['times']
        key = metadata.get('hash')
        if key is None:
            key = values['src'].hash()
        self.observe(key=key, cache='hit' if values['cache_hit'] else 'miss',
            dur=max(0, int(times.total))*1000, backend=self.backend, ver=self.version,
            kname=metadata.get('name'), stage_dur=dict(ir_initialization=times.ir_initialization*1000,
                store_results=times.store_results*1000, **{name:value*1000 for name,value in times.lowering_stages}))

    def uninstall(self):
        if self.installed:
            if self.knobs.compilation.listener is self.listener:
                self.knobs.compilation.listener = self.previous
            else:
                self.manager.failure(self.id, RuntimeError('listener ownership changed; later listener preserved'))
        super().uninstall()

    def observe(self, *, key, cache, dur, backend, ver, kname=None, err=None, stage_dur=None):
        if cache not in ('hit', 'miss', 'invalid'):
            raise ValueError('structural cache observation required')
        event = dict(t='compile', corr=self.context.current_corr(), backend=backend,
                     key=str(key), cache=cache, dur=dur, ver=ver)
        scope = self.context.current_scope()
        if scope:
            event['ext'] = dict(scope=scope.corr)
        if kname:
            event['kname'] = kname
        if err:
            event['err'] = err
        if stage_dur is not None:
            event['stage_dur'] = stage_dur
        return self.context.writer.emit(event)
