# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Attachment owner; optional torch imports happen only during installation."""
import os
from .base import clock


class ProbeManager:
    def __init__(self, context, config, state, *, official=False, state_callback=None):
        self.context, self.config, self.state = context, config, state
        self.official = official
        self.state_callback=state_callback
        self.module = self.dispatch = self.compile = self.component = self.tensor = None
        self.route=self.user=None
        self.communication=None
        self.errors = []
        self.pid = os.getpid()
        self.state.update(probes={}, effective_config=config, route_unobserved=0,
                          component=dict(flaggems=config['flaggems'], owned=True))

    def unavailable(self, name, reason):
        self.state['probes'][name] = dict(status='unavailable', reason=reason)

    def failure(self, name, error):
        probe=getattr(self,name,None)
        if probe is not None and getattr(probe,'failed',False):
            return
        if probe is not None:
            probe.failed=True
        self.errors.append(name + ': ' + repr(error))
        self.unavailable(name, type(error).__name__ + ': ' + str(error))
        if self.state_callback:
            try:self.state_callback()
            except Exception as secondary:self.errors.append('probe state save: '+repr(secondary))

    def install(self):
        enabled = self.config['probes'] != 'none'
        if self.config['flaggems'] == 'on':
            from ..components.flaggems import FlagGems
            self.component = FlagGems(self.config['flaggems_include'],
                self.route_observer if enabled and self.config['dispatch'] != 'off' else None,
                on_error=lambda error:self.failure('route',error))
            self.state['component']['observer_errors']=self.component.observer_errors
            component_started=clock()
            try:
                self.component.enable()
                self.state.update(route_table=self.component.table, flaggems_identity=self.component.identity)
                self.context.writer.emit(dict(t='meta', kind='env', data=dict(flaggems=self.component.identity)))
            except Exception as error:
                self.failure('route', error)
                self.state['component']['error'] = repr(error)
            finally:
                self.state['component']['install_ns']=clock()-component_started
        if not enabled:
            for name in ('module', 'dispatch', 'route', 'compile', 'user'):
                self.unavailable(name, 'probes_disabled')
            return
        from .modules import ModuleProbe
        from .dispatch import DispatchProbe
        for attr, factory, wanted in [('module', ModuleProbe, True),
                                      ('dispatch', DispatchProbe, self.config['dispatch'] != 'off')]:
            if not wanted:
                self.unavailable(attr, 'disabled')
                continue
            probe = factory(self)
            try:
                probe.install()
                setattr(self, attr, probe)
                self.state['probes'][attr] = dict(status='available', descriptor=probe.descriptor())
            except Exception as error:
                probe.uninstall()
                self.failure(attr, error)
        import sys
        torch=sys.modules.get('torch')
        if torch is not None:self.state['torch_version']=torch.__version__
        if 'tensor_blob' in self.config:
            from .tensor import TensorProbe
            self.tensor=TensorProbe(self)
            self.tensor.install()
            self.state['probes']['tensor']=dict(status='available',descriptor=self.tensor.descriptor())
        from .user import UserProbe
        self.user=UserProbe(self);self.user.install()
        if self.config['flaggems'] == 'off':
            self.unavailable('route', 'flaggems_off')
        elif self.dispatch is None:
            self.unavailable('route', 'dispatch_disabled')
        elif 'error' not in self.state['component']:
            from .route import RouteProbe
            self.route=RouteProbe(self);self.route.install()
        # Hook semantics are admitted only after the fixed-image structural test.
        try:
            from .compile import CompileProbe
            self.compile = CompileProbe(self)
            self.compile.install()
        except Exception as error:
            self.failure('compile', error)

    def route_observer(self, action, key, impl, token):
        if self.pid != os.getpid() or self.context.closed:
            return None
        frames = self.context.probe_state()['frames']
        if not frames:
            return None
        if action == 'enter':
            frame = frames[-1]
            from ..components.flaggems import aten_key
            owner=frame['route_depth']==0 and aten_key(key)==frame['op_key']
            frame['route_depth']+=1
            if owner:
                frame['route'] = dict(decision='hit', impl=impl, depth=len(frames)-1)
            return dict(frame=frame,owner=owner,depth=len(frames)-1)
        if token and action=='error' and token['owner']:
            token['frame']['route'] = dict(decision='error', reason='runtime_error', depth=token['depth'])
        if token and action=='exit':
            token['frame']['route_depth']-=1

    def route_decision(self, name, overload, frame, dev, error):
        from .dispatch import accelerator
        from ..components.flaggems import aten_key
        if self.state['probes'].get('route', {}).get('status') != 'available' or frame['nested'] or not accelerator(dev):
            return None
        if self.route is not None and not self.route.active():
            return None
        if frame['route']:
            return frame['route']
        row = self.state.get('route_table', {}).get(aten_key(name, overload))
        if row is None:
            return dict(decision='fallback', reason='not_registered')
        if row['device_unsupported']:
            return dict(decision='excluded', reason='device_unsupported')
        if row['policy_excluded']:
            return dict(decision='excluded', reason='policy_excluded')
        self.state['route_unobserved'] += 1
        return None

    def register_model(self, model):
        if self.module:
            try:
                self.module.register_model(model)
                if self.tensor:self.tensor.register_model(model)
            except Exception as error:
                self.failure('module',error)
                self.module.uninstall()
                self.module=None

    def close(self):
        if self.pid != os.getpid():
            return
        for probe in (self.communication,self.tensor,self.dispatch, self.compile, self.module,self.route,self.user):
            if probe:
                try:
                    probe.uninstall()
                except Exception as error:
                    self.failure(probe.id, error)
        if self.component:
            started=clock()
            self.component.close()
            self.state['component']['close_ns']=clock()-started
