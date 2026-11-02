# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Lifecycle of an observer over an independently owned FlagGems component."""
from .base import Probe


class RouteProbe(Probe):
    id='route'
    produces=('route',)
    requires=('dispatch','flaggems')

    def install(self):
        if self.installed:return
        if self.manager.dispatch is None:
            self.manager.unavailable(self.id,'dispatch_disabled');return
        if self.manager.component is None or self.manager.component.lib is None:
            self.manager.unavailable(self.id,'flaggems_off');return
        self.manager.component.observer=self.manager.route_observer
        self.installed=True
        self.manager.state['probes'][self.id]=dict(status='available',descriptor=self.descriptor())

    def uninstall(self):
        if self.manager.component:
            self.manager.component.observer=None
        super().uninstall()
