# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Attachment-owned enablement for the public optional scope API."""
from .base import Probe


class UserProbe(Probe):
    id='user'
    produces=('span',)
    requires=()

    def install(self):
        self.installed=True
        self.manager.state['probes'][self.id]=dict(status='available',descriptor=self.descriptor())
