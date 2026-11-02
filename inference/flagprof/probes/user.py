"""Attachment-owned enablement for the public optional scope API."""
from .base import Probe


class UserProbe(Probe):
    id='user'
    produces=('span',)
    requires=()

    def install(self):
        self.installed=True
        self.manager.state['probes'][self.id]=dict(status='available',descriptor=self.descriptor())
