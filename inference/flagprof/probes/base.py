# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Attachment-owned probe lifecycle and safe metadata callbacks."""
import os
import time


def clock():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


class Probe:
    id = 'probe'
    produces = ()
    requires = ()

    def __init__(self, manager):
        self.manager = manager
        self.context = manager.context
        self.config = manager.config
        self.installed = False
        self.failed = False
        self.pid = os.getpid()

    def descriptor(self):
        return dict(id=self.id, kind='probe', version='1', module=type(self).__module__,
                    factory=type(self).__name__, requires=list(self.requires),
                    produces=list(self.produces), overhead='medium', sources=['python structural hooks'])

    def active(self):
        return self.installed and not self.failed and self.pid == os.getpid() and not self.context.closed

    def safe(self, callback, *args, **kwargs):
        if self.active():
            try:
                return callback(*args, **kwargs)
            except Exception as error:
                self.manager.failure(self.id, error)

    def uninstall(self):
        self.installed = False
