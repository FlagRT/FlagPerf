# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""FlagPerf inference v2: independent profiling infrastructure."""

__version__ = "0.3.0"


def attach(*args, **kwargs):
    from .runtime import attach as implementation
    return implementation(*args, **kwargs)


def current_attachment():
    from .runtime import current_attachment as implementation
    return implementation()


def declare_execution_mode(mode):
    """Record the workload's selected mode for an active final profiler session."""
    import sys
    from .vendors.final_session import FinalSession
    handle = current_attachment()
    session = getattr(handle, 'session', None)
    if isinstance(session, FinalSession):
        caller = sys._getframe(1)
        session.declare_execution_mode(mode, caller.f_code, caller.f_lineno)


def scope(name, **attrs):
    """An optional user interval; inert when no active user probe exists."""
    from contextlib import contextmanager
    @contextmanager
    def interval():
        handle = current_attachment()
        if handle is None or handle.probe_manager is None or handle.probe_manager.user is None or not handle.probe_manager.user.active():
            yield None
            return
        from .probes.base import clock
        parent = handle.context.current_scope()
        start = clock()
        with handle.context.scope(name) as token:
            try:
                yield token
            finally:
                event = dict(t='span', ph='x', cat='user', name=name, corr=token.corr,
                             ts=start, dur=clock()-start, attrs=attrs)
                if parent:
                    event['parent'] = parent.corr
                handle.writer.emit(event)
    return interval()
