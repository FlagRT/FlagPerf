# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Training adapters over the existing attachment and evidence Store.

Importing this module does not import torch. Observation never calls backward,
changes model mode, or selects a numerical policy on the application's behalf.
"""
from contextlib import contextmanager
import sys


def attachment():
    """Do not initialize the Linux collector for an unobserved application."""
    runtime = sys.modules.get('flagprof.runtime')
    return runtime.current_attachment() if runtime is not None else None


@contextmanager
def phase(name, *, step, microstep=None, **attrs):
    """Optional training interval; host duration, not device completion time."""
    if name not in ('step', 'zero_grad', 'forward', 'backward', 'optimizer'):
        raise ValueError('unknown training phase: '+name)
    if type(step) is not int or step < 0:
        raise ValueError('step must be a nonnegative integer')
    if microstep is not None and (type(microstep) is not int or microstep < 0):
        raise ValueError('microstep must be a nonnegative integer')
    from flagprof import scope
    handle = attachment()
    if handle is None:
        yield
        return
    import torch
    outcome = dict(status='running')
    fields = dict(attrs, training_schema='training/1', phase=name, step=step, outcome=outcome)
    if microstep is not None:
        fields['microstep'] = microstep
    state = handle.context.probe_state()
    previous = state.get('training')
    state['training'] = fields
    try:
        with scope('training.'+name, **fields):
            with torch.profiler.record_function('flagprof/training/'+name+'/'+str(step)):
                try:
                    yield
                except BaseException as error:
                    outcome.update(status='failed', error=type(error).__name__)
                    raise
                else:
                    outcome['status'] = 'complete'
    finally:
        if previous is None:
            state.pop('training', None)
        else:
            state['training'] = previous


def step(index, **attrs):
    """Use around an application's unchanged training iteration."""
    return phase('step', step=index, **attrs)


def register_model(model):
    handle = attachment()
    if handle is not None:
        handle.register_model(model)
    return model
