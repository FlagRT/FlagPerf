# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Owned FlagGems registration, with an optional per-invocation observer."""
import functools
import hashlib
import json
from pathlib import Path


def aten_key(name, overload=''):
    name = name.removeprefix('aten::').removeprefix('aten.')
    return name + ('.' + overload if overload and overload != 'default' else '')


class FlagGems:
    def __init__(self, include, observer=None, on_error=None):
        self.include = list(include)
        self.observer = observer
        self.on_error=on_error
        self.observer_errors=[]
        self.lib = None
        self.table = {}
        self.identity = {}
        self.module = self.registrar = self.previous_registrar = None

    def notify(self, *args):
        if self.observer is None:
            return None
        try:
            return self.observer(*args)
        except Exception as error:
            self.observer_errors.append(repr(error))
            self.observer=None
            if self.on_error:
                try:self.on_error(error)
                except Exception as secondary:self.observer_errors.append(repr(secondary))
            return None

    def enable(self):
        if self.lib is not None:
            return self
        import torch
        import flag_gems
        from flag_gems.runtime.op_registrar import GeneralOpRegistrar
        owner = self
        self.module = flag_gems
        self.previous_registrar = flag_gems.current_work_registrar
        registered = {}
        class Registrar(GeneralOpRegistrar):
            def register_impl(self, key, fn, extra_dispatch_keys=()):
                if owner.observer is None:
                    wrapped = fn
                else:
                    @functools.wraps(fn)
                    def wrapped(*args, **kwargs):
                        token = owner.notify('enter', key, fn.__name__, None)
                        try:
                            return fn(*args, **kwargs)
                        except BaseException as error:
                            owner.notify('error', key, fn.__name__, token)
                            raise
                        finally:
                            owner.notify('exit', key, fn.__name__, token)
                super().register_impl(key, wrapped, extra_dispatch_keys)
                registered[key] = fn.__name__
        self.lib = torch.library.Library('aten', 'IMPL')
        try:
            flag_gems.only_enable(include=self.include, lib=self.lib, registrar=Registrar, record=False, once=True)
            registrar = flag_gems.current_work_registrar
            self.registrar = registrar
            unused = set(registrar.get_vendor_unused_op())
            for item in flag_gems._FULL_CONFIG:
                key, fn = item[:2]
                name, dot, overload = key.partition('.')
                canonical = aten_key(name, overload)
                self.table[canonical] = dict(name='aten::'+name, overload=overload if dot else '',
                    registered=key in registered, impl=registered.get(key),
                    policy_excluded=fn.__name__ not in self.include,
                    device_unsupported=fn.__name__ in unused or key in unused or not registrar._config_enabled(item))
            if not registered or not set(self.include) <= set(registered.values()):
                raise RuntimeError('requested FlagGems functions did not all register')
            root = Path(flag_gems.__file__).parent
            source = hashlib.sha256()
            for path in sorted(root.rglob('*.py')):
                source.update(str(path.relative_to(root)).encode())
                source.update(hashlib.sha256(path.read_bytes()).digest())
            data = json.dumps(self.table, sort_keys=True, separators=(',', ':')).encode()
            self.identity = dict(source_sha256=source.hexdigest(), include=self.include,
                unused=sorted(unused), route_table_sha256=hashlib.sha256(data).hexdigest(), route_table_bytes=len(data))
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.lib is not None:
            self.lib._destroy()
            self.lib = None
        if self.module is not None and self.module.current_work_registrar is self.registrar:
            self.module.current_work_registrar = self.previous_registrar
