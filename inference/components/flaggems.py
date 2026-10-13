"""FlagGems registration and private discovery hooks, isolated from the engine."""
from collections import defaultdict
import functools
import json
from pathlib import Path
import torch
from runtime.tensor_signature import signature

class GemsRoute:
    """Wrap actual registered functions, not the surrounding ATen call count."""
    def __init__(self, include, directory):
        self.include, self.directory = include, Path(directory)
        self.calls = defaultdict(int)
        self.signatures = {}
        self.registered = []
        self.last_error = None
        self.lib = None
        self.log = None
        self.depth = 0
        self.scopes = None

    def enable(self):
        import flag_gems
        from flag_gems.runtime.op_registrar import GeneralOpRegistrar
        owner = self
        class EvidenceRegistrar(GeneralOpRegistrar):
            def register_impl(self, key, fn, extra_dispatch_keys=()):
                @functools.wraps(fn)
                def observed(*args, **kwargs):
                    owner.calls[fn.__name__] += 1
                    if owner.scopes: owner.scopes.observe(fn.__name__, 'function')
                    if fn.__name__ not in owner.signatures:
                        sig = signature({'args':args,'kwargs':kwargs})
                        owner.signatures[fn.__name__] = sig
                        owner.log.write(json.dumps({'function':fn.__name__,'aten_key':key,'signature':sig})+'\n')
                        owner.log.flush()
                    try:
                        owner.depth += 1
                        return fn(*args, **kwargs)
                    except Exception as error:
                        owner.last_error = {'function':fn.__name__, 'aten_key':key,
                                            'error':str(error), 'signature':signature({'args':args,'kwargs':kwargs})}
                        raise
                    finally:
                        owner.depth -= 1
                super().register_impl(key, observed, extra_dispatch_keys)
                owner.registered.append({'aten_key':key,'function':fn.__name__})
        self.log = (self.directory / 'gems-calls.jsonl').open('w')
        if not self.include:
            return
        self.lib = torch.library.Library('aten','IMPL')
        flag_gems.only_enable(include=self.include, lib=self.lib, registrar=EvidenceRegistrar,
                              record=True, once=True, path=str(self.directory/'gems.log'))
        if not self.registered:
            raise RuntimeError('requested FlagGems functions did not register')

    def evidence(self):
        return {'requested':True, 'allowed':self.include,'registered':self.registered,
                'actual_function_calls':dict(self.calls), 'signatures':self.signatures,
                'observed':bool(self.calls), 'last_error':self.last_error,
                'evidence_level':'Python FlagGems function invocation, not hardware kernel trace'}

    def close(self):
        if self.lib is not None: self.lib._destroy()
        if self.log is not None: self.log.close()


def candidates(inventory):
    import flag_gems
    by_function = {}
    for item in flag_gems._FULL_CONFIG:
        key, fn = item[:2]
        if key in inventory:
            row = by_function.setdefault(fn.__name__, {'function':fn.__name__, 'aten_keys':[], 'signatures':[]})
            row['aten_keys'].append(key)
            row['signatures'].extend(inventory[key]['signatures'])
    return [by_function[k] for k in sorted(by_function)]



class FlagGemsAdapter:
    def source_identity(self):
        import importlib.util
        from runtime.common import digest, file_hash
        spec = importlib.util.find_spec('flag_gems')
        if not spec or not spec.submodule_search_locations:
            raise ValueError('selected FlagGems component is not installed')
        root = Path(next(iter(spec.submodule_search_locations)))
        return digest({str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*.py'))})

    def candidates(self, inventory):
        return candidates(inventory)

    def route(self, include, directory):
        return GemsRoute(include, directory)

    def enable_timing(self, include):
        import flag_gems
        lib = torch.library.Library("aten", "IMPL")
        try:
            flag_gems.only_enable(include=include, lib=lib, record=False)
            registrar = flag_gems.current_work_registrar
            registered = list(zip(registrar.all_keys, registrar.all_ops))
            if not set(include) <= {name for _, name in registered}:
                raise RuntimeError("FlagGems registration is incomplete; see selected policy")
            return lib, registered
        except BaseException:
            lib._destroy()
            raise
