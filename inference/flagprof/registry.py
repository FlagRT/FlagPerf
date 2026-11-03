# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Lazy component descriptors, separate from project binding registries."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import re

BUILTINS = (
    dict(id='thead', kind='vendor', version='1', module='flagprof.vendors.thead.session',
         factory='TheadProfiler', requires=['torch'], format_vendor='thead'),
    dict(id='ascend', kind='vendor', version='1', module='flagprof.vendors.ascend.session',
         factory='AscendProfiler', requires=['torch_npu'], format_vendor='ascend'),
    dict(id='nvidia', kind='vendor', version='1', module='flagprof.vendors.nvidia.session',
         factory='NvidiaProfiler', requires=['torch'], format_vendor='nvidia'),
    *(dict(id=name,kind='probe',version='1',module='flagprof.probes.'+module,
           factory=factory,requires=requires) for name,module,factory,requires in (
        ('module','modules','ModuleProbe',['torch']),('dispatch','dispatch','DispatchProbe',['torch']),
        ('route','route','RouteProbe',['dispatch','flaggems']),('compile','compile','CompileProbe',['triton']),
        ('user','user','UserProbe',[]),('tensor','tensor','TensorProbe',['torch']))),
    dict(id='flaggems',kind='component',version='1',module='flagprof.components.flaggems',factory='FlagGems',requires=['flag_gems']),
    *(dict(id=name,kind='analyzer',version='1',module='flagprof.analyzers.core',factory=factory,requires=[])
      for name,factory in (('perf.op','make_perf_op'),('perf.layer','make_perf_layer'),
                          ('route.coverage','make_route_coverage'),('compile.health','make_compile_health'))),
    dict(id='perf.gap',kind='analyzer',version='1',module='flagprof.analyzers.gap',factory='make_perf_gap',requires=[]),
    dict(id='perf.device',kind='analyzer',version='1',module='flagprof.analyzers.device',factory='make_perf_device',requires=[]),
    dict(id='memory',kind='analyzer',version='1',module='flagprof.analyzers.memory',factory='make_memory',requires=[]),
    *(dict(id='oracle.'+name,kind='analyzer',version='1',module='flagprof.analyzers.oracle',factory='make_'+name,requires=[])
      for name in ('identity_diff','layer','first_divergence','op_sensitivity')),
)


class Registry:
    def __init__(self, descriptors=None):
        self._items = {}
        for item in (BUILTINS if descriptors is None else descriptors):
            d = deepcopy(item)
            for key in ('id', 'kind', 'version', 'module', 'factory'):
                if not isinstance(d.get(key), str) or not d[key]:
                    raise ValueError('descriptor requires ' + key)
            if d['kind'] not in ('vendor', 'probe', 'analyzer', 'component', 'responder'):
                raise ValueError('unknown descriptor kind')
            if not re.fullmatch(r'[A-Za-z_]\w*(\.[A-Za-z_]\w*)*', d['module']) or not d['factory'].isidentifier():
                raise ValueError('invalid factory location')
            if d['id'] in self._items:
                raise ValueError('duplicate descriptor id: ' + d['id'])
            if not isinstance(d.get('requires'), list) or not all(isinstance(x,str) for x in d['requires']):
                raise ValueError('requires must be a list of names')
            if d['kind'] == 'vendor' and d.get('format_vendor') not in ('ascend', 'nvidia', 'thead'):
                raise ValueError('unsupported format family')
            self._items[d['id']] = d

    def descriptors(self):
        return deepcopy(list(self._items.values()))

    def describe(self, ident):
        if ident not in self._items:
            raise ValueError('unknown component: ' + ident)
        return deepcopy(self._items[ident])

    def create(self, ident, **kwargs):
        d = self.describe(ident)
        module = importlib.import_module(d['module'])
        factory = getattr(module, d['factory'])
        if not callable(factory):
            raise ValueError('factory is not callable')
        path = Path(module.__file__)
        identity = dict(d, implementation=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if d['kind']=='analyzer':
            root=Path(__file__).resolve().parent.parent
            paths=sorted(p for p in (root/'flagprof').rglob('*') if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts)
            modules=[p.relative_to(root).as_posix() for p in paths]
            closure=dict(algorithm='analyzer-static-closure-v1',sources=[(name,hashlib.sha256(p.read_bytes()).hexdigest()) for name,p in zip(modules,paths)])
            identity['shipped_source_closure']=dict(modules=modules,sha256=hashlib.sha256(json.dumps(closure,sort_keys=True,separators=(',',':')).encode()).hexdigest())
        else:
            from .vendors.fingerprint import parser_identity
            identity['shipped_source_closure'] = parser_identity(d.get('format_vendor',d['kind']), d['version'])
        identity['external_dependency_boundary'] = 'external factory module hashed; transitive external dependencies are not certified'
        return factory(**kwargs), identity
