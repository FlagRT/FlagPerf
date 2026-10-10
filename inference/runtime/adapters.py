"""Explicit, repository-local adapter discovery; descriptors never import devices."""
from copy import deepcopy
import importlib
import json
from pathlib import Path, PurePosixPath
import re
import ast

from runtime.common import ROOT

KINDS = {'models': 'benchmarks', 'vendors': 'vendors', 'components': 'components', 'engines': 'inference_engine'}
SCHEME = 'scoped-adapters-v1'


def local_file(name, root=None):
    root = Path(root or ROOT).resolve()
    if not isinstance(name, str):
        raise ValueError('unsafe adapter source: ' + str(name))
    path = PurePosixPath(name)
    if (not isinstance(name, str) or not name or '\\' in name or path.is_absolute()
            or any(p in ('', '.', '..') for p in name.split('/'))):
        raise ValueError('unsafe adapter source: ' + str(name))
    target = root / name
    if (not target.is_file() or not target.resolve().is_relative_to(root)
            or any(p.is_symlink() for p in [target, *target.parents] if p != root.parent)):
        raise ValueError('missing or unsafe adapter source: ' + name)
    return target


def descriptor(kind, name, root=None):
    if (kind not in KINDS or not isinstance(name, str) or '..' in name
            or not re.fullmatch(r'[a-z][a-z0-9_.]*', name)):
        raise ValueError('unknown adapter: ' + str(kind) + '/' + str(name))
    filename = f'registry/{kind}/{name}.json'
    try:
        value = json.loads(local_file(filename, root).read_text())
    except (FileNotFoundError, ValueError) as error:
        raise ValueError('unknown adapter: ' + kind + '/' + name) from error
    allowed = {'schema_version', 'id', 'factory', 'capabilities', 'sources', 'analysis_sources',
               'packages', 'config_defaults', 'path_fields', 'legacy_contract', 'component_sources', 'decoder', 'capture_patterns'}
    if (not isinstance(value, dict) or set(value) - allowed or value.get('schema_version') != 1
            or value.get('id') != name or not isinstance(value.get('capabilities'), dict)):
        raise ValueError('invalid adapter descriptor: ' + filename)
    factory = value.get('factory', '')
    if (not isinstance(factory, str) or not re.fullmatch(r'[a-zA-Z_][\w.]*:[a-zA-Z_]\w*', factory)
            or factory.split('.')[0] != KINDS[kind]):
        raise ValueError('unsafe adapter factory: ' + str(factory))
    module = factory.split(':')[0].replace('.', '/') + '.py'
    if module not in value.get('sources', []):
        raise ValueError('adapter factory is outside declared source closure: ' + filename)
    for key in ('sources', 'analysis_sources'):
        names = value.get(key, [])
        if not isinstance(names, list) or len(set(names)) != len(names):
            raise ValueError('invalid adapter sources: ' + filename)
        for source in names:
            local_file(source, root)
    for names in value.get('component_sources', {}).values():
        for source in names:
            local_file(source, root)
    if set(value.get('sources',[])) & set(value.get('analysis_sources',[])):
        raise ValueError('adapter execution and analysis sources overlap')
    for cap, enabled in value['capabilities'].items():
        if not isinstance(enabled,(bool,list,dict)):
            raise ValueError('invalid adapter capability: '+cap)
        if isinstance(enabled,list) and len({str(v) for v in enabled}) != len(enabled):
            raise ValueError('invalid adapter capability values: '+cap)
    validate_source_closure(value,root)
    decoder = value.get('decoder')
    if decoder is not None and (not isinstance(decoder, str) or
            not re.fullmatch(r'vendors\.[a-zA-Z_]\w*(\.[a-zA-Z_]\w*)*', decoder) or
            decoder.replace('.', '/') + '.py' not in value.get('analysis_sources', [])):
        raise ValueError('decoder is outside declared analysis sources')
    value = deepcopy(value)
    value['_descriptor'] = filename
    return value


def create(kind, name):
    entry = descriptor(kind, name)
    module, factory = entry['factory'].split(':')
    imported = importlib.import_module(module)
    if Path(imported.__file__).resolve() != local_file(module.replace('.', '/') + '.py').resolve():
        raise ValueError('adapter import is outside its declared repository source')
    adapter = getattr(imported, factory)()
    if kind == 'models':
        from runtime.workload import ModelAdapter
        if not isinstance(adapter,ModelAdapter): raise ValueError('model factory must implement ModelAdapter')
        for field in ('semantics','timing','parallel_strategy','throughput_unit'):
            if not isinstance(getattr(adapter,field,None),str) or not getattr(adapter,field):
                raise ValueError('model adapter lacks semantic metadata: '+field)
        for cap, method in [('oracle','reference_contract'),('performance-baseline','reference_contract'),('export','export_spec')]:
            if entry['capabilities'].get(cap) and getattr(type(adapter),method) is getattr(ModelAdapter,method):
                raise ValueError('advertised model capability lacks implementation: '+cap)
        if (entry['capabilities'].get('oracle') or entry['capabilities'].get('performance-baseline')) and not getattr(adapter,'reference_dtype',None):
            raise ValueError('reference capability requires an explicit reference dtype')
    return adapter


def validate_source_closure(entry,root=None):
    """Validate same-package imports and shared benchmark imports, including relative imports."""
    root = Path(root or ROOT)
    declared = set(entry.get('sources',[])) | set(entry.get('analysis_sources',[]))
    for sources in entry.get('component_sources',{}).values(): declared.update(sources)
    package = entry['factory'].split(':')[0].rsplit('.',1)[0].replace('.','/')
    for source in sorted(declared):
        if not source.endswith('.py'): continue
        tree = ast.parse(local_file(source,root).read_text(),filename=source)
        for node in ast.walk(tree):
            modules = [a.name for a in node.names] if isinstance(node,ast.Import) else []
            if isinstance(node,ast.ImportFrom):
                if node.level:
                    parent = source.split('/')[:-1]
                    parent = parent[:len(parent)-node.level+1]
                    modules = ['.'.join(parent+([node.module] if node.module else []))]
                    if node.module is None: modules = [m+'.'+a.name for m in modules for a in node.names]
                else: modules = [node.module or '']
                modules += [m+'.'+a.name for m in list(modules) for a in node.names if a.name != '*']
            for module in modules:
                path = module.replace('.','/')+'.py'
                init = module.replace('.','/')+'/__init__.py'
                target = path if (root/path).is_file() else init if (root/init).is_file() else None
                owned = target and (target.startswith(package+'/') or
                                    (package.startswith('benchmarks/') and target.startswith('benchmarks/')))
                if owned and target not in declared:
                    raise ValueError('undeclared transitive adapter source: '+target)


def model_adapter(cfg):
    return create('models', cfg['model']['name'])


def vendor_adapter(cfg):
    return create('vendors', cfg['runtime']['vendor'])


def required_components(cfg, command=None):
    """Union of both comparison sides, fixed before any worker is launched."""
    if '_required_components' in cfg:
        return list(cfg['_required_components'])
    command = command or cfg.get('_command', 'preview' if cfg.get('_preview_trial') else 'accuracy')
    section = cfg.get('_stack', cfg.get(command, cfg.get('performance' if command == 'performance-baseline' else 'accuracy', {})))
    required = []
    if command == 'preview' or section.get('flaggems', 'off') != 'off':
        required += ['flaggems', 'compiler']
    if section.get('flagtree', 'off') != 'off' and 'compiler' not in required:
        required.append('compiler')
    if cfg['runtime'].get('parallelism') == 'tp' and section.get('flagcx', 'off') != 'off':
        required.append('flagcx')
    return sorted(required)


def source_scope(cfg, root=None):
    """Shared execution plus selected adapters; analysis and unused code stay auditable."""
    root = Path(root or ROOT)
    selected = set()
    for folder in ('runtime',):
        selected.update(p.relative_to(root).as_posix() for p in (root / folder).rglob('*.py'))
    selected.update(p.relative_to(root).as_posix() for p in root.glob('*.py'))
    # Compatibility facades are part of the public execution surface.
    selected.update(p.relative_to(root).as_posix() for p in (root / 'vendors').glob('*.py'))
    entries = [('engines', 'pytorch'), ('models', cfg['model']['name']), ('vendors', cfg['runtime']['vendor'])]
    entries += [('components', name) for name in required_components(cfg)]
    for kind, name in entries:
        entry = descriptor(kind, name, root)
        selected.add(entry['_descriptor'])
        selected.update(entry['sources'])
        for component in required_components(cfg):
            selected.update(entry.get('component_sources', {}).get(component, []))
    return sorted(selected)


def bind_scope(cfg, command):
    cfg.pop('_required_components', None)
    cfg.pop('_source_scope', None)
    cfg['_command'] = command
    cfg['_required_components'] = required_components(cfg, command)
    cfg['_source_scope'] = source_scope(cfg)
    return cfg


def require_capability(entry, capability, value=None):
    supported = entry['capabilities'].get(capability, False)
    if not supported or (value is not None and isinstance(supported, list) and value not in supported):
        raise ValueError(f"{entry['id']} adapter does not support {capability}" +
                         (f'={value}' if value is not None else ''))
