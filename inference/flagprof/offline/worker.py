"""Run disk-backed operations with a process-private Python and SQLite temp directory."""
from dataclasses import asdict, is_dataclass
import errno
import importlib
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys

from .workspace import attempt

_active = None
_lease = None


def lease_fds():
    return () if _lease is None else (_lease,)


def active_directory():
    return _active


def _encode(value):
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError('unsupported operation argument: ' + type(value).__name__)


def on_disk(volume, function, *args, **kwargs):
    # Inner report panels and readers can share the parent operation. An import
    # needs its own staging area and key even when another worker requested it.
    if _active is not None and function.__name__ != '_import_bundle':
        return function(*args, **kwargs)
    with attempt(volume) as (directory, lease):
        request = dict(module=function.__module__, function=function.__name__, args=args, kwargs=kwargs)
        (directory / 'request.json').write_text(json.dumps(request, default=_encode))
        env = dict(os.environ, TMPDIR=str(directory), SQLITE_TMPDIR=str(directory))
        # Also support the source-tree entry point from an arbitrary cwd.
        root = str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH'] = os.pathsep.join(filter(None, (root, env.get('PYTHONPATH'))))
        with subprocess.Popen([sys.executable, '-B', '-m', 'flagprof.offline.worker', str(directory), str(lease)],
                              env=env, pass_fds=(lease,)) as child:
            try:
                code = child.wait()
            except BaseException:
                child.terminate()
                child.wait()
                raise
        response = directory / 'response.json'
        if not response.is_file():
            raise RuntimeError(f'{function.__name__} worker exited {code} in {directory}')
        value = json.loads(response.read_text())
        if code:
            errors = {t.__name__: t for t in (ValueError, TypeError, KeyError, OSError, FileExistsError,
                                               PermissionError, FileNotFoundError, RuntimeError)}
            cls = errors.get(value['type'], RuntimeError)
            if issubclass(cls, OSError) and value.get('errno') is not None:
                raise cls(value['errno'], value['message'])
            raise cls(value['message'])
        return value


def output_operation(out, inputs, function, *args, **kwargs):
    out = Path(out).resolve()
    for source in inputs:
        if source is None:
            continue
        path = Path(source).resolve()
        if path.name == 'group.json':
            path = path.parent
        if out.is_relative_to(path) or path.is_relative_to(out):
            raise ValueError('output must be separate from the input: ' + str(path))
    # A new child directory inherits its parent's mount. For an existing
    # mountpoint use the mount itself, even when the parent is read-only.
    volume = out if out.is_dir() else out.parent
    try:
        return on_disk(volume, function, *args, out=str(out), **kwargs)
    except OSError as error:
        if isinstance(error, PermissionError) or error.errno == errno.EROFS:
            raise PermissionError(f'cannot write output {out}; select a writable --out: {error}') from error
        raise


def new_output(path):
    from .workspace import private_name
    path.mkdir(parents=True, exist_ok=True)
    if any(entry.name != private_name() for entry in path.iterdir()):
        raise FileExistsError('output directory contains files: ' + str(path))


def main(directory, lease):
    global _active, _lease
    _active, _lease = Path(directory), lease
    request = json.loads((_active / 'request.json').read_text())
    module = request['module']
    operations = {
        'flagprof.cli': ('_inspect_store',), 'flagprof._import_worker': ('_import',),
        'flagprof.offline.service': ('_inspect_run', '_import_bundle'),
        'flagprof.offline.capture': ('_analyse_locked',),
        'flagprof.report': ('_generate', '_generate_oracle'),
        'flagprof.report.group': ('_generate_group',), 'flagprof.report.product': ('_redraw',),
        'flagprof.report.performance': ('_generate',), 'flagprof.report.comparison': ('_generate',),
        'flagprof.monitor.commands': ('_release',), 'flagprof.monitor.report': ('_generate',),
        'flagprof.replay': ('_prepare',),
    }
    try:
        if request['function'] not in operations.get(module, ()):
            raise ValueError('unknown FlagPerf operation')
        operation = getattr(importlib.import_module(module), request['function'])
        result = operation(*request['args'], **request['kwargs'])
        code = 0
    except (sqlite3.OperationalError, OSError) as error:
        result = dict(type=type(error).__name__, errno=getattr(error, 'errno', None),
                      message=f"{error} (operation directory {directory})")
        code = 1
    except Exception as error:
        result = dict(type=type(error).__name__, message=str(error))
        code = 1
    (_active / 'response.json').write_text(json.dumps(result, default=_encode))
    return code


if __name__ == '__main__':
    # Import the canonical module so nested operations see the same context.
    from flagprof.offline.worker import main
    sys.exit(main(sys.argv[1], int(sys.argv[2])))
