#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Extract a vendor Triton compiler from a user-selected local Docker image."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

# Locate installed packages inside the source image; never assume a Python prefix.
PACKAGE_QUERY = r'''
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
spec = importlib.util.find_spec('triton')
if spec is None or spec.origin is None:
    raise SystemExit('Source image must contain the vendor triton-ascend compiler')
module = Path(spec.origin).resolve()
try:
    tree = md.distribution('flagtree')
except md.PackageNotFoundError:
    tree = None
if tree is not None and Path(tree.locate_file('triton/__init__.py')).resolve() == module:
    raise SystemExit('Source image exposes FlagTree; select an image with the native vendor compiler')
parent = module.parent.parent
packages = [module.parent.name]
versions = {}
for name in ('triton', 'triton-ascend'):
    distribution = md.distribution(name)
    metadata = next((p for p in distribution.files or ()
                     if p.name == 'METADATA' and p.parent.name.endswith('.dist-info')), None)
    if metadata is None:
        raise SystemExit('Missing wheel metadata for ' + name)
    folder = Path(distribution.locate_file(metadata)).resolve().parent
    if folder.parent != parent:
        raise SystemExit('Compiler and metadata must share one site-packages directory')
    packages.append(folder.name)
    versions[name] = distribution.version
print(json.dumps({'directory': str(parent), 'packages': packages, 'versions': versions}))
'''


def prepare(image, output):
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError('Destination already exists; choose a new --output to preserve the sealed bundle')
    info = json.loads(subprocess.check_output(
        ['docker', 'image', 'inspect', image], text=True))[0]
    source_id = info['Id']
    container = ['docker', 'run', '--rm', '--read-only', '--network', 'none',
                 '-e', 'TORCH_DEVICE_BACKEND_AUTOLOAD=0', '-e', 'PYTHONDONTWRITEBYTECODE=1']
    layout = json.loads(subprocess.check_output(
        [*container, '--entrypoint', 'python3', source_id, '-c', PACKAGE_QUERY], text=True))
    packages = layout['packages']
    if (not isinstance(packages, list) or not packages
            or any(not isinstance(p, str) or Path(p).name != p or p in ('.', '..') for p in packages)):
        raise ValueError('Source image returned an invalid compiler package layout')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.compiler-', dir=output.parent) as temp:
        folder = Path(temp) / 'bundle'
        folder.mkdir()
        process = subprocess.Popen(
            [*container, '--entrypoint', 'tar', source_id, '-C', layout['directory'], '-cf', '-', *packages],
            stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
                for member in archive:
                    target = (folder / member.name).resolve()
                    if (folder not in target.parents or member.issym() or member.islnk()
                            or not (member.isfile() or member.isdir())):
                        raise ValueError('Unsafe compiler archive entry: ' + member.name)
                    archive.extract(member, folder)
            if process.wait(timeout=30):
                raise RuntimeError('Compiler extraction failed')
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
        for path in folder.rglob('__pycache__'):
            shutil.rmtree(path)
        hashes = {}
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                        digest.update(block)
                hashes[str(path.relative_to(folder))] = digest.hexdigest()
        manifest = {'schema_version': 1, 'source_image': source_id, 'packages': packages,
                    'versions': layout['versions'], 'files': hashes}
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        folder.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='Local image containing native triton-ascend, not FlagTree')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'runtime_assets/ascend-triton')
    args = parser.parse_args()
    try:
        print(prepare(args.image, args.output))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f'Compiler preparation failed: {error}\n')


if __name__ == '__main__':
    main()
