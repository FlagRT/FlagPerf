"""Compiler identity and optional dispatch observation live with the component."""
from pathlib import Path
from runtime.common import file_hash, digest


def imported_compiler_identity(provider):
    import triton
    root = Path(triton.__file__).resolve().parent
    files = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob('*'))
             if p.is_file() and p.suffix in ('.py', '.so')}
    return {'provider': provider, 'import_version': triton.__version__, 'module_path': str(root / '__init__.py'),
            'source_sha256': digest(files), 'bundle_manifest_sha256': None}


class CompilerAdapter:
    def identity(self, cfg):
        from runtime.adapters import vendor_adapter
        return vendor_adapter(cfg).compiler_identity(cfg)
