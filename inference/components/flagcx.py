"""FlagCX import and native ownership are bounded to a worker process."""
import os


class FlagCXAdapter:
    def __init__(self):
        self.owner = None

    def source_identity(self):
        import importlib.util
        from pathlib import Path
        from runtime.common import digest, file_hash
        spec = importlib.util.find_spec('flagcx')
        if not spec or not spec.submodule_search_locations:
            raise ValueError('selected FlagCX component is not installed')
        root = Path(next(iter(spec.submodule_search_locations)))
        return digest({p.relative_to(root).as_posix():file_hash(p) for p in sorted(root.rglob('*'))
                       if p.is_file() and p.suffix in ('.py', '.so')})

    def activate(self):
        previous = os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')
        import flagcx
        if previous is not None:
            os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = previous

    def retain_owner(self, dist):
        # The locked build borrows the runtime stream; final-reference release calls
        # streamDestroy on that borrowed stream. Ownership therefore ends at exit.
        self.owner = dist.distributed_c10d._get_default_group()
        return self.owner
