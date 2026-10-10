"""Untimed observations, with optional component hooks loaded on demand."""
from runtime.common import write_json
from runtime.adapters import required_components
from vendors.stack import selection, communication_backend


class StackAudit:
    def __init__(self, cfg, root, compiler=None):
        self.cfg, self.root, self.checked = cfg, root, compiler
        self.compiler_audit = None
        self.collectives, self.restore = [], []

    def __enter__(self):
        import torch.distributed as dist
        try:
            if 'compiler' in required_components(self.cfg) or (self.checked and self.checked.get('provider') != 'not_collected'):
                from components.audit import CompilerAudit
                self.compiler_audit = CompilerAudit(self.cfg, self.root, self.checked)
                self.compiler_audit.__enter__()
            if self.cfg['runtime'].get('parallelism') == 'tp':
                for name in ('all_reduce', 'all_gather', 'all_gather_into_tensor', 'reduce_scatter_tensor'):
                    native = getattr(dist, name)
                    def observed(*args, _name=name, _native=native, **kwargs):
                        position = 3 if _name == 'reduce_scatter_tensor' else 2
                        group = kwargs.get('group', args[position] if len(args) > position else None)
                        tensor = args[0] if args else kwargs.get('tensor')
                        row = {'collective': _name, 'backend': dist.get_backend(group)}
                        if hasattr(tensor, 'numel'):
                            row.update(shape=list(tensor.shape), dtype=str(tensor.dtype),
                                       logical_bytes=tensor.numel()*tensor.element_size())
                        result = _native(*args, **kwargs)
                        self.collectives.append(row)
                        return result
                    setattr(dist, name, observed)
                    self.restore.append(lambda name=name, native=native: setattr(dist, name, native))
            return self
        except BaseException:
            for restore in reversed(self.restore): restore()
            if self.compiler_audit:
                for restore in reversed(self.compiler_audit.restore): restore()
            raise

    def __exit__(self, *exc):
        for restore in reversed(self.restore): restore()
        compiler = {'provider': 'not_collected', 'observation': 'not_collected',
                    'jit_run_calls': None, 'jit_warmup_calls': None, 'launch_hook_calls': None,
                    'reason': 'compiler audit not selected; this is not evidence of no compiler participation'}
        if self.compiler_audit:
            self.compiler_audit.__exit__(*exc)
            compiler = self.compiler_audit.result
        tp = self.cfg['runtime'].get('parallelism') == 'tp'
        expected = communication_backend(self.cfg) if tp else None
        mismatch = any(c['backend'] != expected for c in self.collectives)
        self.result = {'requested': selection(self.cfg), 'compiler': compiler,
            'communication': {'expected_backend': expected,
                'observed_backends': sorted({c['backend'] for c in self.collectives}),
                'status': 'not_applicable' if not tp else ('mismatch' if mismatch else 'observed' if self.collectives else 'not_observed'),
                'model_collective_calls': len(self.collectives), 'collectives': self.collectives,
                'scope': 'model forward only; control-group synchronization excluded'}}
        if self.compiler_audit and self.cfg.get('_preview_trial'):
            self.result['cache_groups'] = dict(self.compiler_audit.cache_groups,
                scope='Python file cache group reads/writes; not hardware kernel coverage')
        write_json(self.root/'components.json', self.result)
        if mismatch and exc[0] is None:
            raise ValueError('model collective used an unexpected backend')
