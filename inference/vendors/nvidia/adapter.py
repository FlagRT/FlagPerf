"""CUDA requirements belong to the reference vendor, never an implicit fallback."""
import os
from pathlib import Path
from vendors.base import VendorAdapter


class NvidiaAdapter(VendorAdapter):
    visible_devices = 'CUDA_VISIBLE_DEVICES'
    device_type = 'cuda'

    def validate_config(self, cfg, command):
        if set(cfg['vendors']['nvidia']) - {'env'}:
            raise ValueError('unknown NVIDIA vendor configuration')
        if command not in ('oracle', 'performance-baseline', 'list') and not cfg.get('_performance_capture'):
            raise ValueError('NVIDIA/A100 supports reference capture only: use run_inference.py oracle '
                             'or run_inference.py performance-baseline; run candidate tests on the target vendor')

    def environment(self, cfg):
        env = super().environment(cfg)
        if cfg.get('_oracle_capture'):
            env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        elif cfg.get('_performance_native'):
            env.pop('CUBLAS_WORKSPACE_CONFIG', None)
        return env

    def initialize(self, cfg):
        import torch
        backend = torch.cuda
        if not backend.is_available():
            raise RuntimeError('selected device backend is unavailable; no CPU fallback')
        rank = int(os.environ.get('LOCAL_RANK', '0')) if cfg['runtime'].get('parallelism') == 'tp' else 0
        backend.set_device(rank)
        torch.manual_seed(cfg['runtime']['seed'])
        backend.manual_seed_all(cfg['runtime']['seed'])
        return f'cuda:{rank}', backend

    def configure_numerics(self, cfg, backend):
        from vendors.nvidia.numerics import configure_oracle, configure_performance
        if cfg.get('_oracle_capture'):
            configure_oracle(backend, cfg['runtime']['seed'])
        elif cfg.get('_performance_native'):
            configure_performance(backend)

    def numerical_settings(self):
        from vendors.nvidia.numerics import settings
        return settings()

    def environment_snapshot(self):
        from vendors.nvidia.numerics import environment_snapshot
        return environment_snapshot()

    def communication_backend(self, cfg, selected):
        return 'nccl'

    def container_arguments(self, cfg, root):
        raise ValueError('A100 capture uses run_inference.py in the prepared CUDA environment; '
                         'run.py launches Ascend containers only')

    def compiler_identity(self, cfg):
        from components.compiler import imported_compiler_identity
        return imported_compiler_identity('vendor')

    def profiler(self, cfg, root, warmup, active):
        import torch
        from runtime.common import write_json
        write_json(Path(root) / 'trace-capture.json', {'schema_version': 1, 'vendor': 'nvidia',
            'format': 'kineto', 'version': '1'})
        p = torch.profiler
        return p.profile(activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.CUDA],
            schedule=p.schedule(wait=0, warmup=warmup, active=active, repeat=1),
            on_trace_ready=p.tensorboard_trace_handler(str(Path(root) / 'profiler')), record_shapes=True)

    def stream_id(self, backend):
        return int(backend.current_stream().cuda_stream)
