"""Ascend-specific requirements are local and versioned by a stack profile."""
import os
from pathlib import Path
import subprocess

from runtime.common import read_json
from vendors.base import VendorAdapter


def profile(cfg=None):
    name = (cfg or {}).get('vendors', {}).get('ascend', {}).get('profile', 'cann9_torch210')
    if name != 'cann9_torch210':
        raise ValueError('unknown Ascend stack profile: ' + str(name))
    return read_json(Path(__file__).parent / 'profiles' / (name + '.json'))


class AscendAdapter(VendorAdapter):
    visible_devices = 'ASCEND_RT_VISIBLE_DEVICES'
    device_type = 'npu'

    def needs_bundle(self, cfg):
        from runtime.adapters import required_components
        selected = cfg.get('_stack', cfg.get(cfg.get('_command', 'accuracy'), {}))
        return 'compiler' in required_components(cfg) and selected.get('flagtree', 'off') != 'on'

    def validate_launch_assets(self, cfg):
        if self.needs_bundle(cfg):
            from vendors.ascend.compiler import validate_bundle
            validate_bundle(cfg['vendors']['ascend']['vendor_compiler'])

    def validate_config(self, cfg, command):
        selected = cfg['vendors']['ascend']
        if set(selected) - {'env', 'vendor_compiler', 'profile'}:
            raise ValueError('unknown Ascend vendor configuration')
        if not isinstance(selected['env'], dict):
            raise ValueError('vendors.ascend.env must be a mapping')
        for key, value in profile(cfg)['environment_constraints'].items():
            if selected['env'].get(key) != value:
                raise ValueError('validated Ascend stack requires ' + key + '=' + value)

    def initialize(self, cfg):
        import torch_npu
        import torch
        backend = torch.npu
        if not backend.is_available():
            raise RuntimeError('selected device backend is unavailable; no CPU fallback')
        rank = int(os.environ.get('LOCAL_RANK', '0')) if cfg['runtime'].get('parallelism') == 'tp' else 0
        backend.set_device(rank)
        torch.manual_seed(cfg['runtime']['seed'])
        backend.manual_seed_all(cfg['runtime']['seed'])
        return f'npu:{rank}', backend

    def communication_backend(self, cfg, selected):
        return 'flagcx' if selected.get('flagcx') == 'on' else 'hccl'

    def container_arguments(self, cfg, root):
        tp = cfg['runtime'].get('parallelism') == 'tp'
        devices = cfg['runtime']['devices'] if tp else [cfg['runtime']['device']]
        argv = []
        for dev in [*[f'davinci{d}' for d in sorted(devices)], 'davinci_manager', 'devmm_svm', 'hisi_hdc']:
            if not Path('/dev', dev).exists():
                raise ValueError(f'missing Ascend device /dev/{dev}')
            argv += ['--device', f'/dev/{dev}:/dev/{dev}:rwm']
        for path in ['/usr/local/Ascend/driver', '/usr/local/Ascend/firmware', '/usr/local/dcmi', '/etc/ascend_install.info', '/usr/local/bin/npu-smi']:
            if Path(path).exists():
                argv += ['-v', f'{path}:{path}:ro']
        visible = ','.join(str(sorted(devices).index(d)) for d in devices)
        argv += ['-e', f'ASCEND_RT_VISIBLE_DEVICES={visible}']
        from runtime.adapters import required_components
        if self.needs_bundle(cfg):
            folder = cfg['vendors']['ascend']['vendor_compiler']
            argv += ['-v', f'{folder}:{folder}:ro']
        return argv

    def validate_image(self, cfg, image_id):
        if (not isinstance(image_id, str) or not image_id.startswith('sha256:')
                or len(image_id) != 71
                or any(c not in '0123456789abcdef' for c in image_id[7:])):
            raise ValueError('Docker must resolve the selected image to a complete SHA256 ID')

    def before_after_snapshot(self):
        state = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True)
        return 'npu', state.stdout + state.stderr

    def telemetry_provider(self):
        from vendors.ascend.telemetry import AutoTelemetry
        return AutoTelemetry()

    def worker_environment(self, cfg, root, compiler_required):
        env = self.environment(cfg)
        if compiler_required:
            from vendors.ascend.compiler import worker_environment
            selected = worker_environment(cfg, root)
            env.update({k: v for k, v in selected.items() if k in ('PYTHONPATH', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'PYTHONDONTWRITEBYTECODE')})
        return env

    def compiler_identity(self, cfg):
        from vendors.ascend.compiler import compiler_identity
        return compiler_identity(cfg)

    def profiler(self, cfg, root, warmup, active):
        import torch_npu
        from torch_npu.version import git_version
        from runtime.common import write_json
        write_json(Path(root) / 'trace-capture.json', {'schema_version': 1, 'vendor': 'ascend',
            'format': profile(cfg)['trace_format'], 'version': profile(cfg)['trace_version'],
            'collective_runtime':{'torch_npu_version':torch_npu.__version__,
                                  'torch_npu_git_version':git_version,
                                  'device_name':torch_npu.npu.get_device_name()}})
        p = torch_npu.profiler
        return p.profile(activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU],
            schedule=p.schedule(wait=0, warmup=warmup, active=active, repeat=1),
            on_trace_ready=p.tensorboard_trace_handler(str(Path(root) / 'profiler')), record_shapes=True,
            experimental_config=p._ExperimentalConfig(profiler_level=p.ProfilerLevel.Level1))

    def stream_id(self, backend):
        return int(backend.current_stream().npu_stream)
