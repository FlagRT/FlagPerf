# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""CANN 9 / Torch-FL integration. All Ascend host and runtime policy lives here."""
import importlib
import json
import os
from pathlib import Path
import re
import sys

BASE = Path(__file__).resolve().parents[3] / 'base'


def base_common():
    sys.path.insert(0, str(BASE))
    return importlib.import_module('executors.common')


class Adapter:
    name = 'ascend'

    def default_image(self):
        return base_common().runtime_lock_record()['image_manifest']['image']

    def runtime_identity(self, image, info):
        return base_common().validate_runtime_identity({'image': image}, info)

    def bootstrap(self, device_id):
        import torch_fl
        import torch
        torch.flagos.set_device(device_id)
        self.device = torch.device(f'flagos:{device_id}')
        return self.device

    def identity(self):
        import importlib.util
        import hashlib
        import torch_fl
        root = Path(torch_fl.__file__).parent
        config = root / 'configs/backends_ascend.conf'
        extension = Path(importlib.util.find_spec('torch_fl._C').origin)
        return {'backend_config': config.read_text(),
                'backend_config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
                'extension_sha256': hashlib.sha256(extension.read_bytes()).hexdigest(),
                'fp32_math_mode': 'runtime default; strict IEEE mode not asserted by harness',
                'visibility': os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}

    def synchronize(self):
        import torch
        torch.flagos.synchronize(self.device)

    def kernel_time(self, fn, cfg):
        # CUDA Triton do_bench is not a valid device timer for this locked stack.
        return None

    def environment(self, device, mode="probe", local=False):
        env = {'ASCEND_RT_VISIBLE_DEVICES': str(device), 'GEMS_VENDOR': 'ascend',
                'TRITON_ENABLE_TASKQUEUE': 'false', 'DO_NOT_TRACK': '1',
                'FLAGOS_LOG_FALLBACK': '1', 'FLAGOS_LOG_DISPATCH': '0' if mode == 'measure' else '1',
                'TORCH_DEVICE_BACKEND_AUTOLOAD': '0'}
        if local: env.pop('ASCEND_RT_VISIBLE_DEVICES')
        return env

    def measurement_fallback(self, log):
        return '[flagos cpu_fallback]' in log

    def docker_options(self, device, args):
        opts = ['--ipc=host', '--shm-size=8g']
        if args.allow_privileged_root:
            opts += ['--privileged']
        for path in ['/dev/davinci_manager', '/dev/hisi_hdc', '/dev/devmm_svm', f'/dev/davinci{device}']:
            opts += ['--device', f'{path}:{path}:rwm']
        for path in ['/usr/local/dcmi', '/usr/local/Ascend/driver', '/usr/local/Ascend/firmware',
                     '/usr/local/Ascend/version.info', '/usr/local/bin/npu-smi', '/etc/ascend_install.info']:
            if Path(path).exists():
                opts += ['-v', f'{path}:{path}:ro']
        return opts

    def lease_root(self):
        # Interoperate with the existing Base lease; a second namespace is unsafe.
        return base_common().DEFAULT_LEASE_ROOT

    def preflight(self, root, devices, label='preflight'):
        base_common()
        from executors.toolkit import run_host_preflight
        expected = sorted(int(p.name[7:]) for p in Path('/dev').glob('davinci[0-9]*') if p.name[7:].isdigit())
        path = run_host_preflight(root, expected, device_ids=','.join(map(str, devices)), label=label)
        return json.loads(path.read_text())

    def diagnose(self, error, root=None):
        logs = ''.join(p.read_text() for p in root.glob('*-error.json')) if root is not None else ''
        logs += ''.join(p.read_text() for p in root.glob('*.log')) if root is not None else ''
        if 'backend not registered' in error and ('torch_fl/' in logs or 'flag_gems/' in logs):
            return 'dependency-blocked: selected Torch-FL dispatch has no registered backend; no library patch applied'
        if root is not None and 'too many values to unpack' in error:
            if 'flag_gems/utils/random_utils.py' in logs:
                return 'dependency-blocked: FlagGems Philox RNG-state layout does not match this runtime'
        if 'SPLIT_K' in error and 'unrecognised' in error and 'triton/runtime/jit.py' in logs and 'flag_gems/' in logs:
            return 'dependency-blocked: FlagGems autotune/kernel SPLIT_K argument incompatible with Triton signature'
        if "Expected a value of type 'Tensor'" in error and 'aten::mul' in error and 'flag_gems/ops/mul.py' in logs:
            return 'dependency-blocked: FlagGems scalar mul redispatch selected the Tensor overload'
        return None

    def profile_call(self, frame):
        """Observe an existing launcher; never wrap or replace backend functions."""
        if frame.f_code.co_filename.endswith('/triton/backends/ascend/driver.py') and frame.f_code.co_name == '__call__':
            launcher = frame.f_locals.get('self')
            return {'class': type(launcher).__name__,
                    'compile_only': getattr(launcher, 'compile_only', None),
                    'register_tensor_only': getattr(launcher, 'enable_msprof_register_tensor', None)}
        return None

    def route(self, probe, log):
        target = log.split('OPERATION_TARGET_BEGIN', 1)[-1].split('OPERATION_TARGET_END', 1)[0]
        complete = log.count('OPERATION_TARGET_BEGIN') == log.count('OPERATION_TARGET_END') == 1
        fallback = '[flagos cpu_fallback]' in target
        dispatch = re.findall(r'\[flagos dispatch\]\s+([^\n]+)', target)
        calls = probe.get('calls', [])
        gems = [c for c in calls if 'flag_gems' in c['path'] and ('/ops/' in c['path'] or '/fused/' in c['path'])]
        launches = [c for c in calls if 'triton' in c['path'] and c['function'] in ('run', 'launch')]
        runners = [c for c in calls if c['path'].endswith('/triton/compiler/compiler.py') and c['function'] == 'runner']
        drivers = [c for c in calls if c['path'].endswith('/triton/backends/ascend/driver.py') and c['function'] == '__call__']
        executing = [c for c in drivers if any(s.get('class') == 'NPULauncher' and
                     s.get('compile_only') is False and s.get('register_tensor_only') is False
                     for s in c.get('launcher_states', []))]
        skipped = 'skip running kernel' in log or any(s.get('compile_only') is True or
                  s.get('register_tensor_only') is True for c in drivers for s in c.get('launcher_states', []))
        cached_launch = bool(runners and executing and not skipped)
        if cached_launch: launches += runners + executing
        # A transpose alone does not prove the matmul inside a Linear/mv target.
        meaningful = [d for d in dispatch if d.split(' -> ')[0] not in {'t','transpose.int','view','detach','reshape','clone','copy_','_to_copy','empty.memory_format','empty_like','empty_strided','new_empty','as_strided','alias'}]
        native = any('-> ascend' in d for d in meaningful)
        devices = probe.get('output_devices', [])
        on_device = bool(devices) and all(d == probe.get('device') and d.startswith('flagos:') for d in devices)
        proven = complete and on_device and not skipped and (native or (gems and launches))
        return {'status': 'failed' if fallback else 'passed' if proven else 'partial',
                'target_cpu_fallback': fallback, 'dispatch': dispatch, 'implementation': gems,
                'jit_launches': launches, 'output_devices': devices,
                'cached_launch_proven': cached_launch,
                'boundary': 'target interval incl. backward if selected; Python/JIT and Torch-FL dispatch evidence, not hardware kernel profiling'}
