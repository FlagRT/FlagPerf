"""T-Head PPU Operation adapter for the CUDA-compatible PPU SDK."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from vendors.compat import Adapter as CompatAdapte

PROFILE = Path(__file__).parent / 'profiles/torch210-ppu.json'


class Adapter(CompatAdapter):
    name = 'thead'
    strict_worker_lifecycle = True
    postflight_after_preflight = True
    supports_shared_device = True

    def __init__(self):
        super().__init__('thead')
        self.lock = json.loads(PROFILE.read_text())
        self.host_session = None
        self.device = None

    def default_image(self): return self.lock['image']

    def validate_options(self, args, devices):
        if any(d not in self.lock['expected_device_ids'] for d in devices):
            raise ValueError('PPU device IDs must be unique and in 0..15')
        if args.allow_privileged_root or args.container_device or args.mount:
            raise ValueError('PPU uses managed device mappings; privileged/additional mappings are unsupported')

    def runtime_identity(self, image, info):
        if info.get('Id') != self.lock['image_id'] or info.get('Architecture') != 'amd64':
            raise ValueError('PPU image ID differs from runtime profile')
        return {'runtime_profile': self.lock['runtime_profile'], 'image': image, 'image_id': info['Id'],
                'validated': False, 'validation': 'candidate runtime identity recorded; Operation is not promoted',
                'hardware_model': self.lock['hardware_model'], 'execution_status': 'candidate-runtime'}

    def task_metadata(self, args, device):
        return {'runtime_config': {'runtime_profile': self.lock['runtime_profile'],
                                   'profile_sha256': hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
                                   'execution': args.execution,
                                   'sharing_policy': 'shared-authorized' if args.allow_shared_device else 'exclusive-required',
                                   'performance_qualification': 'partial-shared-host' if args.allow_shared_device else 'exclusive-host-required'},
                'device_binding': {'status': 'pending-host-observation' if args.execution == 'docker' else 'externally-managed',
                                   'requested_id': device}}

    def ensure_execution_ready(self):
        if not PROFILE.is_file(): raise RuntimeError('PPU runtime profile is missing')

    def bootstrap(self, device_id):
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('PPU CUDA-compatible backend unavailable')
        if not 0 <= device_id < torch.cuda.device_count(): raise RuntimeError('PPU local device outside visible range')
        torch.cuda.set_device(device_id); self.api = torch.cuda; self.device = torch.device('cuda:%d' % device_id)
        torch.backends.cuda.matmul.allow_tf32 = False
        return self.device

    def bootstrap_task(self, task):
        device = self.bootstrap(task['worker_device'])
        if task['runtime_config']['execution'] == 'docker':
            if torch_device_count() != 1:
                raise RuntimeError('PPU Docker selection must expose exactly one visible device')
            expected = task.get('device_binding', {}).get('uuid')
            observed = normalize_uuid(self._observed_uuid(0))
            if not expected or normalize_uuid(expected) != observed:
                raise RuntimeError('PPU framework UUID differs from host selected device')
            self.framework_observation = {'framework_logical_id': 0, 'uuid': observed,
                                          'host_physical_id': task['device_id']}
        else:
            self.framework_observation = {'framework_logical_id': task['worker_device'],
                                          'uuid': normalize_uuid(self._observed_uuid(task['worker_device']))}
        return device

    def _observed_uuid(self, index):
        return str(self.api.get_device_properties(index).uuid)

    def identity(self):
        import torch
        record_path = os.environ.get('PPU_OPERATION_BOOTSTRAP_RECORD')
        record = json.loads(Path(record_path).read_text()) if record_path else {}
        return {'vendor': 'thead', 'chip': 'PPU', 'runtime_profile': self.lock['runtime_profile'],
                'device': str(self.device), 'device_name': torch.cuda.get_device_name(self.device),
                'framework_observation': self.framework_observation,
                'torch': torch.__version__, 'cuda_compatibility_mode': True,
                **{k: record.get(k) for k in ('executable', 'profile_sha256', 'packages', 'hook_hashes')},
                'fp32_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                'hardware_validation': 'device/backend identity observed; runtime remains candidate'}

    def synchronize(self): self.api.synchronize(self.device)

    def memory_stats(self):
        raw = self.api.memory_stats(self.device.index)
        return {**{f'{k}_bytes': raw.get(f'{k}_bytes.all.current') for k in ('allocated','reserved')},
                **{f'peak_{k}_bytes': raw.get(f'{k}_bytes.all.peak') for k in ('allocated','reserved')}}

    def reset_peak_memory_stats(self): self.api.reset_peak_memory_stats(self.device)
    def kernel_time(self, fn, cfg): return None

    def environment(self, device, mode='probe', local=False):
        cpu = mode in ('reference','reference-check','diagnose')
        env = {'DO_NOT_TRACK':'1', 'TORCH_DEVICE_BACKEND_AUTOLOAD':'0', 'USE_FLAGGEMS':'0',
               'CUDA_DEVICE_ORDER':'PCI_BUS_ID', 'PPU_OPERATION_EXECUTION':'local' if local else 'docker'}
        if cpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        elif not local:
            env['CUDA_VISIBLE_DEVICES'] = '0'
        if mode in ('probe','profiling','profile-target'): env['CUDA_LAUNCH_BLOCKING'] = '0'
        return env

    worker_environment = environment
    def worker_command(self, local=False): return [sys.executable if local else '/usr/local/bin/python3', '-B', '-S', '-X', 'faulthandler']
    worker_entry = 'vendors/thead/bootstrap.py'

    def docker_options(self, device, args):
        if self.host_session is None: raise RuntimeError('PPU Docker mapping requires locked host session')
        binding = self.host_session.binding(device)
        node = binding['host_device_node']
        return ['--device', node + ':' + node + ':rwm', '--device', '/dev/alixpu:/dev/alixpu:rwm',
                '--ipc=private', '--shm-size=16g', '--cap-drop=ALL', '--security-opt=no-new-privileges', '--pids-limit=512']

    def create_lease(self, root, devices, run_id, allow_shared=False):
        from .host import HostSession
        self.host_session = HostSession(root, devices, run_id, self.lock, allow_shared=allow_shared)
        return self.host_session.prepare_lease()

    def preflight(self, root, devices, label='preflight'):
        if self.host_session is None: raise RuntimeError('PPU host session is missing')
        return self.host_session.preflight(label)

    def cleanup_worker(self, worker, run_id):
        from .host import cleanup_worke
        return cleanup_worker(worker, run_id)

    def cleanup_process(self, proc):
        from .host import cleanup_process
        return cleanup_process(proc)

    def bind_tasks(self, tasks, host):
        for task in tasks:
            binding = self.host_session.binding(task['device_id'])
            task['device_binding'] = {**binding, 'status': 'host-observed; worker UUID check applies'}
            task['runtime_config'] = {**task['runtime_config'], 'physical_uuid': binding['uuid'],
                                      'pci_bdf': binding['pci_bdf'], 'device_minor': binding['device_minor']}
            task['worker_device'] = 0

    def diagnose(self, error, root=None):
        text = error + '\n' + (''.join(p.read_text(errors='replace') for p in root.glob('*.log')) if root else '')
        if 'flag_gems' in text and any(x in text for x in ('CompilationError', 'OutOfResources', 'CUDA_ERROR_INVALID_PTX', 'No module named')):
            return 'dependency-blocked: FlagGems/compiler failure signature; input and call attribution still require triage; original traceback retained'
        if 'cuda' in text.lower() and any(x in text.lower() for x in ('device-side assert', 'invalid device function', 'launch failed')):
            return 'runtime-blocked: CUDA-compatible launch failure signature; root cause not established; original traceback retained'
        return None

    def route(self, probe, log):
        from .routing import route
        return route(probe, log)

    def route_observer(self, task, backward):
        from .routing import Observe
        return Observer(task, backward)

    def probe_window(self, root, observer, name):
        from .dispatch import capture
        return capture(root, self, observer, name)

    def profile_target(self, root, task, call):
        from .profiling import target_window
        target_window(self, root, task, call)

    def collect_profile(self, root, task):
        from .profiling import collect
        collect(root, task)

    def measurement_fallback(self, log): return False

    def check_cpu_construction(self, tensors, fn):
        import torch
        values = list(tensors)
        if isinstance(fn, torch.nn.Module):
            values += list(fn.parameters()) + list(fn.buffers())
        if any(t.device.type != 'cpu' for t in values):
            raise RuntimeError('PPU case construction moved tensors off CPU before explicit transfer')

    def verify_execution_payload(self, root, bundle, inputs, fn, phase):
        from .execution import verify_payload
        verify_payload(root, self, bundle, inputs, fn, phase)

    def worker_identity(self, task, gems_enabled):
        if (task['oplib'] == 'flaggems') != gems_enabled:
            raise RuntimeError('explicit FlagGems registration does not match PPU worker path')
        return {'registration': {'oplib': task['oplib'], 'public_enable_called': gems_enabled,
                                 'automatic_gems': 'disabled'}, 'runtime_config': task['runtime_config']}

    def task_identity(self, task):
        root = Path(__file__).resolve().parents[2] / 'benchmarks' / task['case']
        task['runtime_config'].update({name: hashlib.sha256((root / filename).read_bytes()).hexdigest()
                                       for name, filename in (('case_source_sha256', 'main.py'),
                                                              ('case_defaults_sha256', 'case_config.yaml'))})

    def prepare_replay(self, task, args):
        task.update(self.task_metadata(args, task['device_id']))
        self.task_identity(task)

    def audit_phase(self, root, phase):
        import torch
        record_path = Path(os.environ.get('PPU_OPERATION_BOOTSTRAP_RECORD', ''))
        record = json.loads(record_path.read_text()) if record_path.is_file() else {}
        cpu = phase in ('reference', 'diagnose', 'reference-check')
        imported = any(n == 'flag_gems' or n.startswith('flag_gems.') for n in sys.modules)
        if cpu and (torch.cuda.is_initialized() or imported):
            raise RuntimeError('PPU CPU phase imported FlagGems or initialized CUDA backend')
        if cpu and os.environ.get('PPU_OPERATION_EXECUTION') == 'docker' and list(Path('/dev').glob('alixpu*')):
            raise RuntimeError('PPU CPU container unexpectedly has accelerator device mappings')
        from runtime.evidence import write
        write(root / (phase + '-bootstrap.json'), {**record, 'cuda_initialized': torch.cuda.is_initialized(),
                                                   'cpu_backend_imported': imported if cpu else None})

    def qualify_result(self, item, probe):
        measured = item.get('measurement', {})
        fields = ('vendor_runtime', 'packages', 'case_sha256', 'input_sha256', 'registration', 'runtime_config', 'device', 'input_devices')
        mismatches = [key for key in fields if key not in probe or key not in measured or probe[key] != measured[key]]
        item['phase_consistency'] = {'status': 'failed' if mismatches else 'passed', 'mismatches': mismatches}
        if mismatches:
            item.update(status='failed', failure_stage='phase-consistency', diagnosis='probe/measure identity differs')
        shared = item['runtime_config']['sharing_policy'] == 'shared-authorized'
        local = item['runtime_config']['execution'] == 'local'
        qualified = not (shared or local) and item.get('status') == 'passed'
        item['performance_qualification'] = {'status': 'passed' if qualified else 'partial',
            'scope': 'shared-authorized' if shared else 'externally managed local device' if local else 'host exclusivity checked',
            'reason': None if qualified else 'independent exclusive device and complete routing evidence required'}
        if (shared or local) and item.get('status') == 'passed':
            item.update(status='partial', failure_stage='performance-qualification',
                        diagnosis='shared device explicitly authorized; no exclusive performance qualification' if shared else
                                  'local host reservation is externally managed; no host exclusivity proof')

    def supplementary_gate(self, item, phase):
        if (phase in ('memory', 'profiling') and item.get('status') == 'partial'
                and item.get('correctness', {}).get('status') == 'passed'
                and item.get('measurement', {}).get('correctness', {}).get('status') == 'passed'
                and item.get('phase_consistency', {}).get('status') == 'passed'
                and item.get('routing', {}).get('status') in ('passed', 'partial')
                and item.get('measurement_fallback_observed') is False
                and not item.get('routing', {}).get('target_cpu_fallback')
                and not any(item.get(k) for k in ('error', 'cleanup_error', 'postflight_error'))):
            return {'allowed': True, 'scope': 'diagnostic-only',
                    'reason': 'correctness and phase identity passed; original route/performance status is retained'}
        return {'allowed': False, 'scope': None, 'reason': 'requires passed correctness and phase identity'}

    def qualify_supplement(self, item, phase):
        if phase != 'memory':
            return
        record, measured = item[phase], item['measurement']
        keys = ('vendor_runtime', 'packages', 'case_sha256', 'input_sha256',
                'registration', 'runtime_config', 'device', 'input_devices')
        mismatches = [k for k in keys if k not in record or k not in measured or record[k] != measured[k]]
        record['phase_consistency'] = {'status': 'failed' if mismatches else 'passed', 'mismatches': mismatches}
        if mismatches:
            record.update(status='partial', reason='memory/measure identity differs')
            for key in list(record):
                if key.endswith('_bytes'):
                    record.pop(key)

    def snapshot_paths(self):
        repo = Path(__file__).resolve().parents[3]
        return [repo / 'base/vendors/thead/torch210_ppu_candidate/image-manifest.json',
                repo / 'base/vendors/thead/torch210_ppu_candidate/stack.lock.yaml',
                *[repo / path for path in ('base/executors/common.py', 'base/executors/lifecycle.py',
                  'base/vendors/thead/provider.py', 'base/vendors/thead/preflight.py',
                  'base/vendors/thead/runtime.py', 'base/vendors/protocol.py',
                  'base/vendors/registry.py', 'base/monitoring/thead_usage.py')]]


def normalize_uuid(value):
    return str(value).lower().removeprefix('gpu-')


def torch_device_count():
    import torch
    return torch.cuda.device_count()
