# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Explicit Kunlunxin P800 Operation profile and conservative route gates."""
import hashlib
import json
from importlib import metadata
from pathlib import Path
import os
import re
import sys
import time

from vendors.compat import Adapter as CompatAdapter

PROFILE = Path(__file__).parent / 'profiles/p800-xpytorch29.json'


class Adapter(CompatAdapter):
    strict_worker_lifecycle = True
    postflight_after_preflight = True
    supports_shared_device = True

    def __init__(self):
        super().__init__('kunlunxin')
        self.lock = json.loads(PROFILE.read_text())
        self.host_session = None
        self.device = None
        self.framework_observation = None
        self._source_fingerprint = None

    def default_image(self):
        return self.lock['image']

    def validate_options(self, args, devices):
        if args.execution == 'docker' and any(d not in self.lock['expected_device_ids'] for d in devices):
            raise ValueError('P800 Docker physical device IDs must be unique and in 0..7')
        if args.allow_privileged_root or args.container_device or args.mount:
            raise ValueError('P800 uses managed device mappings and mounts; privileged/additional mappings are not supported')

    def task_metadata(self, args, device):
        shared = bool(getattr(args, 'allow_shared_device', False))
        return {'worker_device': None if args.execution == 'docker' else device,
                'runtime_config': {'runtime_profile': self.lock['runtime_profile'],
                                   'profile_sha256': hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
                                   'source_sha256': self.source_fingerprint(),
                                   'execution': args.execution,
                                   'expected_image_id': self.lock['image_id'] if args.execution == 'docker' else None,
                                   'sharing_policy': 'shared-authorized' if shared else 'exclusive-required',
                                   'performance_qualification': 'partial-shared-host' if shared else 'exclusive-host-required'},
                'device_binding': {'status': 'pending-host-and-framework-observation' if args.execution == 'docker'
                                  else 'externally-managed', 'requested_id': device}}

    def runtime_identity(self, image, info):
        if info.get('Id') != self.lock['image_id']:
            raise ValueError('P800 image ID differs from the Operation runtime lock')
        return {'runtime_profile': self.lock['runtime_profile'], 'image': image, 'image_id': info['Id'],
                'validated': False, 'validation': 'image identity recorded; worker performs interpreter/package checks; Operation is not promoted',
                'expected_packages': self.lock['packages'], 'packages_observed': False,
                'execution_status': 'candidate-runtime'}

    def ensure_execution_ready(self):
        if not PROFILE.is_file() or self.lock.get('runtime_profile') != 'p800-xpytorch29':
            raise RuntimeError('P800 runtime profile is missing or inconsistent')

    def bootstrap(self, device_id):
        raise RuntimeError('P800 bootstrap requires a task and observed framework UUIDs')

    def bootstrap_task(self, task):
        from .bootstrap import base_helper
        if task['runtime_config']['source_sha256'] != self.source_fingerprint():
            raise RuntimeError('P800 worker source fingerprint differs from task')
        actual_case = {'case':task['case'], 'runtime_config':{}}
        self.task_identity(actual_case)
        if any(task['runtime_config'].get(k) != v for k,v in actual_case['runtime_config'].items()):
            raise RuntimeError('P800 worker case/default configuration fingerprint differs from task')
        self.bootstrap_record = json.loads(Path(os.environ['P800_OPERATION_BOOTSTRAP_RECORD']).read_text())
        if self.bootstrap_record['cpu_isolated'] or self.bootstrap_record['profile_sha256'] != task['runtime_config']['profile_sha256']:
            raise RuntimeError('worker bootstrap/profile differs from task')
        import torch
        import torch_xmlir
        if not torch.cuda.is_available():
            raise RuntimeError('P800 backend unavailable')
        uuids = [str(torch.cuda.get_device_properties(i).uuid).lower() for i in range(torch.cuda.device_count())]
        if len(set(uuids)) != len(uuids): raise ValueError('duplicate framework UUIDs')
        if task['runtime_config']['execution'] == 'docker':
            binding = base_helper('device_mapping').resolve_logical_devices([task['device_binding']], uuids)[0]
            index = binding['logical_device']
        else:
            index = task['worker_device']
            if not 0 <= index < len(uuids): raise ValueError('local logical device outside visible framework set')
            binding = {'status':'externally-managed', 'logical_device':index, 'uuid':uuids[index]}
        torch.cuda.set_device(index)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.api = torch.cuda
        self.device = torch.device(f'cuda:{index}')
        self.framework_observation = {**binding, 'framework_logical_id':index, 'observed_uuids':uuids}
        return self.device

    def identity(self):
        from .bootstrap import ENV_KEYS
        return {'vendor': 'kunlunxin', 'runtime_profile': self.lock['runtime_profile'],
                'device': str(self.device), 'framework_observation': self.framework_observation,
                'packages': self.bootstrap_record['packages'], 'executable':self.bootstrap_record['executable'],
                'hook_hashes': self.bootstrap_record['hook_hashes'],
                'environment': {k:os.environ.get(k) for k in ENV_KEYS},
                'profile_sha256': hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
                'hardware_validation': 'device/backend identity observed; runtime remains candidate'}

    def synchronize(self):
        self.api.synchronize(self.device)

    def memory_stats(self):
        raw = self.api.memory_stats(self.device.index)
        return {**{f'{k}_bytes':raw.get(f'{k}_bytes.all.current') for k in ('allocated','reserved')},
                **{f'peak_{k}_bytes':raw.get(f'{k}_bytes.all.peak') for k in ('allocated','reserved')}}

    def reset_peak_memory_stats(self):
        import torch
        torch.cuda.reset_peak_memory_stats(self.device)

    def kernel_time(self, fn, cfg):
        return None

    def profile_target(self, root, task, call):
        from .profiling import target_window
        target_window(self, root, task, call)

    def collect_profile(self, root, task):
        from .profiling import collect
        collect(root, task)

    def environment(self, device, mode='probe', local=False):
        from .bootstrap import CPU_PHASES
        cpu = mode in CPU_PHASES
        env = {'DO_NOT_TRACK':'1', 'TORCH_DEVICE_BACKEND_AUTOLOAD':'0',
               'USE_FLAGGEMS':'0', 'XRAY_DEBUG_ENABLE':'0', 'DISABLE_XPYTORCH':'1' if cpu else '0',
               'P800_OPERATION_EXECUTION':'local' if local else 'docker'}
        profiling = mode in ('profiling', 'profile-target')
        env['XPU_ENABLE_PROFILER_TRACING'] = '1' if mode == 'trace' or profiling else '0'
        env['XPU_CUPTI_ENABLE_DEVICE'] = str(device if local else 0) if profiling else ''
        if not local:
            env.update(CUDA_VISIBLE_DEVICES='' if cpu else '0',
                       PATH=self.lock['conda_prefix'] + '/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin')
        return env

    worker_environment = environment

    def worker_command(self, local=False):
        return [sys.executable if local else self.lock['conda_prefix'] + '/bin/python', '-B', '-S', '-X', 'faulthandler']

    worker_entry = 'vendors/kunlunxin/bootstrap.py'

    def docker_options(self, device, args):
        self.validate_options(args, [device])
        if self.host_session is None:
            raise RuntimeError('P800 device mapping requires a host session')
        binding = self.host_session.binding(device)
        return ['--device', binding['host_device_node'] + ':' + binding['container_node'] + ':rwm',
                '--device', '/dev/xpuctrl:/dev/xpuctrl:rwm', '--ipc=private',
                '--cap-drop=ALL', '--security-opt=no-new-privileges', '--pids-limit=512']

    def lease_root(self):
        from .host import base_modules
        return base_modules()[0].DEFAULT_LEASE_ROOT

    def local_lease_root(self):
        return Path('/tmp/flagperf-operation-kunlunxin-local-leases')

    def create_lease(self, root, devices, run_id, allow_shared=False):
        from .host import HostSession
        self.host_session = HostSession(root, devices, run_id, self.lock, allow_shared=allow_shared)
        return self.host_session.prepare_lease()

    def preflight(self, root, devices, label='preflight'):
        if self.host_session is None or self.host_session.root != Path(root) or self.host_session.devices != list(devices):
            raise RuntimeError('P800 preflight does not match its host session')
        return self.host_session.preflight(label)

    def bind_tasks(self, tasks, host):
        for task in tasks:
            binding = self.host_session.binding(task['device_id'])
            task['device_binding'] = {**binding, 'status': 'host-observed; framework-probe-required',
                                      'framework_logical_id': None}
            task['runtime_config'] = {**task['runtime_config'], 'physical_uuid': binding['uuid'],
                                      'pci_bdf': binding['pci_bdf'], 'device_minor': binding['device_minor']}
            task['worker_device'] = None

    def snapshot_paths(self):
        repo = Path(__file__).resolve().parents[3]
        names = ['base/executors/__init__.py','base/vendors/__init__.py',
                 'base/executors/common.py','base/executors/lifecycle.py','base/vendors/protocol.py',
                 'base/vendors/registry.py','base/vendors/ascend/provider.py','base/vendors/kunlunxin/provider.py',
                 'base/vendors/kunlunxin/preflight.py','base/vendors/kunlunxin/reuse.py','base/benchmarks/day6_contract.py']
        prefix = 'base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/'
        names += [prefix + n for n in ('stack.lock.yaml','image-manifest.json','validation-summary.json',
                                       'device_mapping.py','verify_runtime.py')]
        return [repo / n for n in names]

    def source_fingerprint(self):
        if self._source_fingerprint is None:
            from runtime.cli import snapshot_files
            root = Path(__file__).resolve().parents[3]
            files = [p for p in snapshot_files() if not ('benchmarks' in p.parts and p.name in ('main.py','case_config.yaml'))]
            files += self.snapshot_paths()
            operation = Path(__file__).resolve().parents[2]
            entries = {('operation/' + p.relative_to(operation).as_posix() if p.is_relative_to(operation)
                        else p.relative_to(root).as_posix()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
            self._source_fingerprint = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
        return self._source_fingerprint

    def task_identity(self, task):
        root = Path(__file__).resolve().parents[2] / 'benchmarks' / task['case']
        task['runtime_config'].update({name:hashlib.sha256((root/file).read_bytes()).hexdigest()
                                      for name,file in [('case_source_sha256','main.py'),('case_defaults_sha256','case_config.yaml')]})

    def prepare_replay(self, task, args):
        # The source task remains sealed under source-task/. A new replay must
        # identify its actual current generator, not claim the old source hash.
        task.update(self.task_metadata(args, task['device_id']))
        self.task_identity(task)

    def cleanup_worker(self, worker, run_id):
        from .host import cleanup_worker
        began = time.monotonic()
        result = cleanup_worker(worker, run_id)
        return {**result, 'elapsed_s':time.monotonic() - began}

    def cleanup_process(self, proc):
        from .host import cleanup_process
        began = time.monotonic()
        result = cleanup_process(proc)
        return {**result, 'elapsed_s':time.monotonic() - began}

    def route_observer(self, task, backward):
        from .routing import Observer
        return Observer(task, backward)

    def route(self, probe, log):
        from .routing import route
        return route(probe, log)

    def check_cpu_construction(self, tensors, fn):
        import torch
        values = list(tensors)
        if isinstance(fn, torch.nn.Module): values += list(fn.parameters()) + list(fn.buffers())
        if any(t.device.type != 'cpu' for t in values):
            raise RuntimeError('case construction moved tensors off CPU before explicit input transfer')

    def worker_identity(self, task, gems_enabled):
        if (task['oplib'] == 'flaggems') != gems_enabled:
            raise RuntimeError('explicit FlagGems registration does not match worker oplib')
        return {'registration':{'oplib':task['oplib'], 'public_enable_called':gems_enabled,
                                'automatic_gems':'disabled'}, 'runtime_config':task['runtime_config']}

    def audit_phase(self, root, phase):
        from .bootstrap import CPU_PHASES
        record = json.loads(Path(os.environ['P800_OPERATION_BOOTSTRAP_RECORD']).read_text())
        import torch
        cpu = phase in CPU_PHASES
        imported = any(n == 'torch_xmlir' or n.startswith(('torch_xmlir.', 'flag_gems')) for n in sys.modules)
        if cpu and (imported or torch.cuda.is_initialized()):
            raise RuntimeError('CPU phase imported/initialized an accelerator backend')
        if cpu and os.environ.get('P800_OPERATION_EXECUTION') == 'docker' and list(Path('/dev').glob('xpu*')):
            raise RuntimeError('CPU container unexpectedly has XPU device mappings')
        from runtime.evidence import write
        write(root / (phase + '-bootstrap.json'), {**record, 'cpu_backend_imported':imported if cpu else None,
                                                  'cuda_initialized':torch.cuda.is_initialized()})

    def qualify_result(self, item, probe):
        measured = item['measurement']
        fields = ('vendor_runtime','packages','case_sha256','input_sha256','registration','runtime_config')
        mismatches = [key for key in fields if key not in probe or key not in measured or probe[key] != measured[key]]
        item['phase_consistency'] = {'status':'failed' if mismatches else 'passed', 'mismatches':mismatches}
        if mismatches:
            item.update(status='failed', failure_stage='phase-consistency', diagnosis='probe/measure identity differs')
        shared = item.get('runtime_config',{}).get('sharing_policy') == 'shared-authorized'
        item['performance_qualification'] = {'status':'partial' if shared else 'passed',
            'scope':'shared device: timings are diagnostic only' if shared else 'exclusive host checks; other gates still apply'}
        if shared and item['status'] == 'passed':
            item.update(status='partial', diagnosis='shared device: no exclusive performance qualification')

    def supplementary_gate(self, item, phase):
        # Independent diagnostic workers are safe after a successful same-input
        # measurement, even when route proof/exclusive performance is incomplete.
        # Failed execution, fallback, mismatched identity or correctness never opt in.
        if (phase in ('memory', 'profiling') and item.get('status') == 'partial'
                and item.get('correctness', {}).get('status') == 'passed'
                and item.get('measurement', {}).get('correctness', {}).get('status') == 'passed'
                and item.get('phase_consistency', {}).get('status') == 'passed'
                and item.get('routing', {}).get('status') in ('passed', 'partial')
                and item.get('measurement_fallback_observed') is False
                and not item.get('routing', {}).get('target_cpu_fallback')
                and not any(item.get(k) for k in ('error', 'diagnostic_execution_blocked',
                                                'cleanup_error', 'postflight_error'))):
            return {'allowed': True, 'scope': 'diagnostic-only',
                    'reason': 'correctness and phase identity passed; route or exclusive performance remains incomplete; capture does not qualify the task'}
        return None

    def qualify_supplement(self, item, phase):
        record = item[phase]
        if phase != 'memory' or record.get('status') not in ('measured', 'partial'):
            return
        fields = ('vendor_runtime', 'packages', 'case_sha256', 'input_sha256',
                  'registration', 'runtime_config', 'device', 'input_devices')
        measured = item['measurement']
        mismatches = [k for k in fields if k not in record or k not in measured or record[k] != measured[k]]
        record['phase_consistency'] = {'status': 'failed' if mismatches else 'passed', 'mismatches': mismatches}
        if mismatches:
            record.update(status='partial', reason='memory replay identity differs from clean measurement')
            # Keep raw observations but do not display mismatched values as the target's memory.
            for key in list(record):
                if key.endswith('_bytes'):
                    record.pop(key)

    def diagnose(self, error, root=None):
        if root is not None and 'uni_sram PassManager::run failed' in error:
            for path in Path(root).glob('*-error.json'):
                saved = json.loads(path.read_text())
                log = path.with_name(path.name.replace('-error.json','.log'))
                if saved.get('type') != 'OutOfResources' or not log.is_file(): continue
                trace = log.read_text().split('Traceback (most recent call last):')[-1]
                frames = re.findall(r'File "([^"]+)", line \d+, in (\w+)', trace)
                if (any(p.endswith('/flag_gems/runtime/backend/_kunlunxin/ops/mm.py') and f == 'mm' for p,f in frames)
                        and any(p.endswith('/triton/backends/xpu/compiler.py') and f == 'make_ttxir' for p,f in frames)
                        and 'triton.runtime.errors.OutOfResources:' in trace
                        and 'uni_sram PassManager::run failed' in saved.get('error','')):
                    return 'dependency-blocked: locked FlagGems mm XPU compiler fails its MLIR pipeline (uni_sram); no kernel/protocol patch applied'
        return None
