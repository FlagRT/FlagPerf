#!/usr/bin/env python3
"""Isolated PR0 probes. Run only inside an explicitly reserved one-card container.

The supervisor starts with -S. Only gated worker processes enable site hooks.
This produces candidate evidence, never a validated runtime or a benchmark.
"""
import argparse
import faulthandler
from datetime import datetime, timezone
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from verify_runtime import ROOT, PREFIX, sha256, validate_manifest, validate_packages, normalize

STAGES = ('import_device', 'fp32', 'seed', 'memory', 'pinned_copy', 'event')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_binding(binding, environ, nodes, now=None):
    require(binding.get('schema_version') == 1, 'binding schema must be 1')
    physical = binding.get('host_physical_id')
    require(type(physical) is int and 0 <= physical <= 7, 'physical card must be 0..7')
    container_node = binding.get('container_node')
    host_node = binding.get('host_device_node', '')
    require(bool(re.fullmatch(r'/dev/xpu[0-7]', host_node)), 'invalid host device node')
    require(container_node == host_node, 'unsupported container node mapping')
    require(binding.get('device_minor') == int(host_node[-1]), 'device minor mismatch')
    require(bool(re.fullmatch(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
                              binding.get('uuid', ''))), 'device UUID required')
    require(binding.get('logical_device') == 0, 'single-card logical device must be 0')
    require(bool(binding.get('reservation_reference')), 'reservation reference required')
    require(bool(re.fullmatch(r'[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]',
                              binding.get('pci_bdf', ''))), 'host PCI BDF required')
    start = datetime.fromisoformat(binding['reservation_start'].replace('Z', '+00:00'))
    end = datetime.fromisoformat(binding['reservation_end'].replace('Z', '+00:00'))
    require(start.tzinfo is not None and end.tzinfo is not None, 'reservation needs timezone')
    now = now or datetime.now(timezone.utc)
    require(start <= now < end, 'outside reservation window')
    visible = binding.get('cuda_visible_devices')
    require(visible == '0', 'single-card visibility must be zero')
    require(environ.get('CUDA_VISIBLE_DEVICES') == visible, 'visibility must match binding')
    require(environ.get('XPU_VISIBLE_DEVICES') is None and binding.get('xpu_visible_devices') is None, 'native XPU visibility must be unset')
    require(set(nodes) == {container_node, '/dev/xpuctrl'}, 'unexpected or missing device nodes')
    require(environ.get('XPU_EVENT_KL3_ENABLE') is None, 'KL3 must be unset for native probe')
    require(environ.get('USE_FLAGGEMS') in (None, '', '0'), 'FlagGems is a separate qualification scope')
    return end


def loaded_libraries():
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith('/') and '.so' in parts[5]:
            path = parts[5]
            if any(key in Path(path).name.lower() for key in ('xpu', 'xmlir', 'bkcl', 'xccl', 'cuda', 'flagcx')) or '/torch_xmlir/' in path:
                paths.add(path)
    return [{'path': p, 'sha256': sha256(p)} for p in sorted(paths) if Path(p).is_file()]


def worker(stage, binding, output_dir):
    # This function is entered only after binding and image checks in main().
    faulthandler.dump_traceback_later(20, repeat=True)
    if os.environ.get('P800_PROFILE_ROUTE') == '1':
        os.environ['XPU_TRACING_OUTPUT_NAME'] = str(output_dir/(stage+'-native-trace'))
        os.environ['XPU_PRINT_API_SUMMARY'] = '1'
    import site
    site.main()
    packages = {normalize(d.metadata.get('Name', '')): {'version': d.version}
                for d in metadata.distributions()}
    errors = validate_packages(packages, json.loads((ROOT/'stack.lock.yaml').read_text())['packages'])
    require(not errors, f'package lock mismatch: {errors}')
    import torch
    import torch_xmlir
    print(json.dumps({'diagnostic': 'backend-discovery', 'torch_file': torch.__file__,
                      'torch_xmlir_version': str(torch_xmlir.__version__),
                      'device_count': torch.cuda.device_count(),
                      'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                      'loaded_libraries': loaded_libraries()}), flush=True)
    require(torch.cuda.is_available(), 'cuda-compatible P800 backend unavailable')
    require(torch.cuda.device_count() == 1, 'expected exactly one visible accelerator')
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    name = torch.cuda.get_device_name(0)
    # This XPYTORCH build exposes the generic label GPU even on P800.
    # Host PCI identity plus actual opened nodes must establish the mapping.
    require('P800' in name.upper() or name == 'GPU', f'unexpected accelerator: {name}')
    properties = torch.cuda.get_device_properties(0)
    require(str(properties.uuid).lower() == binding['uuid'], 'framework UUID differs from selected physical card')
    print('checkpoint: device properties read', flush=True)
    result = {'stage': stage, 'torch': torch.__version__, 'device': str(device),
              'device_name': name, 'torch_xmlir_path': torch_xmlir.__file__,
              'properties': str(properties),
              'property_fields': {key: str(getattr(properties, key)) for key in dir(properties)
                                  if not key.startswith('_') and not callable(getattr(properties, key))}}
    result['device_node_numbers'] = {str(p): {'major': os.major(p.stat().st_rdev),
                                            'minor': os.minor(p.stat().st_rdev)}
                                     for p in Path('/dev').glob('xpu*')}
    print(json.dumps({'diagnostic': 'device-properties', **result}), flush=True)
    if stage == 'import_device':
        print('checkpoint: creating device tensor', flush=True)
        tensor = torch.ones(4, device=device)
        print('checkpoint: device tensor created', flush=True)
        require(tensor.device == device, 'tensor device mismatch')
        torch.cuda.synchronize(device)
        print('checkpoint: device synchronized', flush=True)
        require(tensor.cpu().tolist() == [1.0] * 4, 'device tensor readback failed')
        result['properties'] = str(torch.cuda.get_device_properties(0))
    elif stage in ('fp32', 'route_control'):
        torch.manual_seed(519)
        a, b = torch.randn(32, 32), torch.randn(32, 32)
        reference = a.double() @ b.double()
        ad, bd = a.to(device), b.to(device)
        torch.cuda.synchronize(device)
        if stage == 'route_control':
            result.update(shape=[32, 32, 32], matmul_calls=0,
                          control='same input copies and synchronization, no matmul')
            result['loaded_libraries'] = loaded_libraries()
            faulthandler.cancel_dump_traceback_later()
            return result
        start = time.perf_counter()
        output = ad @ bd
        torch.cuda.synchronize(device)
        result['wall_seconds_with_sync'] = time.perf_counter() - start
        require(output.device == device and output.dtype == torch.float32, 'output device/dtype mismatch')
        actual = output.cpu().double()
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)
        result.update(shape=[32, 32, 32], max_abs_error=(actual-reference).abs().max().item(),
                      rtol=1e-4, atol=1e-4, fallback_status='requires selected-card telemetry review')
        if os.environ.get('P800_PROFILE_ROUTE') == '1':
            # Vendor cu_xpu_launch_async is not represented in this build's
            # Kineto kernel events. Compare native launch counters against a
            # separate input-copy-only control, with no second CUPTI subscriber.
            for _ in range(3):
                output = ad @ bd
            torch.cuda.synchronize(device)
            result['matmul_calls'] = 4
            torch.testing.assert_close(output.cpu().double(), reference, rtol=1e-4, atol=1e-4)
    elif stage == 'seed':
        torch.cuda.manual_seed_all(519)
        first = torch.rand(32, device=device)
        torch.cuda.synchronize(device)
        torch.cuda.manual_seed_all(519)
        second = torch.rand(32, device=device)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(first.cpu(), second.cpu(), rtol=0, atol=0)
        result['seed'] = 519
    elif stage == 'memory':
        free, total = torch.cuda.mem_get_info(0)
        require(0 < free <= total, 'invalid free/total memory')
        oom = getattr(torch.cuda, 'OutOfMemoryError', None)
        require(isinstance(oom, type) and issubclass(oom, Exception), 'OOM exception API unavailable')
        result.update(free_bytes=free, total_bytes=total, memory_stats=torch.cuda.memory_stats(0),
                      oom_exception=f'{oom.__module__}.{oom.__name__}', oom_allocation_tested=False)
    elif stage == 'pinned_copy':
        source = torch.arange(256, dtype=torch.float32).pin_memory()
        require(source.is_pinned(), 'pin_memory did not pin buffer')
        target = torch.empty(256, device=device)
        destination = torch.empty(256, pin_memory=True)
        target.copy_(source, non_blocking=True)
        torch.cuda.synchronize(device)
        destination.copy_(target, non_blocking=True)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(destination, source, rtol=0, atol=0)
        result['payload_bytes'] = source.numel() * source.element_size()
    elif stage == 'event':
        first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        tensor = torch.ones(256, device=device)
        first.record()
        tensor.add_(1)
        last.record()
        torch.cuda.synchronize(device)
        require(last.query(), 'event not complete after device synchronization')
        elapsed = first.elapsed_time(last)
        result.update(elapsed_ms=elapsed, event_timing_usable=math.isfinite(elapsed) and elapsed > 0,
                      recommended_timer='perf_counter with device synchronize')
    result['loaded_libraries'] = loaded_libraries()
    opened_nodes = set()
    for descriptor in Path('/proc/self/fd').iterdir():
        try:
            target = os.readlink(descriptor)
        except FileNotFoundError:
            continue
        if target.startswith('/dev/xpu'):
            opened_nodes.add(target)
    result['opened_device_nodes'] = sorted(opened_nodes)
    faulthandler.cancel_dump_traceback_later()
    return result


def run_isolated(argv, directory, stage, timeout):
    stdout_path, stderr_path = directory / f'{stage}.stdout.log', directory / f'{stage}.stderr.log'
    started = time.monotonic()
    with stdout_path.open('w') as stdout, stderr_path.open('w') as stderr:
        process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    return {'stage': stage, 'returncode': process.returncode, 'timed_out': timed_out,
            'duration_seconds': time.monotonic()-started,
            'stdout': stdout_path.name, 'stderr': stderr_path.name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-candidate', action='store_true')
    parser.add_argument('--binding-json', type=Path, required=True)
    parser.add_argument('--inspect-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=45)
    parser.add_argument('--stage', choices=STAGES+('route_control',), help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(sys.flags.no_site, 'use bootstrap or python -S; gate must precede site hooks')
    require(args.allow_candidate, 'explicit --allow-candidate required')
    require(1 <= args.timeout <= 120, 'timeout must be 1..120 seconds')
    validate_manifest(json.loads((ROOT/'image-manifest.json').read_text()),
                      json.loads(args.inspect_json.read_text()))
    binding = json.loads(args.binding_json.read_text())
    end = validate_binding(binding, os.environ, [str(p) for p in Path('/dev').glob('xpu*')])
    if args.stage:
        result = worker(args.stage, binding, args.output_dir)
        (args.output_dir/f'{args.stage}.json').write_text(json.dumps(result, indent=2)+'\n')
        return 0
    # A fresh directory prevents stale successful results from masking a failed worker.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {'validated': False, 'release_stage': 'candidate', 'binding': binding,
               'image_id': json.loads((ROOT/'image-manifest.json').read_text())['image_id'],
               'stages': [], 'status': 'running', 'scope': 'native single-card PR0 probes',
               'physical_mapping_verified': False, 'cpu_fallback_excluded': False}
    summary_path = args.output_dir/'summary.json'
    stages = STAGES
    if os.environ.get('P800_PROFILE_ROUTE') == '1':
        stages = ('import_device', 'route_control')+STAGES[1:]
    for stage in stages:
        remaining = (end-datetime.now(timezone.utc)).total_seconds()
        if remaining < args.timeout+10:
            summary['status'] = 'reservation_expiring'
            break
        argv = [str(PREFIX/'bin/python'), '-S', str(Path(__file__).resolve()),
                '--allow-candidate', '--binding-json', str(args.binding_json.resolve()),
                '--inspect-json', str(args.inspect_json.resolve()),
                '--output-dir', str(args.output_dir.resolve()), '--stage', stage]
        record = run_isolated(argv, args.output_dir, stage, args.timeout)
        stderr = (args.output_dir/f'{stage}.stderr.log').read_text()
        record['native_api_summary_present'] = 'API calls end' in stderr
        record['native_kernel_launch_calls'] = sum(int(count) for count in re.findall(
            r'name=\s*cu_xpu_launch_async,\s*count=\s*(\d+)', stderr))
        record['passed'] = record['returncode'] == 0 and (args.output_dir/f'{stage}.json').is_file()
        summary['stages'].append(record)
        summary_path.write_text(json.dumps(summary, indent=2)+'\n')
        if not record['passed'] and stage in ('import_device', 'fp32'):
            summary['status'] = 'core_failed'
            break
    else:
        summary['status'] = 'probes_passed_review_required' if all(r['passed'] for r in summary['stages']) else 'partial'
    summary_path.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    return 0 if summary['status'] == 'probes_passed_review_required' else 1


if __name__ == '__main__':
    raise SystemExit(main())
