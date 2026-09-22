"""Small, context-bound capability observations; never performance qualification."""
import hashlib
import json
from pathlib import Path
import traceback


PRECISIONS = ('FP16', 'BF16', 'INT8', 'INT8-ROUTES', 'MATMUL-ROUTES', 'XTORCH-INT8', 'BF16-ROUTES', 'FP64', 'FP8', 'TF32')


def probe(precision, device, identity, output):
    import torch
    import torch_xmlir
    record = {'schema_version': 1, **identity, 'precision': precision, 'tests': [],
              'scope': 'bounded API and numerical observations, not performance or universal fallback exclusion'}
    generator = torch.Generator().manual_seed(519)

    def attempt(name, operation, reference):
        item = {'name': name, 'status': 'failed'}
        record['tests'].append(item)
        try:
            actual = operation()
            torch.cuda.synchronize(device)
            item.update(dtype=str(actual.dtype), device=str(actual.device), shape=list(actual.shape))
            cpu = actual.cpu().double()
            expected = reference.double()
            difference = (cpu - expected).abs()
            item.update(status='observed', finite=bool(torch.isfinite(cpu).all()),
                        max_abs_error=difference.max().item(), actual=cpu.tolist(), reference=expected.tolist())
        except Exception as exc:
            item.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        (output / 'capability.json').write_text(json.dumps(record, indent=2) + '\n')

    if precision in ('FP16', 'BF16', 'FP64'):
        dtype = {'FP16': torch.float16, 'BF16': torch.bfloat16, 'FP64': torch.float64}[precision]
        left = (torch.randn(17, 29, generator=generator).double() / 29**0.5).to(dtype)
        right = torch.randn(29, 11, generator=generator).to(dtype)
        attempt('torch.mm', lambda: torch.mm(left.to(device), right.to(device)), left.double() @ right.double())
        if precision == 'FP64':
            left = torch.tensor([[1 + 2**-40, 1.0]], dtype=dtype)
            right = torch.tensor([[1.0], [-1.0]], dtype=dtype)
            attempt('double-not-float32', lambda: torch.mm(left.to(device), right.to(device)), left @ right)
    elif precision == 'INT8':
        left = torch.randint(-3, 4, (16, 32), generator=generator, dtype=torch.int8)
        right = torch.randint(-3, 4, (32, 16), generator=generator, dtype=torch.int8)
        reference = left.long() @ right.long()
        attempt('torch.mm', lambda: torch.mm(left.to(device), right.to(device)), reference)
        attempt('torch._int_mm', lambda: torch._int_mm(left.to(device), right.to(device)), reference)
    elif precision == 'FP8':
        for format_name in ('float8_e4m3fn', 'float8_e5m2'):
            dtype = getattr(torch, format_name)
            left = torch.randint(-2, 3, (16, 32), generator=generator).float().to(dtype)
            right = torch.randint(-2, 3, (16, 32), generator=generator).float().to(dtype).t()
            reference = left.float().double() @ right.float().double()
            attempt(format_name + ':torch.mm', lambda: torch.mm(left.to(device), right.to(device)), reference)
            attempt(format_name + ':torch._scaled_mm',
                    lambda: torch._scaled_mm(left.to(device), right.to(device),
                                             scale_a=torch.ones(1, device=device), scale_b=torch.ones(1, device=device),
                                             out_dtype=torch.float32), reference)
            attempt(format_name + ':device-cast:_scaled_mm',
                    lambda: torch._scaled_mm(left.float().to(device).to(dtype), right.float().to(device).to(dtype),
                                             scale_a=torch.ones(1, device=device), scale_b=torch.ones(1, device=device),
                                             out_dtype=torch.float32), reference)
    elif precision == 'INT8-ROUTES':
        import time as _time
        record['route_scope'] = 'timing decomposition only; not a qualification and not a universal fallback exclusion'
        record['torch_threads'] = torch.get_num_threads()
        record['cpu_name'] = open('/proc/cpuinfo').read().split('model name')[1].split(':')[1].split('\n')[0].strip()

        def timed(where, size, iterations, label):
            item = {'name': label, 'status': 'failed', 'shape': [size, size, size], 'iterations': iterations}
            record['tests'].append(item)
            try:
                generator = torch.Generator().manual_seed(519)
                left = torch.randint(-8, 9, (size, size), generator=generator, dtype=torch.int8).to(where)
                right = torch.randint(-8, 9, (size, size), generator=generator, dtype=torch.int8).to(where)
                for _ in range(2):
                    torch._int_mm(left, right)
                if where != torch.device('cpu'):
                    torch.cuda.synchronize(device)
                wall0, cpu0, thread0 = _time.perf_counter(), _time.process_time(), _time.thread_time()
                for _ in range(iterations):
                    result = torch._int_mm(left, right)
                if where != torch.device('cpu'):
                    torch.cuda.synchronize(device)
                wall = _time.perf_counter() - wall0
                cpu = _time.process_time() - cpu0
                thread = _time.thread_time() - thread0
                item.update(status='observed', wall_seconds=wall, process_cpu_seconds=cpu, thread_cpu_seconds=thread,
                            tops=2 * size ** 3 * iterations / wall / 1e12, ms_per_iteration=wall / iterations * 1e3,
                            process_cpu_over_wall=cpu / wall, thread_cpu_over_wall=thread / wall,
                            result_dtype=str(result.dtype), result_device=str(result.device))
                del left, right, result
            except Exception as exc:
                item.update(error_type=type(exc).__name__, error=str(exc)[:400], traceback=traceback.format_exc()[-800:])
            (output / 'capability.json').write_text(json.dumps(record, indent=2) + '\n')

        for size, iterations in ((1024, 20), (2048, 10), (4096, 3), (8192, 1)):
            timed(device, size, iterations, 'device_int_mm_%d' % size)
        for size, iterations in ((1024, 20), (2048, 10), (4096, 3)):
            timed(torch.device('cpu'), size, iterations, 'cpu_int_mm_%d' % size)

        # Transfer-only cost for the same tensors a host-executed 2048 int8 matmul needs each call.
        transfer = {'name': 'transfer_round_trip_2048', 'status': 'failed'}
        record['tests'].append(transfer)
        try:
            generator = torch.Generator().manual_seed(519)
            left = torch.randint(-8, 9, (2048, 2048), generator=generator, dtype=torch.int8)
            right = torch.randint(-8, 9, (2048, 2048), generator=generator, dtype=torch.int8)
            result = torch._int_mm(left, right)
            left_dev, right_dev, result_dev = left.to(device), right.to(device), result.to(device)
            torch.cuda.synchronize(device)
            wall0, cpu0 = _time.perf_counter(), _time.process_time()
            for _ in range(10):
                back = left_dev.cpu(), right_dev.cpu(), result_dev.cpu()
            wall = _time.perf_counter() - wall0
            transfer.update(status='observed', iterations=10, wall_seconds=wall, process_cpu_seconds=_time.process_time() - cpu0,
                            ms_per_iteration=wall / 10 * 1e3,
                            bytes_per_iteration=int(left.numel() * 2 + result.numel() * 4) * 3,
                            scope='three int8 operands plus int32 result round-tripped once; host-execution lower bound')
        except Exception as exc:
            transfer.update(error_type=type(exc).__name__, error=str(exc)[:400])

        # Case-parity candidates: scaled int8 matmul with bf16 output (the Ascend contract).
        for label, call in (
                ('int8_scaled_mm_bf16', lambda a, b: torch._scaled_mm(a, b, scale_a=torch.ones(1, device=device),
                                                                       scale_b=torch.ones(1, device=device),
                                                                       out_dtype=torch.bfloat16)),
                ('int8_scaled_mm_fp32', lambda a, b: torch._scaled_mm(a, b, scale_a=torch.ones(1, device=device),
                                                                      scale_b=torch.ones(1, device=device),
                                                                      out_dtype=torch.float32)),
                ('int8_mm_aten', lambda a, b: torch.mm(a, b))):
            attempt(label, lambda call=call: call(torch.ones(64, 64, dtype=torch.int8, device=device),
                                                 torch.ones(64, 64, dtype=torch.int8, device=device)),
                    torch.full((64, 64), 64.0))
        record['aten_int_mm_schema'] = str(getattr(torch.ops.aten._int_mm, '_schemas', None))
        try:
            import xtorch_ops  # noqa: F401
            record['xtorch_ops_symbols'] = [name for name in dir(xtorch_ops)
                                            if any(word in name.lower() for word in ('int8', 'matmul', 'mm', 'quant'))][:60]
        except Exception as exc:
            record['xtorch_ops_symbols'] = {'import_error': type(exc).__name__ + ': ' + str(exc)[:200]}
        (output / 'capability.json').write_text(json.dumps(record, indent=2) + '\n')
    elif precision == 'MATMUL-ROUTES':
        import time as _time
        record['route_scope'] = 'device-execution evidence via process CPU time; timing only, not a qualification'
        record['torch_threads'] = torch.get_num_threads()

        def run_mm(where, dtype, size, iterations, label, operator=None):
            item = {'name': label, 'status': 'failed', 'shape': [size, size, size], 'iterations': iterations,
                    'dtype': str(dtype)}
            record['tests'].append(item)
            try:
                generator = torch.Generator().manual_seed(519)
                op = operator or torch.mm
                if dtype in (torch.float16, torch.bfloat16, torch.float32):
                    left = (torch.randn(size, size, generator=generator) / size ** 0.5).to(dtype)
                    right = (torch.randn(size, size, generator=generator) / size ** 0.5).to(dtype)
                else:
                    left = torch.randint(-8, 9, (size, size), generator=generator, dtype=dtype)
                    right = torch.randint(-8, 9, (size, size), generator=generator, dtype=dtype)
                left, right = left.to(where), right.to(where)
                for _ in range(2):
                    op(left, right)
                if where != torch.device('cpu'):
                    torch.cuda.synchronize(device)
                wall0, cpu0 = _time.perf_counter(), _time.process_time()
                for _ in range(iterations):
                    result = op(left, right)
                if where != torch.device('cpu'):
                    torch.cuda.synchronize(device)
                wall = _time.perf_counter() - wall0
                cpu = _time.process_time() - cpu0
                item.update(status='observed', wall_seconds=wall, process_cpu_seconds=cpu,
                            tops=2 * size ** 3 * iterations / wall / 1e12, ms_per_iteration=wall / iterations * 1e3,
                            process_cpu_over_wall=cpu / wall, result_dtype=str(result.dtype), result_device=str(result.device))
                del left, right, result
            except Exception as exc:
                item.update(error_type=type(exc).__name__, error=str(exc)[:300], traceback=traceback.format_exc()[-600:])
            (output / 'capability.json').write_text(json.dumps(record, indent=2) + '\n')

        for dtype, name in ((torch.float16, 'fp16'), (torch.bfloat16, 'bf16'), (torch.float32, 'fp32')):
            run_mm(device, dtype, 4096, 3, 'device_mm_%s_4096' % name)
        # xtorch_ops int8 device kernels: schemas plus a bounded small-shape call.
        try:
            import xtorch_ops
            namespace = getattr(torch.ops, 'xtorch_ops', None)
            names = [name for name in dir(xtorch_ops) if 'gemm' in name.lower() or 'I8' in name]
            record['xtorch_gemm_symbols'] = sorted(names)[:40]
            schemas = {}
            for name in sorted(names):
                target = getattr(namespace, name, None) if namespace is not None else None
                schemas[name] = str(getattr(target, '_schemas', None))[:400]
            record['xtorch_gemm_schemas'] = schemas
            generator = torch.Generator().manual_seed(519)
            small_a = torch.randint(-8, 9, (64, 128), generator=generator, dtype=torch.int8).to(device)
            small_b = torch.randint(-8, 9, (64, 128), generator=generator, dtype=torch.int8).to(device)
            record['xtorch_call_probes'] = []
            for name in sorted(names):
                target = getattr(namespace, name, None) if namespace is not None else None
                if target is None:
                    continue
                entry = {'name': name}
                try:
                    actual = target(small_a, small_b)
                    torch.cuda.synchronize(device)
                    entry.update(status='returned', dtype=str(actual.dtype), device=str(actual.device), shape=list(actual.shape))
                except Exception as exc:
                    entry.update(status='raised', error_type=type(exc).__name__, error=str(exc)[:300])
                record['xtorch_call_probes'].append(entry)
                (output / 'capability.json').write_text(json.dumps(record, indent=2) + '\n')
        except Exception as exc:
            record['xtorch_gemm_schemas'] = {'import_error': type(exc).__name__ + ': ' + str(exc)[:300]}
        (output / 'capability.json').write_text(json.dumps(record, indent=2) + '\n')
    elif precision == 'XTORCH-INT8':
        record['route_scope'] = 'device int8 kernel convention discovery; not a qualification'

        def note(name, **values):
            item = {'name': name, 'status': values.pop('status', 'observed'), **values}
            record['tests'].append(item)
            (output / 'capability.json').write_text(json.dumps(record, indent=2) + '\n')
            print('[probe]', json.dumps(item)[:700], flush=True)

        import xtorch_ops
        package = Path(xtorch_ops.__file__).parent
        hits = []
        for source in sorted(package.rglob('*.py')):
            try:
                content = source.read_text(errors='replace')
            except OSError:
                continue
            for number, line in enumerate(content.splitlines(), 1):
                if 'gemm_I8_I8_bf16_nt' in line or ('I8_I8' in line and 'quant' in line.lower()):
                    hits.append({'path': str(source.relative_to(package)), 'line': number, 'text': line.strip()[:220]})
        note('package_usage_hits', package=str(package), count=len(hits), hits=hits[:25])

        kernel = getattr(xtorch_ops, 'gemm_I8_I8_bf16_nt')

        def call(a_int8, b_int8, a_scale, b_scale, out):
            kernel((a_int8, a_scale), (b_int8, b_scale), out)

        # Ones inputs at unit scale: if the accumulate is a plain int32 sum the
        # output must be exactly K everywhere.
        for size in (128, 256, 512):
            blocks = (size + 127) // 128
            a_int8 = torch.ones(size, size, dtype=torch.int8, device=device)
            b_int8 = torch.ones(size, size, dtype=torch.int8, device=device)
            for scale in (1.0, 2.0, 0.5):
                label = 'ones_%d_scale_%s' % (size, scale)
                try:
                    out = torch.zeros(size, size, dtype=torch.bfloat16, device=device)
                    a_scale = torch.full((size, blocks), scale, dtype=torch.float32, device=device)
                    b_scale = torch.full((size, blocks), scale, dtype=torch.float32, device=device)
                    call(a_int8, b_int8, a_scale, b_scale, out)
                    torch.cuda.synchronize(device)
                    value = out.float().flatten()[0].item()
                    note(label, first_element=value, expected_sum=float(size), ratio_to_sum=value / size,
                         unique=len(torch.unique(out.float()).tolist()))
                except Exception as exc:
                    note(label, status='failed', error_type=type(exc).__name__, error=str(exc)[:200])

        # Scale orientation and magnitude effects with a fixed random input pair.
        generator = torch.Generator().manual_seed(519)
        size = 256
        blocks = (size + 127) // 128
        a_int8 = torch.randint(-8, 9, (size, size), generator=generator, dtype=torch.int8).to(device)
        b_int8 = torch.randint(-8, 9, (size, size), generator=generator, dtype=torch.int8).to(device)
        reference = a_int8.cpu().long() @ b_int8.cpu().long().t()
        for label, a_shape, b_shape, value in (
                ('scales_row_block_1', (size, blocks), (size, blocks), 1.0),
                ('scales_block_row_1', (blocks, size), (blocks, size), 1.0),
                ('scales_128_1', (size, 1), (size, 1), 1.0),
                ('scales_row_block_2', (size, blocks), (size, blocks), 2.0),
                ('scales_row_block_half', (size, blocks), (size, blocks), 0.5)):
            try:
                out = torch.zeros(size, size, dtype=torch.bfloat16, device=device)
                a_scale = torch.full(a_shape, value, dtype=torch.float32, device=device)
                b_scale = torch.full(b_shape, value, dtype=torch.float32, device=device)
                call(a_int8, b_int8, a_scale, b_scale, out)
                torch.cuda.synchronize(device)
                result = out.cpu().double()
                ratio = (result.abs().max().item() / reference.double().abs().max().item()) if reference.double().abs().max() > 0 else None
                note(label, max_output=result.abs().max().item(), max_reference=reference.double().abs().max().item(),
                     amplitude_ratio=ratio)
            except Exception as exc:
                note(label, status='failed', error_type=type(exc).__name__, error=str(exc)[:200])

        # Maybe the second tuple member is written by the kernel rather than read.
        try:
            out = torch.zeros(size, size, dtype=torch.bfloat16, device=device)
            a_scale = torch.zeros(size, blocks, dtype=torch.float32, device=device)
            b_scale = torch.zeros(size, blocks, dtype=torch.float32, device=device)
            call(a_int8, b_int8, a_scale, b_scale, out)
            torch.cuda.synchronize(device)
            note('workspace_written', a_scale_after=str(a_scale.flatten()[:4].tolist()),
                 b_scale_after=str(b_scale.flatten()[:4].tolist()),
                 a_scale_nonzero=int((a_scale != 0).sum().item()), b_scale_nonzero=int((b_scale != 0).sum().item()))
        except Exception as exc:
            note('workspace_written', status='failed', error_type=type(exc).__name__, error=str(exc)[:200])
        (output / 'capability.json').write_text(json.dumps(record, indent=2) + '\n')
    elif precision == 'BF16-ROUTES':
        import time as _time
        record['route_scope'] = 'vendor grouped bf16 kernel as a single-group dense GEMM; not a qualification'

        def note(name, **values):
            item = {'name': name, 'status': values.pop('status', 'observed'), **values}
            record['tests'].append(item)
            (output / 'capability.json').write_text(json.dumps(record, indent=2) + '\n')
            print('[probe]', json.dumps(item)[:600], flush=True)

        import xtorch_ops
        generator = torch.Generator().manual_seed(519)
        kernel = getattr(xtorch_ops, 'm_grouped_gemm_bf16_bf16_bf16_nt_contiguous_v3')

        def build(size):
            left = (torch.randn(size, size, generator=generator) / size ** 0.5).to(torch.bfloat16).to(device)
            right = (torch.randn(size, size, generator=generator) / size ** 0.5).to(torch.bfloat16).to(device)
            out = torch.empty(size, size, dtype=torch.bfloat16, device=device)
            return left, right, out

        # Single group: every row belongs to group 0. Try the two plausible index
        # encodings and keep whichever returns clean results.
        for label, make_indices in (('m_indices_zeros_M', lambda size: torch.zeros(size, dtype=torch.int32, device=device)),
                                    ('m_indices_zeros_1', lambda size: torch.zeros(1, dtype=torch.int32, device=device))):
            try:
                size = 4096
                left, right, out = build(size)
                indices = make_indices(size)
                kernel(left, right, out, indices)
                torch.cuda.synchronize(device)
                rows = [0, 1, size - 1]
                columns = [0, 5, size - 1]
                reference = left[rows].double() @ right.t()[:, columns].double()
                actual = out[rows][:, columns].double()
                error = (actual - reference).abs().max().item()
                note('grouped_%s_correctness' % label, max_abs_error=error,
                     reference_scale=reference.abs().max().item(), indices_shape=list(indices.shape))
            except Exception as exc:
                note('grouped_%s_correctness' % label, status='failed',
                     error_type=type(exc).__name__, error=str(exc)[:250])

        # Time the working encoding at the qualification shape.
        size = 8192
        left, right, out = build(size)
        indices = torch.zeros(size, dtype=torch.int32, device=device)

        def call():
            kernel(left, right, out, indices)

        try:
            for _ in range(2):
                call()
            torch.cuda.synchronize(device)
            wall0, cpu0 = _time.perf_counter(), _time.process_time()
            for _ in range(3):
                call()
            torch.cuda.synchronize(device)
            wall = _time.perf_counter() - wall0
            note('grouped_bf16_8192', ms_per_iteration=wall / 3 * 1e3, tops=2 * size ** 3 * 3 / wall / 1e12,
                 process_cpu_over_wall=(_time.process_time() - cpu0) / wall)
        except Exception as exc:
            note('grouped_bf16_8192', status='failed', error_type=type(exc).__name__, error=str(exc)[:250])

        # Reference points measured in the same container for a clean comparison.
        fp16_left = left.to(torch.float16)
        fp16_right = right.to(torch.float16)

        def fp16_call():
            torch.mm(fp16_left, fp16_right)

        for _ in range(2):
            fp16_call()
        torch.cuda.synchronize(device)
        wall0 = _time.perf_counter()
        for _ in range(3):
            fp16_call()
        torch.cuda.synchronize(device)
        wall = _time.perf_counter() - wall0
        note('aten_fp16_8192_reference', ms_per_iteration=wall / 3 * 1e3, tops=2 * size ** 3 * 3 / wall / 1e12)

        def bf16_call():
            torch.mm(left, right)

        for _ in range(2):
            bf16_call()
        torch.cuda.synchronize(device)
        wall0 = _time.perf_counter()
        for _ in range(3):
            bf16_call()
        torch.cuda.synchronize(device)
        wall = _time.perf_counter() - wall0
        note('aten_bf16_8192_reference', ms_per_iteration=wall / 3 * 1e3, tops=2 * size ** 3 * 3 / wall / 1e12)
        (output / 'capability.json').write_text(json.dumps(record, indent=2) + '\n')
    elif precision == 'TF32':
        record['initial_allow_tf32'] = torch.backends.cuda.matmul.allow_tf32
        left = torch.tensor([[1 + 2**-12, 1.0]], dtype=torch.float32).repeat(16, 16)
        right = torch.tensor([[1.0], [-1.0]], dtype=torch.float32).repeat(16, 16)
        for enabled in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = enabled
            attempt('allow_tf32=' + str(enabled), lambda: torch.mm(left.to(device), right.to(device)), left.double() @ right.double())
        torch.backends.cuda.matmul.allow_tf32 = record['initial_allow_tf32']
    record['extension_path'] = str(Path(torch_xmlir.__file__).parent)
    record['api_schemas'] = {name: str(getattr(getattr(torch.ops.aten, name, None), '_schemas', None))
                             for name in ('mm', '_int_mm', '_scaled_mm')}
    record['extension_symbols'] = [name for name in dir(torch_xmlir) if any(word in name.lower() for word in ('matmul', 'precision', 'tf32', 'int8', 'fp8'))]
    record['source_precision_mentions'] = []
    for source in sorted(Path(torch_xmlir.__file__).parent.rglob('*.py')):
        content = source.read_text(errors='replace')
        matches = [{'line': number, 'text': line[:400]} for number, line in enumerate(content.splitlines(), 1)
                   if any(word in line.lower() for word in ('tf32', '_int_mm', '_scaled_mm', 'float64', 'float8'))]
        if matches:
            record['source_precision_mentions'].append({'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'matches': matches[:40]})
    (output / 'capability.json').write_text(json.dumps(record, indent=2) + '\n')
