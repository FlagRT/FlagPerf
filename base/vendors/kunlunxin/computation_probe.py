"""Small, context-bound capability observations; never performance qualification."""
import hashlib
import json
from pathlib import Path
import traceback


PRECISIONS = ('FP16', 'BF16', 'INT8', 'FP64', 'FP8', 'TF32')


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
