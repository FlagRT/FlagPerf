"""Deterministic CPU references for floating-point matrix multiplication."""
import torch


def compare(actual, reference, *, atol, rtol):
    if actual.shape != reference.shape:
        raise RuntimeError('correctness shape mismatch')
    actual = actual.detach().cpu().double()
    reference = reference.double()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    difference = (actual - reference).abs()
    allowed = atol + rtol * reference.abs()
    failures = (~torch.isfinite(actual)) | (~torch.isfinite(reference)) | (difference > allowed)
    passed = finite and not bool(failures.any())
    return {'passed': passed, 'finite': finite, 'atol': atol, 'rtol': rtol,
            'max_abs_error': difference.max().item() if finite else None,
            'max_relative_error': (difference / reference.abs().clamp_min(atol)).max().item() if finite else None,
            'relative_denominator': 'max(abs(reference), atol)',
            'failed_elements': int(failures.sum()),
            'first_failure': failures.nonzero()[0].tolist() if bool(failures.any()) else None}


def input_pair(rows, inner, columns, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    left = torch.randn(rows, inner, generator=generator, dtype=torch.float32) / inner**0.5
    right = torch.randn(inner, columns, generator=generator, dtype=torch.float32)
    return left, right


def sampled_reference(left, right, actual, *, atol, rtol):
    rows = sorted({0, left.shape[0] - 1, *[index * left.shape[0] // 16 for index in range(16)]})
    columns = sorted({0, right.shape[1] - 1, *[index * right.shape[1] // 16 for index in range(16)]})
    reference = left[rows].double() @ right[:, columns].double()
    cpu_output = actual.detach().cpu()
    record = compare(cpu_output[rows][:, columns], reference, atol=atol, rtol=rtol)
    record.update(scope='fixed row/column cross-product, full reduction dimension', rows=rows, columns=columns,
                  reduction_size=left.shape[1], full_output_finite=bool(torch.isfinite(cpu_output).all()))
    record['passed'] = record['passed'] and record['full_output_finite']
    return record
