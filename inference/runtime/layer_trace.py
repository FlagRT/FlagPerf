"""Offline trace coordination; decoders and recovery are vendor-owned."""
from pathlib import Path
from runtime.common import execute, write_json
from runtime.trace_dispatch import decoder


def attribute(events, calls):
    """Compatibility entry for explicitly supplied legacy CANN events."""
    from vendors.ascend.trace import attribute as impl
    return impl(events, calls)


def associate_collectives(detail):
    from runtime.cuda_analysis_job import PublishedResult
    if isinstance(detail, PublishedResult):
        return  # Already done inside the bounded CUDA worker, before publishing.
    from vendors.ascend.trace import associate_collectives as impl
    return impl(detail)


def write_attribution(folder, detail):
    from runtime.cuda_analysis_job import PublishedResult
    path = Path(folder).resolve()/'attribution.json'
    if isinstance(detail, PublishedResult) and detail.published_at == path:
        if not path.is_file():
            raise ValueError('published CUDA attribution is missing')
        return
    write_json(path, detail)


def capture_hashes(source):
    from vendors.ascend.trace import capture_hashes as impl
    return impl(source)


def recover_trace(folder, export=None):
    from vendors.ascend.trace import recover_trace as impl
    return impl(folder, export=export or execute)


def analyze(folder, state, export=None, *, timeout=900, rank=None):
    try:
        selected = decoder(folder)
    except ValueError as error:
        return {'status': 'partial', 'reason': str(error), 'coverage': {}, 'rows': [], 'events': []}
    if selected.__name__ == 'vendors.nvidia.trace':
        from runtime.cuda_analysis_job import run
        return run(folder, state, rank=rank, timeout=timeout)
    return selected.analyze(folder, state, export=export or execute, rank=rank)
