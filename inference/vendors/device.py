"""Compatibility facade for explicit vendor selection; no device fallback."""
from runtime.adapters import vendor_adapter


def environment(cfg):
    return vendor_adapter(cfg).environment(cfg)


def initialize(cfg):
    return vendor_adapter(cfg).initialize(cfg)


def stream_id(backend):
    """Compatibility for raw backends; unknown or ambiguous representations fail."""
    stream = backend.current_stream()
    matches = [getattr(stream, name) for name in ('npu_stream', 'cuda_stream') if hasattr(stream, name)]
    if len(matches) != 1:
        raise ValueError('unknown or ambiguous vendor stream representation')
    return int(matches[0])
