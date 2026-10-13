"""Diagnostic capture selected by vendor; never used in formal total timing."""
def profiler(cfg, root, warmup, active):
    from runtime.adapters import vendor_adapter
    return vendor_adapter(cfg).profiler(cfg, root, warmup, active)
