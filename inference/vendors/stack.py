"""Compatibility facade; implementation is vendor-owned."""
from runtime.adapters import vendor_adapter, required_components, create


def __getattr__(name):
    if name in ('PARENT_IMAGE', 'STACK_IMAGE'):
        from vendors.ascend.adapter import profile
        return profile()['compiler_source_image' if name == 'PARENT_IMAGE' else 'image_id']
    raise AttributeError(name)


def selection(cfg):
    return cfg.get('_stack', {'flaggems': 'off', 'flagtree': 'off', 'flagcx': 'off'})


def validate_bundle(folder):
    from vendors.ascend.compiler import validate_bundle as validate
    return validate(folder)


def worker_environment(cfg, root):
    return vendor_adapter(cfg).worker_environment(cfg, root, 'compiler' in required_components(cfg))


def compiler_identity(cfg):
    if 'compiler' not in required_components(cfg):
        return {'provider': 'not_collected', 'observation': 'not_collected'}
    return create('components', 'compiler').identity(cfg)


def communication_backend(cfg):
    return vendor_adapter(cfg).communication_backend(cfg, selection(cfg))
