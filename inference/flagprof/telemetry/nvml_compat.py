# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Shared discovery and identity reads for the NVML-ABI compatible libraries
the CUDA-compat vendor stacks ship.

PPU and Kunlunxin stacks both expose the standard nvml* symbols through a
compatible shared library, so one reader (vendors/nvidia/telemetry.NVML)
serves all three vendors. The monitor binds a target to telemetry entities
by entity string, so the identity reported by a target must be the exact
UUID string the telemetry provider records - both sides therefore read the
UUID from the same library here.

The monitor worker runs with -S (no site-packages), where importlib cannot
see installed packages, so library discovery uses sysconfig and plain
filesystem globs.
"""
import ctypes as C


def candidate_libraries(extra=()):
    """Libraries to try, in order of preference.

    1. FLAGPROF_TELEMETRY_LIBRARY, an explicit operator override;
    2. vendor-specific ``extra`` paths (e.g. the XPU install tree);
    3. the platform loader's standard NVML sonames;
    4. the NVML-ABI library the running interpreter's own environment
       bundles under triton/backends/*/*/so (derived from sysconfig, so no
       installation path is hard-coded).
    """
    import glob
    import os
    import sysconfig
    candidates = []
    override = os.environ.get('FLAGPROF_TELEMETRY_LIBRARY')
    if override:
        candidates.append(override)
    candidates += [path for path in extra if path]
    candidates += ['/usr/lib64/libnvidia-ml.so', '/usr/lib/libnvidia-ml.so',
                   '/usr/lib64/libnvidia-ml.so.1', '/usr/lib/libnvidia-ml.so.1',
                   '/usr/local/lib64/libnvidia-ml.so', 'libnvidia-ml.so.1']
    try:
        purelib = sysconfig.get_paths()['purelib']
    except (KeyError, TypeError):
        purelib = None
    if purelib:
        candidates += sorted(glob.glob(purelib+'/triton/backends/*/*/so/libxpunvidia-ml.so'))
        candidates += sorted(glob.glob(purelib+'/triton/backends/*/*/so/libxpunvidia-ml.so.*'))
    return tuple(candidates)


def load(library=None):
    """Load the first candidate that answers nvmlInit_v2.

    Returns ``(lib, path, attempts)`` where ``attempts`` records every path
    tried with its failure reason, so callers can surface a diagnosable error.
    Raises OSError whose message enumerates all attempts when none works.
    """
    paths = (library,) if library else candidate_libraries()
    attempts = []
    for path in paths:
        try:
            lib = C.CDLL(path)
            code = lib.nvmlInit_v2()
            if code == 0:
                return lib, path, attempts
            attempts.append(path + ': nvmlInit_v2=' + str(code))
        except (OSError, AttributeError) as error:
            attempts.append(path + ': ' + type(error).__name__ + ': ' + str(error))
    raise OSError('NVML-compatible library unavailable; tried ' + str(len(attempts)) +
                  ' path(s): ' + '; '.join(attempts))


def open_library(library=None):
    """Load the first usable candidate; return (lib, path)."""
    lib, path, _attempts = load(library)
    return lib, path


def device_count(lib):
    count = C.c_uint()
    if lib.nvmlDeviceGetCount_v2(C.byref(count)):
        return 0
    return count.value


def device_uuids(lib):
    """Exact UUID strings in NVML enumeration order (may contain None)."""
    names = []
    for index in range(device_count(lib)):
        handle = C.c_void_p()
        uuid = None
        if lib.nvmlDeviceGetHandleByIndex_v2(index, C.byref(handle)) == 0:
            value = C.create_string_buffer(128)
            if lib.nvmlDeviceGetUUID(handle, value, len(value)) == 0:
                uuid = value.value.decode()
        names.append(uuid)
    return names


def strip_prefix(uuid):
    """UUID body without the vendor prefix (GPU-/PPU-/P800-)."""
    value = str(uuid or '')
    for prefix in ('GPU-', 'PPU-', 'P800-'):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def match_uuid(candidates, wanted):
    """Find the telemetry UUID whose body matches ``wanted``.

    ``candidates`` is the ordered list of UUID strings the telemetry provider
    records; ``wanted`` is the framework-reported UUID. Matching on the body
    keeps vendor prefix differences (torch may report a bare or differently
    prefixed value) from breaking the target-to-chip binding. Returns
    (index, uuid) or (None, None).
    """
    body = strip_prefix(wanted)
    if not body:
        return None, None
    for index, uuid in enumerate(candidates):
        if uuid and strip_prefix(uuid) == body:
            return index, uuid
    return None, None


def logical_identity(device_index, torch_uuid=None):
    """Identity dict for one device, read from the telemetry library.

    Returns ``{'entity': 'chip:<exact telemetry uuid>', 'uuid': ..., 'index': n,
    'library': path, 'source': 'nvml-compat', 'status': 'available'}`` or
    ``None`` when the library is unavailable or no UUID matches.
    """
    try:
        lib, path, _attempts = load()
    except OSError:
        return None
    uuids = device_uuids(lib)
    index, uuid = match_uuid(uuids, torch_uuid if torch_uuid else
                             (uuids[device_index] if device_index < len(uuids) else ''))
    if uuid is None:
        return None
    return dict(entity='chip:'+uuid, uuid=uuid, index=index, library=path,
                source='nvml-compat', status='available')
