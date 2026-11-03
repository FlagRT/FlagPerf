# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Linux pidfds for CPython builds whose libc headers omitted the wrappers.

Use the same kernel primitive, never a recyclable numeric-PID substitute.
Only Linux x86_64/aarch64 syscall ABIs are admitted. Kernel/seccomp rejection
is propagated with errno and does not fall back to kill(pid).
"""
import ctypes
import os
import platform
import signal
import sys


def supported():
    return sys.platform == 'linux' and platform.machine() in ('x86_64', 'aarch64')


def _syscall(number, *arguments):
    if not supported():
        raise RuntimeError('pidfd requires Linux x86_64 or aarch64')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *arguments)
    if result == -1:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def open_pidfd(pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError('positive integer PID required')
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid, 0)
    # __NR_pidfd_open is 434 on both admitted architectures.
    return _syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def send_signal(fd, signum):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd, signum)
    # __NR_pidfd_send_signal is 424 on both admitted architectures.
    _syscall(424, ctypes.c_int(fd), ctypes.c_int(signum), ctypes.c_void_p(), ctypes.c_uint(0))


def create_memfd(name):
    """Create the same anonymous, close-on-exec memory file on older builds."""
    if hasattr(os, 'memfd_create'):
        return os.memfd_create(name, getattr(os, 'MFD_CLOEXEC', 1))
    number = 319 if platform.machine() == 'x86_64' else 279
    return _syscall(number, ctypes.c_char_p(os.fsencode(name)), ctypes.c_uint(1))
