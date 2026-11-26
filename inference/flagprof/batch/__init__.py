"""Batch runner for user-supplied inference workloads."""


def main(argv=None):
    from .batch_driver import main as run
    return run(argv)
