# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Single production configuration and construction point for Context writers."""
from .writer import WriterConfig
from .proxy import ProxyWriter


def online_writer_config():
    return WriterConfig(compress=False)


def online_writer(run, **options):
    return ProxyWriter(run, config=online_writer_config(), **options)
