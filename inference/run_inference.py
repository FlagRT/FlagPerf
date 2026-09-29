#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Container/current-environment entrypoint."""
from runtime.coordinator import main

if __name__ == '__main__':
    raise SystemExit(main())
