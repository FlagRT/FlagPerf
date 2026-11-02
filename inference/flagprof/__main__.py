# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Run the supported CLI without importing optional SDKs."""
from .cli import main
raise SystemExit(main())
