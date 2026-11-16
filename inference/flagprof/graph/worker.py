# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Private subprocess entrypoint; preparation and exporter diagnostics go to a log."""
import contextlib
import json
from pathlib import Path
import sys
import time
import traceback


def main():
    config = json.loads(sys.argv[1])
    root = Path(config['out'])
    root.mkdir(parents=True, exist_ok=False)
    with (root/'export.log').open('w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        from .command import prepare
        from .core import export_into, finish
        from .data import SCHEMA
        try:
            import torch
            torch.set_num_threads(4)
            torch.manual_seed(531)
            start = time.monotonic()
            spec = prepare(config)
            elapsed = time.monotonic()-start
            result = export_into(spec, root, formats=config['formats'], bundle=config['bundle'],
                                 verify_onnx=config['verify_onnx'])
            result['timings_s']['prepare'] = elapsed
            from .data import write_json
            write_json(root/'manifest.json', result)
        except Exception as error:
            result = finish(root, dict(schema=SCHEMA, status='failed', bundle=config['bundle'], formats={},
                                       error=str(error), traceback=traceback.format_exc()))
    print(json.dumps(dict(output=str(root), status=result['status'],
                          formats={k:v['status'] for k,v in result['formats'].items()})))
    return 0 if result['status']=='completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
