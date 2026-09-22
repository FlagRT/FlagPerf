from pathlib import Path
import sys

import torch.distributed as distributed

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from case_assets import load_case_config
from drivers import kunlunxin
from drivers.int8 import run_verified_int8


def main():
    kunlunxin.initialize()
    config = load_case_config(Path.cwd(), 'kunlunxin/P800')
    distributed.init_process_group(backend=config['DIST_BACKEND'])
    try:
        value = run_verified_int8(kunlunxin, config, distributed.get_rank(), distributed.get_world_size(), 0)
        print("[FlagPerf Result]Rank 0's computation-INT8=" + str(value) + 'TOPS', flush=True)
    finally:
        distributed.destroy_process_group()


if __name__ == '__main__':
    main()
