# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Tiny CPU TP2 validation of the public embedding runner's sharded path."""
import os
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from flagprof import attach
    from flagprof.store import create_run,finalize
    from flagprof.runner import __main__ as runner
    root=Path(sys.argv[1]);rank=int(os.environ['RANK'])
    class CPUParallel:
        backend='gloo'
        identity={'backend':'gloo'}
        def __init__(self,device,backend):dist.init_process_group('gloo')
        def mesh(self):return init_device_mesh('cpu',(2,),mesh_dim_names=('tp',))
        def barrier(self):dist.barrier()
        def close(self):dist.destroy_process_group()
    run=create_run(root/('rank-'+str(rank)))
    argv=['runner','hf:'+str(root/'model'),'--device','cpu','--inputs-from',str(root/'oracle'),
          '--outputs',str(root/'outputs'),'--batch-size','1','--warmup','0','--forwards','1','--dtype','float32']
    with patch('flagprof.runner.parallel.Parallel',CPUParallel),patch.object(sys,'argv',argv):
        with attach(run,'ascend',preset='accuracy',config={'tensor_blob':'all'}) as handle:
            runner.main()
            if handle.probe_manager.errors:raise RuntimeError(str(handle.probe_manager.errors))
    finalize(run)


if __name__=='__main__':main()
