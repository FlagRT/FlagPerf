"""A100 accuracy and native-performance numerical profiles."""
import os
import subprocess
import time

def configure_oracle(backend, seed):
    import torch
    import random
    import numpy as np
    if 'A100' not in backend.get_device_name(0):
        raise ValueError('oracle generation requires an actual NVIDIA A100')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)


def settings():
    import torch
    import subprocess
    return {'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32':torch.backends.cudnn.allow_tf32,
            'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cuda_runtime':torch.version.cuda,
            'driver_and_gpu_uuids':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version,uuid',
                '--format=csv,noheader'],text=True,timeout=10).strip().splitlines()}


def configure_performance(backend):
    import torch
    if 'A100' not in backend.get_device_name(0):
        raise ValueError('A100 performance producer requires actual A100 hardware')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
        raise ValueError('native performance profile requires unset CUBLAS_WORKSPACE_CONFIG')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(False)


def environment_snapshot():
    """Read-only provenance; never change clocks, power, affinity or drivers."""
    commands={'gpu':['nvidia-smi','-q'], 'topology':['nvidia-smi','topo','-m'],
              'cpu':['lscpu'], 'dependencies':[__import__('sys').executable,'-m','pip','freeze']}
    values={}
    for name,argv in commands.items():
        try:
            p=subprocess.run(argv,text=True,capture_output=True,timeout=30)
            values[name]={'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
        except (OSError,subprocess.TimeoutExpired) as e:
            values[name]={'error':str(e)}
    values['time_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    values['affinity']=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
    return values
