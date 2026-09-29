# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Single-host container launcher. No SSH, implicit installs, or device fallback."""
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import yaml
from runtime.common import ROOT, Progress, execute, write_json, seal_sources
from runtime.config import parser, load
from runtime.coordinator import output_directory


def prepare_account(root, image_id):
    """HCCL resolves the non-root UID through NSS, not only $USER."""
    raw = subprocess.check_output(['docker','run','--rm','--network','none',
        '--entrypoint','python3',image_id,'-c',
        'import json; from pathlib import Path; print(json.dumps({n:Path("/etc",n).read_text() for n in ["passwd","group"]}))'],text=True)
    files = json.loads(raw)
    uid,gid = os.getuid(),os.getgid()
    for name,number,entry in [
        ('passwd',uid,f'flagperf:x:{uid}:{gid}:FlagPerf:{root}/cache/home:/usr/sbin/nologin'),
        ('group',gid,f'flagperf:x:{gid}:')]:
        rows = [line for line in files[name].splitlines()
                if line.split(':')[0]!='flagperf' and line.split(':')[2]!=str(number)]
        (root/'cache'/name).write_text('\n'.join(rows+[entry])+'\n')


def container_argv(cfg, root, name, image_id):
    tp = cfg['runtime'].get('parallelism') == 'tp'
    argv = ['docker','run','--rm','--name',name,'--network','host' if tp else 'none','--shm-size',cfg['container']['shm_size'],
            '--user',f'{os.getuid()}:{os.getgid()}', '-v',f'{ROOT}:{ROOT}:ro', '-v',f'{root}:{root}:rw',
            '-v',f'{cfg["model"]["path"]}:{cfg["model"]["path"]}:ro',
            '-w',str(ROOT),'-e','HF_HUB_OFFLINE=1','-e','TOKENIZERS_PARALLELISM=false',
            '-e','FLAGPERF_CONTAINER_DEVICE=1','-e',f'FLAGPERF_IMAGE_ID={image_id}',
            '-e',f'HOME={root}/cache/home','-e','USER=flagperf','-e','LOGNAME=flagperf',
            '-e',f'TORCHINDUCTOR_CACHE_DIR={root}/cache/inductor','-e',f'TRITON_CACHE_DIR={root}/cache/triton']
    if tp:
        argv += ['-v',f'{root}/cache/passwd:/etc/passwd:ro',
                 '-v',f'{root}/cache/group:/etc/group:ro']
    if cfg['runtime']['vendor'] == 'ascend':
        folder = cfg['vendors']['ascend']['vendor_compiler']
        argv += ['-v',f'{folder}:{folder}:ro']
    for key in [('inputs','path'),('policy','path')]:
        path = cfg[key[0]][key[1]]
        if path and not Path(path).is_relative_to(ROOT) and not Path(path).is_relative_to(root):
            argv += ['-v',f'{path}:{path}:ro']
    source = cfg.get('preview',{}).get('resume_from')
    if source:
        argv += ['-v',f'{source}:{source}:ro']
    vendor = cfg['runtime']['vendor']
    physical = cfg['runtime']['device']
    if vendor == 'ascend':
        devices = cfg['runtime']['devices'] if tp else [physical]
        # The runtime enumerates mounted devices in physical order.
        for dev in [*[f'davinci{d}' for d in sorted(devices)],'davinci_manager','devmm_svm','hisi_hdc']:
            if not Path('/dev',dev).exists(): raise ValueError(f'missing Ascend device /dev/{dev}')
            argv += ['--device',f'/dev/{dev}:/dev/{dev}:rwm']
        for path in ['/usr/local/Ascend/driver','/usr/local/Ascend/firmware','/usr/local/dcmi','/etc/ascend_install.info','/usr/local/bin/npu-smi']:
            if Path(path).exists(): argv += ['-v',f'{path}:{path}:ro']
        # Only one physical device is mounted; it is logical device zero here.
        visible = ','.join(str(sorted(devices).index(d)) for d in devices)
        argv += ['-e',f'ASCEND_RT_VISIBLE_DEVICES={visible}']
    else:
        argv += ['--gpus',f'device={physical}','-e','CUDA_VISIBLE_DEVICES=0']
    for k,v in cfg['vendors'][vendor]['env'].items(): argv += ['-e',f'{k}={v}']
    argv += ['--entrypoint','python3',image_id,'-u',str(ROOT/'run_inference.py')]
    return argv


def main():
    args = parser().parse_args()
    root, name = None, None
    try:
        if args.command == 'report':
            from reporting.replay import run as run_report
            return run_report(args)
        cfg = load(args)
        if args.command == 'list':
            print((ROOT/'support.json').read_text()); return 0
        if not isinstance(cfg['container']['image'], str) or not cfg['container']['image'].strip():
            raise ValueError('set container.image in your configuration or pass --image YOUR_LOCAL_IMAGE')
        if cfg['runtime']['vendor'] == 'ascend':
            from vendors.stack import validate_bundle
            validate_bundle(cfg['vendors']['ascend']['vendor_compiler'])
        root = output_directory(args,cfg)
        seal_sources(root)
        for path in ['cache/home','cache/triton']: (root/path).mkdir(parents=True)
        image = json.loads(subprocess.check_output(['docker','image','inspect',cfg['container']['image']],text=True))
        write_json(root/'image.json',image)
        if cfg['runtime'].get('parallelism') == 'tp': prepare_account(root,image[0]['Id'])
        (root/'effective.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
        name = 'flagperf-inference-'+uuid.uuid4().hex[:12]
        write_json(root/'host-launch.json',{'container':name,'image_id':image[0]['Id'],'command':args.command})
        if cfg['runtime']['vendor'] == 'ascend':
            snapshot = subprocess.run(['npu-smi','info'],capture_output=True,text=True)
            (root/'npu-before.txt').write_text(snapshot.stdout+snapshot.stderr)
        argv = container_argv(cfg,root,name,image[0]['Id'])
        argv += [args.command,'--config',str(root/'effective.yaml'),'--output',str(root),'--internal-existing-root']
        if args.command == 'performance':
            paths = 2 if any(cfg['performance'].get(k)=='both' for k in ['flaggems','flagtree','flagcx']) else 1
            audits = 2*paths
            phases = 2 if cfg['runtime'].get('parallelism') == 'tp' else 1
            if cfg['performance']['level'] == 'layer':
                # Baseline, timing, trace and memory; memory repeats disjoint groups.
                names = cfg['performance']['layers']
                phases = 4 + (0 if names == ['all'] else len(names))
            timeout = (paths+audits+phases*paths*cfg['performance']['repeats'])*cfg['runtime']['timeout_seconds']+120
        else:
            timeout = (cfg['preview']['budget_seconds'] if args.command=='preview' else 2*cfg['runtime']['timeout_seconds']) + 4*cfg['runtime']['timeout_seconds'] + 120
        with Progress() as progress:
            progress.phase = f'container/{args.command}; log={root}/launcher/run.log'
            state = execute(argv,root/'launcher',timeout)
        if not (root/'result.json').exists():
            write_json(root/'result.json',{'status':'failed','stage':'container','error':'container exited without final result','process':state})
        print(f'Results: {root}',flush=True)
        return state['exit_code']
    except (ValueError,OSError,subprocess.CalledProcessError,KeyboardInterrupt) as error:
        if root is not None:
            write_json(root/'result.json',{'status':'failed','stage':'host','error':str(error)})
        print(f'Host error: {error}',file=sys.stderr)
        return 2
    finally:
        if name:
            # docker client termination alone does not stop its container.
            subprocess.run(['docker','stop','--time','10',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            if root is not None and cfg['runtime']['vendor'] == 'ascend':
                try:
                    snapshot = subprocess.run(['npu-smi','info'],capture_output=True,text=True)
                    (root/'npu-after.txt').write_text(snapshot.stdout+snapshot.stderr)
                except OSError:
                    pass
