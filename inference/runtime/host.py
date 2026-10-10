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
from runtime.adapters import vendor_adapter, bind_scope
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
    adapter = vendor_adapter(cfg)
    cpu_export = cfg.get('_command') == 'export'
    device_arguments = [] if cpu_export else adapter.container_arguments(cfg, root)
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
    for key in [('inputs','path'),('policy','path')]:
        path = cfg[key[0]][key[1]]
        if path and not Path(path).is_relative_to(ROOT) and not Path(path).is_relative_to(root):
            argv += ['-v',f'{path}:{path}:ro']
    source = cfg.get('preview',{}).get('resume_from')
    if source:
        argv += ['-v',f'{source}:{source}:ro']
    reference = cfg.get('reference',{}).get('path') or cfg.get('reference',{}).get('performance_path')
    if reference:
        argv += ['-v',f'{reference}:{reference}:ro']
    from runtime.verification import SESSION_ENV
    if os.environ.get(SESSION_ENV):
        argv += ['-e', f'{SESSION_ENV}={os.environ[SESSION_ENV]}']
        argv += ['-e', f'FLAGPERF_TASK_WORKSPACE={root}/.work']
    vendor = cfg['runtime']['vendor']
    argv += device_arguments
    if os.environ.get('FLAGPERF_MONITOR_ROOT') == str(root):
        argv += ['-e', f'FLAGPERF_MONITOR_ROOT={root}']
    for k,v in cfg['vendors'][vendor]['env'].items(): argv += ['-e',f'{k}={v}']
    if cpu_export:
        argv += ['-e', 'TORCH_DEVICE_BACKEND_AUTOLOAD=0', '-e', 'CUDA_VISIBLE_DEVICES=',
                 '-e', 'ASCEND_RT_VISIBLE_DEVICES=']
    argv += ['--entrypoint','python3',image_id,'-u',str(ROOT/'run_inference.py')]
    return argv


def main():
    args = parser().parse_args()
    from runtime.verification import finish
    finish()
    root, name, monitor = None, None, None
    try:
        if args.command in ('analyze', 'repack-reference'):
            from runtime.offline import main as offline_main
            return offline_main(args)
        if args.command == 'report':
            from runtime.report_replay import run as run_report
            return run_report(args)
        if args.command in ('oracle', 'performance-baseline') and not args.dry_run:
            raise ValueError('A100 capture uses run_inference.py '+args.command+
                             ' in the prepared CUDA environment; run.py does not launch NVIDIA containers')
        cfg = load(args)
        if args.dry_run:
            from runtime.preflight import dry_run
            return dry_run(args, cfg)
        if args.command == 'list':
            print((ROOT/'support.json').read_text()); return 0
        if not isinstance(cfg['container'].get('image'), str) or not cfg['container']['image'].strip():
            raise ValueError('set container.image or pass --image with a compatible local Docker image')
        bind_scope(cfg, args.command)
        adapter = vendor_adapter(cfg)
        adapter.validate_launch_assets(cfg)
        root = output_directory(args,cfg)
        from runtime.preflight import check
        write_json(root/'cost-preflight.json', check(cfg, args.command, root))
        from runtime.verification import start
        start(root)
        seal_sources(root, cfg)
        for path in ['cache/home','cache/triton']: (root/path).mkdir(parents=True)
        image = json.loads(subprocess.check_output(['docker','image','inspect',cfg['container']['image']],text=True))
        adapter.validate_image(cfg, image[0]['Id'])
        write_json(root/'image.json',image)
        if cfg['runtime'].get('parallelism') == 'tp': prepare_account(root,image[0]['Id'])
        (root/'effective.yaml').write_text(yaml.safe_dump({k:v for k,v in cfg.items() if not k.startswith('_')},allow_unicode=True,sort_keys=False))
        name = 'flagperf-inference-'+uuid.uuid4().hex[:12]
        write_json(root/'host-launch.json',{'container':name,'image_id':image[0]['Id'],'command':args.command})
        snapshot = adapter.before_after_snapshot() if args.command != 'export' else None
        if snapshot:
            (root/(snapshot[0]+'-before.txt')).write_text(snapshot[1])
        if args.command == 'performance':
            from runtime.monitor import MonitorSession
            monitor = MonitorSession(cfg, root).start()
        argv = container_argv(cfg,root,name,image[0]['Id'])
        argv += [args.command,'--config',str(root/'effective.yaml'),'--output',str(root),'--internal-existing-root']
        if args.command in ('performance','performance-baseline'):
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
        finish()
        try:
            if name:
                # docker client termination alone does not stop its container.
                subprocess.run(['docker','stop','--time','10',name],stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            if monitor is not None:
                from runtime.monitor import finish_host
                finish_host(monitor)
        if name and args.command != 'export':
            if root is not None:
                try:
                    snapshot = adapter.before_after_snapshot()
                    if snapshot:
                        (root/(snapshot[0]+'-after.txt')).write_text(snapshot[1])
                except OSError:
                    pass
