# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Small verb-based CLI. Inspect consumes stored facts only."""
from __future__ import annotations
import argparse
import json
import shlex
import sys


def _print_result(*values, **options):
    from .progress import finish_display
    finish_display()
    print(*values, **options)


def main(argv=None):
    from .offline.limits import DEFAULT_RECORD_SOURCE_BYTES
    from .vendors import monitored_vendors, vendor_names
    vendors = vendor_names()
    parser=argparse.ArgumentParser(prog='flagprof')
    verbs=parser.add_subparsers(dest='verb',required=True)
    init = verbs.add_parser('init', help='生成单卡、双卡或对照工作流配置')
    init.add_argument('--list', action='store_true', help='列出已交付的模板')
    init.add_argument('--template', default='quickstart')
    init.add_argument('-o', '--out')
    init.add_argument('--force', action='store_true')
    workflow = verbs.add_parser('run', help='检查或执行推理工作流')
    workflow.add_argument('config')
    workflow.add_argument('--set', dest='overrides', metavar='KEY=VALUE', action='append', default=[])
    workflow.add_argument('--out', metavar='JOB_DIR')
    preview = workflow.add_mutually_exclusive_group()
    preview.add_argument('--check', action='store_true')
    preview.add_argument('--dry-run', action='store_true')
    workflow.add_argument('--json', action='store_true')
    from .graph.command import add_parser as add_export_parser
    add_export_parser(verbs)
    telemetry=verbs.add_parser('telemetry',help='inspect the device telemetry stack')
    tele_actions=telemetry.add_subparsers(dest='telemetry_action',required=True)
    tele_check=tele_actions.add_parser('check',help='resolve the provider and print its inventory')
    tele_check.add_argument('--vendor',required=True,choices=('ascend','nvidia','thead','kunlunxin'))
    tele_check.add_argument('--devices',default='all')
    tele_check.add_argument('--json',action='store_true')
    watch=verbs.add_parser('watch',help='independent node monitoring')
    actions=watch.add_subparsers(dest='watch_action',required=True)
    start=actions.add_parser('start'); start.add_argument('--root',required=True)
    start.add_argument('--vendor',required=True,choices=monitored_vendors())
    start.add_argument('--devices',default='all'); start.add_argument('--config')
    status=actions.add_parser('status'); status.add_argument('--root',required=True)
    replay_watch=actions.add_parser('replay'); replay_watch.add_argument('session'); replay_watch.add_argument('--out',required=True)
    release=actions.add_parser('release'); release.add_argument('--root',required=True)
    release.add_argument('--reference',required=True); release.add_argument('--delivery',required=True)
    delivery=verbs.add_parser('delivery',help='publish portable evidence archives and restore evidence trees')
    delivery.add_argument('action',choices=['publish','verify','restore'],help='publish source trees to a new delivery directory, verify a delivery, or restore it to a new output directory')
    delivery.add_argument('paths',nargs='+',help='publish: one or more source trees; verify/restore: exactly one delivery directory')
    delivery.add_argument('--out',help='publish: new delivery directory on any writable filesystem; restore: new output directory (required for both)')
    delivery.add_argument('--max-bytes',type=int,help='byte budget (default 200 GiB for publish, 40 GiB for restore)')
    inspect=verbs.add_parser('inspect',help='read stored identities and evidence states')
    inspect.add_argument('run',help='Store run directory');inspect.add_argument('--verify',action='store_true',help='verify every derived and raw artifact')
    inspect.add_argument('--json',action='store_true',help='print the full inspection as JSON')
    record_parser=verbs.add_parser('record',help='capture one complete official Python session')
    record_parser.add_argument('--store',required=True,help='output root; final mode keeps the Store under <store>/runs')
    record_parser.add_argument('--vendor',required=True,help='vendor adapter, e.g. ascend or nvidia')
    record_parser.add_argument('--target-device',help='logical device used by a monitored target, e.g. npu:0 or cuda:0')
    record_parser.add_argument('--preset',choices=['perf','modules','perf-lite','coverage','accuracy','monitor'],default='perf',help='collection preset (default perf); perf/modules use the official profiler in final mode')
    record_parser.add_argument('--flaggems', choices=['off','on'],help='FlagGems component (preset default off)')
    record_parser.add_argument('--flaggems-include',help='JSON array of FlagGems function names to register')
    record_parser.add_argument('--flagtree',choices=['off','on'],help='compiler selection; omitted preserves the current environment')
    record_parser.add_argument('--vendor-compiler',help='sealed vendor Triton bundle, if the provider is not installed')
    record_parser.add_argument('--dispatch', choices=['off','summary','each-window'],help='per-call dispatch records; each-window links device events to project signatures and layers (preset default)')
    record_parser.add_argument('--layers', choices=['all','selected'],help='module layers to observe (preset default; accuracy requires all)')
    record_parser.add_argument('--probes', choices=['default','none'],help='project probes (preset default)')
    record_parser.add_argument('--profile-memory', action='store_true', help='collect official allocator memory records')
    record_parser.add_argument('--capture-only',action='store_true',help='seal raw capture for later analysis and import')
    record_parser.add_argument('--tensor-blob',default=None,help='none (default), all or comma-separated call_key prefixes')
    record_parser.add_argument('--tensor-blob-bytes',type=int,help='accuracy: total tensor blob budget in bytes (default 1 GiB)')
    record_parser.add_argument('--tensor-ops-in',help='accuracy: record per-operator tensors inside this module path; requires batch size 1')
    record_parser.add_argument('--tensor-boundaries',metavar='model|layer|PATH',help='accuracy: model keeps runner outputs only; layer uses default layer boundaries (default); PATH is a JSON array of module paths; --tensor-ops-in adds operator capture independently')
    record_parser.add_argument('--capture-mode',choices=['windowed','final'],default='final',help='final (default): one official session for the whole target; windowed: step windows')
    record_parser.add_argument('--source-max-bytes',type=int,default=DEFAULT_RECORD_SOURCE_BYTES,help='final capture source limit in bytes (default 4 GiB)')
    record_parser.add_argument('--timeout',type=float,default=1800,help='whole record deadline in seconds')
    record_parser.add_argument('--active',type=int,default=None,help='windowed: active steps per window (default 5)')
    record_parser.add_argument('--windows',type=int,default=None,help='windowed: number of windows (default 2)')
    record_parser.add_argument('--invalid-policy',choices=['stop','continue'],default=None,help='windowed: action after an invalid window (default stop)')
    record_parser.add_argument('--log-file',action='append',default=[],help='explicit log source; may be repeated')
    record_parser.add_argument('--plugins',help='component descriptors JSON (not project binding rules)')
    record_parser.add_argument('command',nargs=argparse.REMAINDER,help='target Python command after --')
    launch=verbs.add_parser('launch',help='explicit two-rank workers with separate Stores')
    launch.add_argument('--nproc-per-node',type=int,default=2,help='number of ranks; only 2 is accepted')
    launch.add_argument('--vendor',choices=vendors,help='select the hardware adapter when detection is ambiguous')
    launch.add_argument('--comm-backend',choices=['hccl','flagcx','nccl'],help='collective backend')
    launch.add_argument('--flaggems',choices=['off','on'],default='off')
    launch.add_argument('--flaggems-include',help='JSON array of FlagGems function names')
    launch.add_argument('--flagtree',choices=['off','on'],help='compiler selection; omitted preserves the current environment')
    launch.add_argument('--vendor-compiler',help='sealed vendor compiler bundle; overrides FLAGPROF_VENDOR_COMPILER')
    launch.add_argument('--capture-only',action='store_true',help='seal per-rank profiler captures for later analysis and import')
    launch.add_argument('--store',required=True,help='group root: one rank-N Store per rank plus group.json for report')
    launch.add_argument('--preset',choices=['none','accuracy','perf','modules','perf-lite','coverage','monitor'],default='perf',help='per-rank preset (default perf); none runs the target without collection')
    launch.add_argument('--timeout',type=float,default=1800,help='whole launch deadline in seconds')
    launch.add_argument('--source-max-bytes',type=int,default=1<<30,help='sealed source limit per rank in bytes (default 1 GiB)')
    launch.add_argument('--tensor-blob',choices=['all','none'],default='all',help='accuracy: save full tensor blobs (default all)')
    launch.add_argument('--tensor-blob-bytes',type=int,default=1<<30,help='accuracy: tensor blob budget in bytes (default 1 GiB)')
    launch.add_argument('--tensor-boundaries',metavar='model|layer|PATH',help='same accuracy boundaries as record')
    launch.add_argument('--tensor-ops-in',help='accuracy: capture operators within this module; batch 1 required')
    launch.add_argument('--profile-memory',action='store_true',help='collect official allocator memory records; requires perf or modules')
    launch.add_argument('command',nargs=argparse.REMAINDER,help='target Python command after --')
    for target_parser in (record_parser,launch):
        target_parser.add_argument('--monitor-root',help='independent watch root; required for agent topology')
    offline=verbs.add_parser('import',help='import a sealed capture into a new run using bounded Plan B')
    offline.add_argument('capture',help='sealed capture directory');offline.add_argument('--store',required=True,help='runs root that receives the new run')
    offline.add_argument('--run-id',help='name of the new run (default UTC time, host and random suffix)')
    offline.add_argument('--calls-from',help='explicit collection run with matching source_run_id')
    page=verbs.add_parser('page',help='verified bounded Plan B device page')
    page.add_argument('run',help='Plan B run directory')
    page.add_argument('--offset',type=int,default=0,help='first row (default 0)');page.add_argument('--limit',type=int,default=100,help='rows per page (default 100)')
    seal=verbs.add_parser('seal',help='seal an externally exported trace; caller asserts normal close')
    seal.add_argument('source',help='exported trace file');seal.add_argument('--out',required=True,help='new sealed capture directory');seal.add_argument('--vendor',choices=vendors,required=True,help='vendor format of the trace')
    seal.add_argument('--normal-close',action='store_true',required=True,help='required assertion that the profiler session closed normally')
    seal.add_argument('--source-max-bytes',type=int,default=1<<30,help='copy limit for the trace in bytes (default 1 GiB)')
    seal.add_argument('--identity-from',help='same-capture attachment state or final capture directory')
    analyse=verbs.add_parser('analyse',help='retry official analysis of already sealed final raw capture')
    analyse.add_argument('capture',help='sealed final raw capture directory')
    report=verbs.add_parser('report',help='generate bounded offline analysis and a self-contained HTML report')
    report.add_argument('run',help='one of: a Store run directory (single-run report); a group manifest file such as <launch store>/group.json (one report per rank); or oracle, baseline, compare, render');report.add_argument('--out',help='new output directory outside the inputs (default <run>-report, <group dir>-report or <ref>-oracle beside the input)')
    report.add_argument('--last',default='1h',help='monitor session window (default 1h)')
    report.add_argument('--reports',nargs='+',help='compare: generated report directories; keep incompatible values with not_comparable reasons')
    report.add_argument('--ref',help='oracle: reference Store run or TP2 group.json');report.add_argument('--cmp',help='oracle: compared Store run or TP2 group.json');report.add_argument('--declare',help='oracle/compare: JSON differences list of explicitly varied identity fields')
    report.add_argument('--perf-ref-run',help='oracle: performance run linked to the reference side by exact signature');report.add_argument('--perf-cmp-run',help='oracle: performance run linked to the compared side by exact signature')
    report.add_argument('--replay-ref-run',help='oracle: run created by replay run --side ref; requires --replay-cmp-run');report.add_argument('--replay-cmp-run',help='oracle: run created by replay run --side cmp; requires --replay-ref-run')
    report.add_argument('--op-kinds',help='operator categories generated in the run torch environment')
    report.add_argument('--limits',help='JSON file of analysis resource limits')
    report.add_argument('--source',help='render: existing report directory; no Store or device access')
    report.add_argument('--refresh-view',action='store_true',help='render: rebuild conclusions from saved machine results and exports')
    report.add_argument('--runner-results',nargs='+',help='saved runner.json files for the same workload condition')
    report.add_argument('--baseline',help='oracle: optional third accuracy Store or group.json')
    replay=verbs.add_parser('replay',help='prepare portable inputs or execute a pure operator')
    replay_verbs=replay.add_subparsers(dest='replay_verb',required=True)
    prepare=replay_verbs.add_parser('prepare',help='build a portable input package from two accuracy runs')
    for name,text in (('ref','reference accuracy run with full tensor blobs'),('cmp','compared accuracy run with full tensor blobs'),('call-key','call_key of the target operator'),('out','new input package directory')):prepare.add_argument('--'+name,required=True,help=text)
    execute=replay_verbs.add_parser('run',help='execute one side of an input package into a new replay Store run')
    execute.add_argument('--vendor-compiler',help='local sealed compiler bundle for the recorded provider')
    execute.add_argument('package',help='input package from replay prepare');execute.add_argument('--side',choices=['ref','cmp'],required=True,help='side whose registered conditions are used')
    execute.add_argument('--device',required=True,help='torch device of the same family as the side, e.g. cuda:0 or npu:0');execute.add_argument('--store',required=True,help='Store root that receives the replay run')
    for entry in (record_parser, launch):
        entry.add_argument('--diagnose', choices=['on','off'], help='business fault diagnosis; default off')
        entry.add_argument('--monitor', choices=['on','off'], help='node resource monitor; default off')
        entry.add_argument('--monitor-topology', choices=['host','local','agent'], help='default: detected provider topology')
        entry.add_argument('--monitor-interval-ms', type=int, help='fast sampling interval, 100..5000 ms; default 1000')
    for entry in (workflow, record_parser, launch, analyse, offline, inspect, report):
        entry.add_argument('--progress', choices=('auto', 'plain', 'off'), default='auto',
                           help='stderr progress: auto detects terminals, plain emits lines, off disables it')
    argv = sys.argv[1:] if argv is None else list(argv)
    args=parser.parse_args(argv)
    if args.verb == 'record' and args.capture_mode == 'final':
        try:
            _check_window_options(args, argv, vendors)
        except ValueError as error:
            parser.error(str(error))
    if args.verb == 'report':
        if args.baseline and args.run!='oracle':parser.error('--baseline is only valid for report oracle')
        if args.source and args.run!='render':parser.error('--source is only valid for report render')
        if args.refresh_view and args.run!='render':parser.error('--refresh-view is only valid for report render')
        if args.runner_results and args.run in ('oracle','compare'):parser.error('--runner-results belongs to a task, baseline or render report')
    if not hasattr(args, 'progress') or getattr(args, 'dry_run', False):
        return _dispatch(args, parser)
    from .progress import phase, session
    mode = args.progress
    # A windowed target inherits stdout and stderr. Use complete lines so its
    # own terminal output cannot begin in the middle of an animated meter.
    if args.verb == 'record' and mode == 'auto' and (
            args.capture_mode == 'windowed' or args.preset in ('perf-lite', 'coverage', 'accuracy', 'monitor')):
        mode = 'plain'
    with session(mode, json_output=getattr(args, 'json', False)) as progress:
        if args.verb == 'run':
            code = _dispatch(args, parser)
        else:
            with phase(args.verb) as operation:
                code = _dispatch(args, parser)
                operation.finish(code)
        progress.finish(code)
        return code


def _dispatch(args, parser):
    from pathlib import Path
    if args.verb in ('init', 'run'):
        from .workflow import dispatch
        return dispatch(args)
    if args.verb=='export':
        try:
            from .graph.command import run
            return run(args)
        except (OSError, ValueError) as error:
            print('flagprof export: '+str(error),file=sys.stderr)
            return 2
    if args.verb=='telemetry':
        from .telemetry.check import run as telemetry_check
        return telemetry_check(args.vendor,args.devices,args.json)
    if args.verb=='watch':
        try:
            from .monitor.session import configuration,load
            if args.watch_action=='start':
                from .monitor.node import start
                return start(args.root,args.vendor,args.devices,configuration(args.config))
            if args.watch_action=='status':
                from .monitor.session import snapshot
                print(json.dumps(snapshot(args.root),ensure_ascii=False)); return 0
            from .monitor.commands import replay, release
            result = replay(args.session,args.out) if args.watch_action=='replay' else release(args.root,args.reference,args.delivery)
            print(json.dumps(result,ensure_ascii=False)); return 0
        except (OSError,ValueError) as error:
            print('flagprof watch: '+str(error),file=sys.stderr); return 2
    if args.verb=='launch':
        try:
            from .launch import launch
            command=args.command[1:] if args.command[:1]==['--'] else args.command
            config={name:getattr(args,name) for name in ('flaggems','flaggems_include','flagtree','vendor_compiler','monitor','monitor_topology','monitor_root','monitor_interval_ms','diagnose') if getattr(args,name) is not None}
            if args.preset=='accuracy':config.update(tensor_blob=args.tensor_blob,tensor_blob_bytes=args.tensor_blob_bytes)
            if args.tensor_boundaries is not None:
                value=args.tensor_boundaries
                if value not in ('model','layer'):
                    value=json.loads(Path(value).read_text())
                    if not isinstance(value,list):raise ValueError('tensor boundaries file must contain a JSON array of module paths')
                config['tensor_boundaries']=value
            if args.tensor_ops_in is not None:config['tensor_ops_in']=args.tensor_ops_in
            if args.profile_memory:
                if args.preset not in ('perf','modules'):raise ValueError('official memory requires perf/modules')
                config['profile_memory']=True
            return launch(command,store=args.store,nproc_per_node=args.nproc_per_node,comm_backend=args.comm_backend,
                          vendor=args.vendor,
                          preset=args.preset,config=config,timeout=args.timeout,source_max_bytes=args.source_max_bytes,
                          capture_only=args.capture_only)
        except (ValueError,OSError,KeyError,RuntimeError) as error:
            _print_result('flagprof launch: '+str(error),file=sys.stderr);return 2
    if args.verb=='delivery':
        try:
            from .delivery import publish_tree,verify,restore
            options={} if args.max_bytes is None else dict(max_bytes=args.max_bytes)
            if args.action=='publish':
                if not args.out:raise ValueError('publish requires --out')
                if len(args.paths)==1 and ((Path(args.paths[0])/'monitor-ref.json').exists() or
                        Path(args.paths[0]).name=='monitor-ref.json' or
                        (Path(args.paths[0]).parent.name=='references' and Path(args.paths[0]).suffix=='.json')):
                    from .monitor.commands import publish
                    result=publish(args.paths[0],args.out,**options)
                else:
                    result=publish_tree(args.paths,args.out,**options)
            else:
                if len(args.paths)!=1:raise ValueError('one delivery directory required')
                if args.action=='verify':result=verify(args.paths[0])
                else:
                    if not args.out:raise ValueError('restore requires --out')
                    result=restore(args.paths[0],args.out,**options)
            print(json.dumps(result));return 0
        except (ValueError,OSError,KeyError,TypeError) as error:
            print('flagprof delivery: '+str(error),file=sys.stderr);return 2
    if args.verb=='replay':
        try:
            from . import replay
            result=(replay.prepare(args.ref,args.cmp,args.call_key,args.out)
                if args.replay_verb=='prepare' else replay.run(args.package,side=args.side,device=args.device,store=args.store,vendor_compiler=args.vendor_compiler))
            print(json.dumps(result,ensure_ascii=False));return 0 if result.get('status','complete')=='complete' else 2
        except (ValueError,OSError,KeyError,TypeError,RuntimeError) as error:
            print('flagprof replay: '+str(error),file=sys.stderr);return 2
    if args.verb=='report':
        from .analyzers.workspace import Limits
        if args.run=='render':
            try:
                if not args.source or not args.out:raise ValueError('render requires --source and --out')
                from .report.product import redraw
                from .analyzers.workspace import Limits
                result=redraw(args.source,args.out,runner_results=args.runner_results,refresh_view=args.refresh_view,
                    limits=Limits(**json.loads(Path(args.limits).read_text())) if args.limits else None)
                _print_result(json.dumps(result,ensure_ascii=False));return 0
            except (ValueError,OSError,KeyError,TypeError) as error:
                _print_result('flagprof report render: '+str(error),file=sys.stderr);return 2
        if args.run in ('compare','baseline'):
            try:
                if args.run=='baseline':
                    if not args.runner_results or not args.out:raise ValueError('report baseline requires --runner-results and --out')
                    from .report.performance import generate as baseline_report
                    baseline_report(args.runner_results,args.out,limits=Limits(**json.loads(Path(args.limits).read_text())) if args.limits else None)
                else:
                    if not args.reports or len(args.reports)<2 or not args.out:raise ValueError('report compare requires --reports (at least two) and --out')
                    from .report import compare_device_reports
                    compare_device_reports(args.reports,args.out,declare=json.loads(Path(args.declare).read_text()) if args.declare else None,limits=Limits(**json.loads(Path(args.limits).read_text())) if args.limits else None)
                _print_result(json.dumps(dict(output=args.out,status='complete')));return 0
            except (ValueError,OSError,KeyError,TypeError) as error:
                _print_result('flagprof report: '+str(error),file=sys.stderr);return 2
        try:
            from .report import generate, generate_oracle
            if args.run=='oracle':
                if not args.ref or not args.cmp:raise ValueError('oracle requires --ref and --cmp')
                result=generate_oracle(args.ref,args.cmp,args.out,baseline=args.baseline,
                    perf_ref_run=args.perf_ref_run,perf_cmp_run=args.perf_cmp_run,
                    replay_ref_run=args.replay_ref_run,replay_cmp_run=args.replay_cmp_run,
                    declare=json.loads(Path(args.declare).read_text()) if args.declare else None,
                    limits=json.loads(Path(args.limits).read_text()) if args.limits else None)
                _print_result(json.dumps(result,ensure_ascii=False));return 0
            if (Path(args.run)/'session.json').exists() and not (Path(args.run)/'manifest.json').exists():
                if args.runner_results:raise ValueError('runner results require a task report or report baseline')
                from .monitor.report import generate as monitor_report
                _print_result(json.dumps(monitor_report(args.run,args.out,last=args.last),ensure_ascii=False));return 0
            if Path(args.run).is_file():
                from .report.group import generate_group
                result=generate_group(args.run,args.out,runner_results=args.runner_results,
                    limits=Limits(**json.loads(Path(args.limits).read_text())) if args.limits else None)
                _print_result(json.dumps(result));return 0
            result=generate(args.run,args.out,runner_results=args.runner_results,
                            op_kinds=json.loads(Path(args.op_kinds).read_text()) if args.op_kinds else None,
                            limits=json.loads(Path(args.limits).read_text()) if args.limits else None)
            _print_result(json.dumps(result,ensure_ascii=False));return 0
        except (ValueError,OSError,KeyError,TypeError) as error:
            _print_result('flagprof report: '+str(error),file=sys.stderr);return 2
    if args.verb in ('import','page','seal','analyse') or (args.verb=='inspect' and (Path(args.run)/'plan-b-capture.json').exists()):
        try:
            from pathlib import Path
            if args.verb=='import':
                from .offline.service import import_bundle
                result=import_bundle(args.capture, store=args.store, run_id=args.run_id, calls_from_run=args.calls_from)
            elif args.verb in ('page','inspect'):
                from .offline.service import inspect_run
                result=inspect_run(args.run,**(dict(offset=args.offset,limit=args.limit) if args.verb=='page' else {}))
            elif args.verb=='seal':
                from .offline.bundle import make_bundle,copy_checked
                from .offline.workspace import attempt
                source=Path(args.source).resolve()
                destination=Path(args.out).resolve()
                destination.parent.mkdir(parents=True,exist_ok=True)
                with attempt(destination.parent) as (operation, _lease):
                    temp = operation/'raw'
                    temp.mkdir()
                    # Streaming copy of the selected input.
                    copy_checked(source,temp/'trace.json',max_bytes=args.source_max_bytes)
                    metadata=dict(source_max_bytes=args.source_max_bytes,origin='external-export',normal_close=True,target_returncode=0,normal_close_evidence='caller assertion')
                    if args.identity_from:
                        from .offline.project import identity_from
                        metadata.update(identity_from(source,args.identity_from))
                    else:
                        metadata['identities']={k:dict(status='unavailable',reason='external_export') for k in ('collector','hardware','exporter')}
                    result=make_bundle(temp,Path(temp)/'trace.json',destination,vendor=args.vendor,metadata=metadata)
            else:
                from .offline.capture import finish_capture
                result=finish_capture(args.capture)
            if args.verb == 'inspect':
                return _print_inspection(result, args.json)
            _print_result(json.dumps(result,ensure_ascii=False));return 0
        except (ValueError,OSError,KeyError,TypeError,RuntimeError) as error:
            _print_result('flagprof '+args.verb+': '+str(error),file=sys.stderr);return 2
    if args.verb=='record':
        from .record import record
        from pathlib import Path
        command=args.command[1:] if args.command[:1]==['--'] else args.command
        try:
            if args.capture_only and (args.capture_mode!='final' or args.preset not in ('perf','modules')):
                raise ValueError('--capture-only requires official final mode')
            config = {name:getattr(args,name) for name in ('flaggems','flaggems_include','flagtree','vendor_compiler','layers','probes','monitor','monitor_topology','monitor_root','monitor_interval_ms','diagnose','target_device') if getattr(args,name) is not None}
            if args.profile_memory:
                if args.preset not in ('perf','modules'):
                    raise ValueError('--profile-memory requires an official profiler preset (perf or modules)')
                config['profile_memory'] = True
            for name in ('tensor_blob','tensor_blob_bytes','tensor_ops_in','tensor_boundaries'):
                value=getattr(args,name)
                if value is not None:
                    if name=='tensor_blob' and value not in ('all','none'):value=value.split(',')
                    if name=='tensor_boundaries' and value not in ('model','layer'):
                        value=json.loads(Path(value).read_text())
                        if not isinstance(value,list):raise ValueError('tensor boundaries file must contain a JSON array of module paths')
                    config[name]=value
            if args.dispatch is not None:
                config['dispatch'] = args.dispatch
            if args.capture_mode=='final':
                if args.preset in ('coverage','perf-lite','accuracy','monitor'):
                    return record(command,store=args.store,vendor=args.vendor,preset=args.preset,config=config,timeout=args.timeout)
                from .offline.capture import record_final
                return record_final(command,store=args.store,vendor=args.vendor,preset=args.preset,
                                    timeout=args.timeout, source_max_bytes=args.source_max_bytes,
                                    **(dict(capture_only=args.capture_only) if args.capture_only else {}),
                                    **(dict(config=config) if config else {}))
            config.update(active=5 if args.active is None else args.active,windows=2 if args.windows is None else args.windows,invalid_policy='stop' if args.invalid_policy is None else args.invalid_policy,
                          log_files=tuple(str(Path(p).resolve()) for p in args.log_file))
            return record(command,store=args.store,vendor=args.vendor,preset=args.preset,
                capture_mode='windowed',
                timeout=args.timeout,
                config=config,
                descriptors=json.loads(Path(args.plugins).read_bytes()) if args.plugins else None)
        except (ValueError,OSError) as error:
            from .progress import finish_display
            finish_display()
            parser.error(str(error))
    from .offline.worker import on_disk
    try:
        result = on_disk(None, _inspect_store, args.run, args.verify)
    except (ValueError,OSError,KeyError,TypeError) as error:
        _print_result('flagprof inspect: '+str(error),file=sys.stderr)
        return 2
    return _print_inspection(result, args.json)


def _check_window_options(args, argv, vendors):
    from pathlib import Path
    window = [flag for flag in ('--active', '--windows', '--invalid-policy', '--plugins', '--log-file')
              if getattr(args, flag[2:].replace('-', '_')) not in (None, [])]
    if not window:
        return
    keep_final = args.capture_only and args.preset in ('perf', 'modules') and args.vendor in vendors
    # argparse already resolved abbreviations and retained the target verbatim in REMAINDER.
    # Only edit record options. Values supplied with '=' may themselves look like options.
    remove = ['--capture-mode']
    if keep_final:
        remove.extend(window)
    if args.capture_only and not keep_final:
        remove.append('--capture-only')
    controls = iter(argv[1:len(argv) - len(args.command)])
    source_cli = Path(__file__).resolve().parents[1] / 'run.py'
    entry = [str(source_cli)] if Path(sys.argv[0]).resolve() == source_cli else ['-m', 'flagprof']
    corrected = [sys.executable, *(['-S'] if sys.flags.no_site else []), *entry, 'record']
    for token in controls:
        option, equal, _ = token.partition('=')
        match = next((name for name in remove if option.startswith('--') and name.startswith(option)), None)
        if match:
            if not equal and match != '--capture-only':
                next(controls)
        else:
            corrected.append(token)
    corrected.extend(['--capture-mode', 'final' if keep_final else 'windowed'])
    corrected.extend(args.command)
    reason = 'Run with --capture-mode windowed'
    if keep_final:
        reason = 'Keep --capture-only in final mode and remove the window options'
    elif args.capture_only:
        reason = 'Run with --capture-mode windowed and remove --capture-only (requires an official perf/modules final capture)'
    raise ValueError(f"final mode does not accept {', '.join(window)}. {reason}:\n{shlex.join(corrected)}")


def _inspect_store(run, verify):
    from .store import StoreView
    with StoreView(run) as view:
        return dict(schema='inspect/2', run=view.run_status(), traces=view.inspect(verify=verify),
                    tensor=view.tensor_status(verify=verify))


def _print_inspection(result, as_json):
    from .progress import finish_display
    finish_display()
    run, rows = result['run'], result['traces']
    if as_json:print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2))
    else:
        print('run:',run['run_id'],' state:',run['state'],' manifest:',run['manifest_status'])
        print('  exits:',json.dumps(run['exits'],ensure_ascii=False))
        print('  writers:',json.dumps(run['writers'],ensure_ascii=False))
        print('  log checks:',json.dumps(run['log_check'],ensure_ascii=False))
        print('  captures:',json.dumps(run['captures'],ensure_ascii=False))
        if result.get('tensor',{}).get('status')=='available':print('  tensor:',json.dumps(result['tensor'],ensure_ascii=False))
        print('  capture loss:',json.dumps(run['capture_loss'],ensure_ascii=False))
        for row in rows:
            print('trace:',row['trace_uid'])
            print('  import:',row['import_status'],' integrity:',row.get('integrity','failed'))
            print('  format:',row.get('format_applicability',{}).get('status','unavailable'))
            print('  native ownership:',json.dumps(row.get('native_ownership',{}),ensure_ascii=False))
            print('  project linkage:',json.dumps(row.get('project_linkage',{}),ensure_ascii=False))
            print('  coverage denominator:',row.get('coverage_scope','unavailable'))
            ids=row.get('identities',{})
            print('  identities:',json.dumps(dict(hardware=ids.get('hardware'),
                collector=ids.get('collector',{}).get('facts'),parser=ids.get('parser')),ensure_ascii=False))
            print('  source disposition:',json.dumps(row.get('disposition',{}),ensure_ascii=False))
            print('  unknown categories:',json.dumps(row.get('unknown_categories',{}),ensure_ascii=False))
            print('  evidence source:',json.dumps(row.get('source',{}),ensure_ascii=False))
            print('  reasons:',json.dumps(row.get('reason_groups',row.get('reasons',[])),ensure_ascii=False))
            print('  capture completeness:',row.get('capture_completeness',{}).get('status','unchecked'))
    return int(any(r['status']=='failed' for r in rows))
