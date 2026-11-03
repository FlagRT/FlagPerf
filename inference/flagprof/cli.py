# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Small verb-based CLI. Inspect consumes stored facts only."""
from __future__ import annotations
import argparse
import json
import sys


def main(argv=None):
    parser=argparse.ArgumentParser(prog='flagprof')
    verbs=parser.add_subparsers(dest='verb',required=True)
    delivery=verbs.add_parser('delivery',help='publish portable evidence archives and restore evidence trees')
    delivery.add_argument('action',choices=['publish','verify','restore'],help='publish source trees to a new delivery directory, verify a delivery, or restore it to a new ext4 directory')
    delivery.add_argument('paths',nargs='+',help='publish: one or more source trees; verify/restore: exactly one delivery directory')
    delivery.add_argument('--out',help='publish: new delivery directory on any writable filesystem; restore: new ext4 directory (required for both)')
    delivery.add_argument('--max-bytes',type=int,help='byte budget (default 200 GiB for publish, 40 GiB for restore)')
    inspect=verbs.add_parser('inspect',help='read stored identities and evidence states')
    inspect.add_argument('run',help='Store run directory');inspect.add_argument('--verify',action='store_true',help='verify every derived and raw artifact')
    inspect.add_argument('--json',action='store_true',help='print the full inspection as JSON')
    inspect.add_argument('--bounded',action='store_true',help='compatibility flag; Plan B runs use bounded verification automatically')
    inspect.add_argument('--work',help='Plan B runs: query work directory outside the runs root');inspect.add_argument('--scratch',help='Plan B runs: ext4 directory for rebuildable indexes (default FLAGPROF_SCRATCH or /var/tmp/flagprof-offline-UID)')
    record_parser=verbs.add_parser('record',help='capture one complete official Python session')
    record_parser.add_argument('--store',required=True,help='output root; final mode keeps the Store under <store>/runs')
    record_parser.add_argument('--vendor',required=True,help='vendor adapter, e.g. ascend or nvidia')
    record_parser.add_argument('--preset',choices=['perf','modules','perf-lite','coverage','accuracy'],default='perf',help='collection preset (default perf); perf/modules use the official profiler in final mode')
    record_parser.add_argument('--flaggems', choices=['off','on'],help='FlagGems component (preset default off)')
    record_parser.add_argument('--flaggems-include',help='JSON array of FlagGems function names to register')
    record_parser.add_argument('--dispatch', choices=['off','summary','each-window'],help='per-call dispatch records; each-window links device events to project signatures and layers (preset default)')
    record_parser.add_argument('--layers', choices=['all','selected'],help='module layers to observe (preset default; accuracy requires all)')
    record_parser.add_argument('--probes', choices=['default','none'],help='project probes (preset default)')
    record_parser.add_argument('--profile-memory', action='store_true', help='collect official allocator memory records')
    record_parser.add_argument('--capture-only',action='store_true',help='seal NVIDIA raw capture for later ext4 import')
    record_parser.add_argument('--capture-filesystem',choices=['ext4','overlay'],default='ext4',help='capture directory filesystem; overlay only with NVIDIA --capture-only')
    record_parser.add_argument('--tensor-blob',default=None,help='none (default), all or comma-separated call_key prefixes')
    record_parser.add_argument('--tensor-blob-bytes',type=int,help='accuracy: total tensor blob budget in bytes (default 1 GiB)')
    record_parser.add_argument('--tensor-ops-in',help='accuracy: record per-operator tensors inside this module path; requires batch size 1')
    record_parser.add_argument('--tensor-boundaries',metavar='model|layer|PATH',help='accuracy: model keeps runner outputs only; layer uses default layer boundaries (default); PATH is a JSON array of module paths; --tensor-ops-in adds operator capture independently')
    record_parser.add_argument('--capture-mode',choices=['windowed','final'],default='final',help='final (default): one official session for the whole target; windowed: step windows')
    record_parser.add_argument('--source-max-bytes',type=int,default=1<<30,help='sealed source limit; OneRec uses 8589934592')
    record_parser.add_argument('--timeout',type=float,default=1800,help='whole record deadline in seconds')
    record_parser.add_argument('--work',help='durable Plan B checkpoints, separate from store')
    record_parser.add_argument('--scratch',help='local disk for rebuildable Plan B indexes')
    record_parser.add_argument('--active',type=int,default=None,help='windowed: active steps per window (default 5)')
    record_parser.add_argument('--windows',type=int,default=None,help='windowed: number of windows (default 2)')
    record_parser.add_argument('--invalid-policy',choices=['stop','continue'],default=None,help='windowed: action after an invalid window (default stop)')
    record_parser.add_argument('--log-file',action='append',default=[],help='explicit log source; may be repeated')
    record_parser.add_argument('--plugins',help='component descriptors JSON (not project binding rules)')
    record_parser.add_argument('command',nargs=argparse.REMAINDER,help='target Python command after --')
    launch=verbs.add_parser('launch',help='explicit two-rank workers with separate Stores')
    launch.add_argument('--nproc-per-node',type=int,default=2,help='number of ranks; only 2 is accepted')
    launch.add_argument('--comm-backend',choices=['hccl','flagcx','nccl'],required=True,help='collective backend of the model group')
    launch.add_argument('--vendor',choices=['ascend','nvidia','thead'],help='explicit hardware adapter; PPU requires thead')
    launch.add_argument('--capture-only',action='store_true',help='seal per-rank profiler captures for later ext4 import')
    launch.add_argument('--capture-filesystem',choices=['ext4','overlay'],default='ext4',help='overlay requires NVIDIA capture-only')
    launch.add_argument('--store',required=True,help='group root: one rank-N Store per rank plus group.json for report')
    launch.add_argument('--preset',choices=['none','accuracy','perf','modules','perf-lite'],default='perf',help='per-rank preset (default perf); none runs the target without collection')
    launch.add_argument('--timeout',type=float,default=1800,help='whole launch deadline in seconds')
    launch.add_argument('--source-max-bytes',type=int,default=1<<30,help='sealed source limit per rank in bytes (default 1 GiB)')
    launch.add_argument('--tensor-blob',choices=['all','none'],default='all',help='accuracy: save full tensor blobs (default all)')
    launch.add_argument('--tensor-blob-bytes',type=int,default=1<<30,help='accuracy: tensor blob budget in bytes (default 1 GiB)')
    launch.add_argument('--profile-memory',action='store_true',help='collect official allocator memory records; requires perf or modules')
    launch.add_argument('command',nargs=argparse.REMAINDER,help='target Python command after --')
    offline=verbs.add_parser('import',help='import a sealed capture into a new run using bounded Plan B')
    offline.add_argument('capture',help='sealed capture directory');offline.add_argument('--store',required=True,help='ext4 runs root that receives the new run');offline.add_argument('--work',help='durable import checkpoints outside the Store');offline.add_argument('--scratch',help='ext4 directory for rebuildable indexes (default FLAGPROF_SCRATCH or /var/tmp/flagprof-offline-UID)')
    offline.add_argument('--run-id',help='name of the new run (default UTC time, host and random suffix)')
    offline.add_argument('--calls-from',help='explicit collection run with matching source_run_id')
    page=verbs.add_parser('page',help='verified bounded Plan B device page')
    page.add_argument('run',help='Plan B run directory');page.add_argument('--work',help='query work directory outside the runs root');page.add_argument('--scratch',help='ext4 directory for rebuildable indexes (default FLAGPROF_SCRATCH or /var/tmp/flagprof-offline-UID)')
    page.add_argument('--offset',type=int,default=0,help='first row (default 0)');page.add_argument('--limit',type=int,default=100,help='rows per page (default 100)')
    seal=verbs.add_parser('seal',help='seal an externally exported trace; caller asserts normal close')
    seal.add_argument('source',help='exported trace file');seal.add_argument('--out',required=True,help='new sealed capture directory on ext4');seal.add_argument('--vendor',choices=['nvidia','ascend','thead'],required=True,help='vendor format of the trace')
    seal.add_argument('--normal-close',action='store_true',required=True,help='required assertion that the profiler session closed normally')
    seal.add_argument('--source-max-bytes',type=int,default=1<<30,help='copy limit for the trace in bytes (default 1 GiB)')
    seal.add_argument('--identity-from',help='same-capture attachment state or final capture directory')
    analyse=verbs.add_parser('analyse',help='retry official analysis of already sealed final raw capture')
    analyse.add_argument('capture',help='sealed final raw capture directory')
    report=verbs.add_parser('report',help='generate bounded offline analysis and a self-contained HTML report')
    report.add_argument('run',help='one of: a Store run directory (single-run report); a group manifest file such as <launch store>/group.json (one report per rank); or the word oracle (compare --ref with --cmp)');report.add_argument('--out',help='new output directory outside the inputs (default <run>-report, <group dir>-report or <ref>-oracle beside the input)');report.add_argument('--work',help='parent of temporary analysis files, outside the input Stores (default system temp)')
    report.add_argument('--reports',nargs='+',help='compare: generated report directories; keep incompatible values with not_comparable reasons')
    report.add_argument('--ref',help='oracle: reference accuracy run');report.add_argument('--cmp',help='oracle: compared accuracy run');report.add_argument('--declare',help='oracle: JSON file whose differences list names identity fields allowed to differ');report.add_argument('--perf-run',help='oracle: alias of --perf-cmp-run')
    report.add_argument('--perf-ref-run',help='oracle: performance run linked to the reference side by exact signature');report.add_argument('--perf-cmp-run',help='oracle: performance run linked to the compared side by exact signature')
    report.add_argument('--replay-ref-run',help='oracle: run created by replay run --side ref; requires --replay-cmp-run');report.add_argument('--replay-cmp-run',help='oracle: run created by replay run --side cmp; requires --replay-ref-run')
    report.add_argument('--op-kinds',help='operator categories generated in the run torch environment')
    report.add_argument('--limits',help='JSON file of analysis resource limits')
    replay=verbs.add_parser('replay',help='prepare portable inputs or execute a pure operator')
    replay_verbs=replay.add_subparsers(dest='replay_verb',required=True)
    prepare=replay_verbs.add_parser('prepare',help='build a portable input package from two accuracy runs')
    for name,text in (('ref','reference accuracy run with full tensor blobs'),('cmp','compared accuracy run with full tensor blobs'),('call-key','call_key of the target operator'),('out','new input package directory')):prepare.add_argument('--'+name,required=True,help=text)
    prepare.add_argument('--work',help='parent of temporary preparation files')
    execute=replay_verbs.add_parser('run',help='execute one side of an input package into a new replay Store run')
    execute.add_argument('package',help='input package from replay prepare');execute.add_argument('--side',choices=['ref','cmp'],required=True,help='side whose registered conditions are used')
    execute.add_argument('--device',required=True,help='torch device of the same family as the side, e.g. cuda:0 or npu:0');execute.add_argument('--store',required=True,help='Store root that receives the replay run')
    args=parser.parse_args(argv)
    from pathlib import Path
    if args.verb=='launch':
        try:
            from .launch import launch
            command=args.command[1:] if args.command[:1]==['--'] else args.command
            config=dict(flaggems='off')
            if args.preset=='accuracy':config.update(tensor_blob=args.tensor_blob,tensor_blob_bytes=args.tensor_blob_bytes)
            if args.profile_memory:
                if args.preset not in ('perf','modules'):raise ValueError('official memory requires perf/modules')
                config['profile_memory']=True
            return launch(command,store=args.store,nproc_per_node=args.nproc_per_node,comm_backend=args.comm_backend,
                          vendor=args.vendor,
                          preset=args.preset,config=config,timeout=args.timeout,source_max_bytes=args.source_max_bytes,
                          capture_only=args.capture_only,capture_filesystem=args.capture_filesystem)
        except (ValueError,OSError,KeyError,RuntimeError) as error:
            print('flagprof launch: '+str(error),file=sys.stderr);return 2
    if args.verb=='delivery':
        try:
            from .delivery import publish_tree,verify,restore
            options={} if args.max_bytes is None else dict(max_bytes=args.max_bytes)
            if args.action=='publish':
                if not args.out:raise ValueError('publish requires --out')
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
            result=(replay.prepare(args.ref,args.cmp,args.call_key,args.out,work=args.work)
                if args.replay_verb=='prepare' else replay.run(args.package,side=args.side,device=args.device,store=args.store))
            print(json.dumps(result,ensure_ascii=False));return 0 if result.get('status','complete')=='complete' else 2
        except (ValueError,OSError,KeyError,TypeError,RuntimeError) as error:
            print('flagprof replay: '+str(error),file=sys.stderr);return 2
    if args.verb=='report':
        if args.run=='compare':
            if not args.reports or len(args.reports)<2 or not args.out:parser.error('report compare requires --reports (at least two) and --out')
            from .report import compare_device_reports
            compare_device_reports(args.reports,args.out)
            print(json.dumps(dict(output=args.out,status='complete')))
            return 0
        try:
            from .report import generate, generate_oracle
            if args.run=='oracle':
                if not args.ref or not args.cmp:raise ValueError('oracle requires --ref and --cmp')
                result=generate_oracle(args.ref,args.cmp,args.out,work=args.work,perf_run=args.perf_run,
                    perf_ref_run=args.perf_ref_run,perf_cmp_run=args.perf_cmp_run,
                    replay_ref_run=args.replay_ref_run,replay_cmp_run=args.replay_cmp_run,
                    declare=json.loads(Path(args.declare).read_text()) if args.declare else None,
                    limits=json.loads(Path(args.limits).read_text()) if args.limits else None)
                print(json.dumps(result,ensure_ascii=False));return 0
            if Path(args.run).is_file():
                from .report.group import generate_group
                result=generate_group(args.run,args.out,work=args.work)
                print(json.dumps(result));return 0
            result=generate(args.run,args.out,work=args.work,
                            op_kinds=json.loads(Path(args.op_kinds).read_text()) if args.op_kinds else None,
                            limits=json.loads(Path(args.limits).read_text()) if args.limits else None)
            print(json.dumps(result,ensure_ascii=False));return 0
        except (ValueError,OSError,KeyError,TypeError) as error:
            print('flagprof report: '+str(error),file=sys.stderr);return 2
    if args.verb in ('import','page','seal','analyse') or (args.verb=='inspect' and (args.bounded or (Path(args.run)/'plan-b-capture.json').exists())):
        try:
            from pathlib import Path
            if args.verb=='import':
                from .offline.service import import_bundle
                options=None
                if args.calls_from:
                    from .offline.bundle import load_bundle
                    from .offline.project import calls_from
                    capture=load_bundle(args.capture)
                    options=dict(calls=calls_from(args.calls_from,capture['metadata'].get('source_run_id')),marker_prefix='flagprof::op#')
                result=import_bundle(args.capture,store=args.store,work=args.work,scratch=args.scratch,run_id=args.run_id,options=options)
            elif args.verb in ('page','inspect'):
                from .offline.service import inspect_run
                result=inspect_run(args.run,args.work,scratch=args.scratch,**(dict(offset=args.offset,limit=args.limit) if args.verb=='page' else {}))
            elif args.verb=='seal':
                from .offline.bundle import make_bundle,copy_checked
                from tempfile import TemporaryDirectory
                from .offline.workspace import check_disk
                source=Path(args.source).resolve()
                destination=Path(args.out).resolve()
                check_disk(destination)
                destination.parent.mkdir(parents=True,exist_ok=True)
                with TemporaryDirectory(prefix='.flagprof-seal-',dir=destination.parent) as temp:
                    # Streaming copy; seal only this file, never its arbitrary parent.
                    copy_checked(source,Path(temp)/'trace.json',max_bytes=args.source_max_bytes)
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
            print(json.dumps(result,ensure_ascii=False));return 0
        except (ValueError,OSError,KeyError,TypeError,RuntimeError) as error:
            print('flagprof '+args.verb+': '+str(error),file=sys.stderr);return 2
    if args.verb=='record':
        from .record import record
        from pathlib import Path
        command=args.command[1:] if args.command[:1]==['--'] else args.command
        try:
            if (args.capture_only or args.capture_filesystem!='ext4') and (args.capture_mode!='final' or args.preset not in ('perf','modules')):
                raise ValueError('capture-only/filesystem options require official final mode')
            config = {name:getattr(args,name) for name in ('flaggems','flaggems_include','layers','probes') if getattr(args,name) is not None}
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
                if args.active is not None or args.windows is not None or args.invalid_policy is not None or args.plugins or args.log_file:
                    raise ValueError('final mode does not accept window/plugin/log options')
                if args.preset in ('coverage','perf-lite','accuracy'):
                    return record(command,store=args.store,work=args.work,scratch=args.scratch,vendor=args.vendor,preset=args.preset,config=config,timeout=args.timeout)
                from .offline.capture import record_final
                return record_final(command,store=args.store,work=args.work,scratch=args.scratch,vendor=args.vendor,preset=args.preset,
                                    timeout=args.timeout,
                                    **(dict(source_max_bytes=args.source_max_bytes) if args.source_max_bytes!=1<<30 else {}),
                                    **(dict(capture_only=args.capture_only,capture_filesystem=args.capture_filesystem) if args.capture_only or args.capture_filesystem!='ext4' else {}),
                                    **(dict(config=config) if config else {}))
            config.update(active=5 if args.active is None else args.active,windows=2 if args.windows is None else args.windows,invalid_policy='stop' if args.invalid_policy is None else args.invalid_policy,
                          log_files=tuple(str(Path(p).resolve()) for p in args.log_file))
            return record(command,store=args.store,vendor=args.vendor,preset=args.preset,
                capture_mode='windowed',
                timeout=args.timeout,
                config=config,
                descriptors=json.loads(Path(args.plugins).read_bytes()) if args.plugins else None)
        except (ValueError,OSError) as error:
            parser.error(str(error))
    from .store import StoreView
    try:
        with StoreView(args.run) as view:
            run=view.run_status()
            rows=view.inspect(verify=args.verify)
            tensor=view.tensor_status(verify=args.verify)
    except (ValueError,OSError,KeyError,TypeError) as error:
        print('flagprof inspect: '+str(error),file=sys.stderr)
        return 2
    return _print_inspection(dict(schema='inspect/2',run=run,traces=rows,tensor=tensor), args.json)


def _print_inspection(result, as_json):
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
