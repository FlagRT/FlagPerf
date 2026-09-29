# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Offline views, portable evidence and bounded reanalysis of copied profiler data."""
from copy import deepcopy
from pathlib import Path
import os
import shutil
import subprocess
import uuid

from runtime.common import ROOT, read_json, write_json, file_hash, execute


GENERATED = {'result.json', 'report.md', 'report-source.json', 'report-view.json',
             'layer-index.md', 'layer-details.md', 'layer-view.json', 'package-manifest.json', 'reanalysis.json'}


def files(root):
    """Materialize view links while detecting directory cycles and special files."""
    def walk(path, relative, parents):
        real = path.resolve(strict=True)
        if real.is_dir():
            if real in parents:
                raise ValueError('证据目录包含循环链接：'+str(path))
            for child in sorted(path.iterdir()):
                yield from walk(child, relative/child.name, parents | {real})
        elif real.is_file():
            yield relative.as_posix(), path
        else:
            raise ValueError('证据包接受普通文件和目录：'+str(path))
    yield from walk(root, Path(), set())


def manifest(root):
    return {name: file_hash(path) for name, path in files(root)}


def export_in_image(root):
    """Host export uses the recorded image and management nodes; data stays in the copy."""
    def export(command, logs, timeout):
        image_file = root/'image.json'
        if not image_file.is_file():
            raise ValueError('导出原始采集需要来源 image.json 中的镜像身份')
        image = read_json(image_file)
        image_id = image[0]['Id'] if isinstance(image, list) else image['Id']
        if os.environ.get('FLAGPERF_IMAGE_ID') == image_id:
            return execute(command, logs, timeout)
        name = 'flagperf-export-'+uuid.uuid4().hex[:12]
        argv = ['docker', 'run', '--rm', '--name', name, '--network', 'none',
                '--user', f'{os.getuid()}:{os.getgid()}', '-v', f'{root}:{root}:rw',
                '-e', 'HOME=/tmp', '-e', 'USER=flagperf', '-e', 'LOGNAME=flagperf',
                '-e', 'TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_flagperf',
                '-e', 'PYTHONDONTWRITEBYTECODE=1']
        for device in ('davinci_manager', 'devmm_svm', 'hisi_hdc'):
            if Path('/dev', device).exists(): argv += ['--device', f'/dev/{device}:/dev/{device}:rwm']
        for path in ('/usr/local/Ascend/driver', '/usr/local/Ascend/firmware', '/usr/local/dcmi', '/etc/ascend_install.info'):
            if Path(path).exists(): argv += ['-v', f'{path}:{path}:ro']
        argv += ['--entrypoint', 'python3', image_id, *command[1:]]
        try:
            return execute(argv, logs, timeout)
        finally:
            subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30)
    return export


def reanalyze_profiles(root, result):
    from analysis.layer import ranks
    from analysis.layer_trace import analyze, associate_collectives
    layer = result['layer']
    records = []
    communication_records = []
    for side, repeats in layer.get('runs', {}).get('layer_profile', {}).items():
        for label, state in repeats.items():
            for rank, worker in ranks(state).items():
                relative = Path('layer/layer_profile')/side/label
                if layer.get('world_size', 1) > 1: relative /= 'rank-'+rank
                folder = root/relative
                if worker.get('status') != 'completed' or state.get('status') != 'completed':
                    records.append({'side': side, 'repeat': label, 'rank': rank, 'status': 'partial',
                                    'reason': '采样 worker 待完成；可用计时和诊断记录保留'})
                    continue
                # Keep earlier export attempts; each new analysis gets one fresh attempt.
                previous = folder/'profiler-recovery'
                if previous.exists():
                    saved = folder/('profiler-recovery-history-'+uuid.uuid4().hex[:8])
                    previous.rename(saved)
                try:
                    detail = analyze(folder, worker, export=export_in_image(root))
                    if layer.get('world_size', 1) > 1 and detail['status'] == 'completed':
                        from analysis.communication import analyze_rank
                        detail['communication'] = analyze_rank(folder, int(rank))
                        associate_collectives(detail)
                except Exception as error:
                    detail = {'status': 'partial', 'reason': str(error), 'rows': [], 'coverage': {}}
                write_json(folder/'attribution.json', detail)
                value = {k: v for k, v in detail.items() if k not in ('events', 'auxiliary_events')}
                value['evidence'] = str(relative/'attribution.json')
                layer.setdefault('profiles', {}).setdefault(side, {}).setdefault(label, {})[rank] = value
                status = detail['status']
                if detail.get('communication', {}).get('status', 'completed') != 'completed': status = 'partial'
                records.append({'side': side, 'repeat': label, 'rank': rank, 'status': status,
                                'reason': detail.get('reason'), 'evidence': value['evidence']})
    if layer.get('world_size', 1) > 1:
        from analysis.communication import analyze as analyze_communication
        for side, repeats in layer.get('runs', {}).get('layer_profile', {}).items():
            for label in repeats:
                try:
                    communication = analyze_communication(root/'layer/layer_profile'/side/label, layer['world_size'])
                    result.setdefault('profiles', {}).setdefault(side, {})[label] = {
                        'status': communication['status'], 'communication': communication}
                    communication_records.append({'side': side, 'repeat': label, 'status': communication['status']})
                    alias = root/'profiles'/side/label
                    shutil.copytree(root/'layer/layer_profile'/side/label, alias, dirs_exist_ok=True)
                except Exception as error:
                    result.setdefault('profiles', {}).setdefault(side, {})[label] = {'status': 'partial', 'reason': str(error)}
                    communication_records.append({'side': side, 'repeat': label, 'status': 'partial', 'reason': str(error)})
    info = {'status': 'completed' if records and all(r['status'] == 'completed' for r in records+communication_records) else 'partial',
            'new_device_execution': False, 'records': records, 'communication': communication_records,
            'scope': 'profiler attribution and communication; source execution status retained'}
    result['report_view']['reanalysis'] = info
    write_json(root/'reanalysis.json', info)
    write_json(root/'layer/summary.json', layer)
    return info


def replay(source, destination, level, *, layers=None, selected_ranks=None, shapes=None, portable=False, reanalyze=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('report source and output directories must not overlap')
    result = read_json(source/'result.json')
    if result.get('command') != 'performance':
        raise ValueError('offline level views require a performance result')
    if level == 'layer' and result.get('level') != 'layer' and 'layer' not in result:
        raise ValueError('source contains no layer evidence; run performance --level layer first')
    if level != 'layer' and (layers or selected_ranks is not None or shapes or reanalyze):
        raise ValueError('层筛选与重新分析请使用 --level layer')
    from reporting.layer_view import selection
    selected = selection(result, layers, selected_ranks, shapes) if level == 'layer' else {}
    portable = portable or reanalyze
    protected = manifest(source) if portable else {
        name: file_hash(source/name) for name in ('result.json', 'report.md', 'source-snapshot.json') if (source/name).is_file()}
    destination.mkdir(parents=True, exist_ok=False)
    if portable:
        for relative, path in files(source):
            if relative.split('/')[0] in GENERATED: continue
            target = destination/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if file_hash(target) != protected[relative]: raise ValueError('证据复制校验失败：'+relative)
        shutil.copy2(source/'result.json', destination/'source-result.json')
        if (source/'report.md').is_file(): shutil.copy2(source/'report.md', destination/'source-report.md')
    else:
        for path in source.iterdir():
            if path.name not in GENERATED:
                (destination/path.name).symlink_to(path, target_is_directory=path.is_dir())
    view = deepcopy(result)
    view['level'] = level
    view['report_view'] = {'source': str(source), 'source_sha256': protected, 'new_device_execution': False,
                           'source_status': result.get('status'), 'portable': portable, 'selection': selected}
    if reanalyze: reanalyze_profiles(destination, view)
    write_json(destination/'result.json', view)
    from reporting.model import render
    render(destination, view)
    write_json(destination/'report-view.json', view['report_view'])
    after = manifest(source) if portable else {n: file_hash(source/n) for n in protected}
    if after != protected: raise RuntimeError('source evidence changed during report generation')
    if portable:
        write_json(destination/'package-manifest.json', {'schema_version': 1, 'files': manifest(destination),
                   'source_unchanged': True, 'layout': 'materialized evidence; relative report links',
                   'verify': 'SHA256 for every packaged file except this manifest'})
    return destination/'report.md'


def run(args):
    if not args.source or not args.output or not args.level:
        raise ValueError('report requires --source, --level total|layer and --output')
    from runtime.config import parser
    defaults = vars(parser().parse_args(['report']))
    allowed = {'command', 'source', 'level', 'output', 'layers', 'ranks', 'shapes', 'portable', 'reanalyze'}
    if any(value != defaults[name] for name, value in vars(args).items() if name not in allowed):
        raise ValueError('report 使用来源、视图筛选、portable、reanalyze 和输出选项')
    print('Report: '+str(replay(args.source, args.output, args.level, layers=args.layers,
          selected_ranks=args.ranks, shapes=args.shapes, portable=args.portable, reanalyze=args.reanalyze)))
    return 0
