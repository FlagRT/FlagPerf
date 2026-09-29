# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Re-render sealed results into new directories, retaining source evidence links."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime.common import ROOT, read_json, write_json, file_hash
from reporting.model import render


def replay(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if not (source/'result.json').is_file():
        raise ValueError('sealed source result.json is missing: '+str(source))
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('replay output and sealed source directories must not overlap')
    result = read_json(source/'result.json')
    if result.get('command') == 'performance':
        # Share the generated-file exclusions and source protection with report.
        # In particular, layer-index.md must never be rendered through a source link.
        from reporting.replay import replay as performance_view
        performance_view(source, destination, result.get('level', 'total'))
        record = {'source': str(source), 'report': str(destination/'report.md'),
                  'source_unchanged': True, 'new_device_execution': False}
        write_json(destination/'report-replay.json', record)
        return record
    destination.mkdir(parents=True,exist_ok=False)
    protected = {name:file_hash(source/name) for name in
                 ['result.json','report.md','comparison.json','policy.yaml','preview/policy.yaml','report-source.json']
                 if (source/name).is_file()}
    for path in source.iterdir():
        if path.name not in ['report.md','report-replay.json','report-source.json']:
            (destination/path.name).symlink_to(path,target_is_directory=path.is_dir())
    render(destination,read_json(source/'result.json'))
    for name,digest in protected.items():
        if file_hash(source/name) != digest:
            raise AssertionError('replay changed sealed evidence: '+name)
    write_json(destination/'report-replay.json', {
        'source':str(source),'source_files_sha256':protected,
        'generator_sha256':{name:file_hash(ROOT/name) for name in
                            ['reporting/model.py','reporting/parallel.py',
                             'reporting/components.py','analysis/assessment.py']},
        'new_device_execution':False})
    return {'source':str(source),'report':str(destination/'report.md'),'source_unchanged':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if any(not (source/'result.json').is_file() for source in args.source):
        parser.error('every source must contain a readable result.json')
    if any(args.output.resolve().is_relative_to(source.resolve())
           or source.resolve().is_relative_to(args.output.resolve()) for source in args.source):
        parser.error('output and sealed source directories must not overlap')
    if len({source.resolve().name for source in args.source}) != len(args.source):
        parser.error('source directory names must be unique when rebuilding multiple reports')
    args.output.mkdir(parents=True,exist_ok=False)
    results=[replay(source,args.output/source.name) for source in args.source]
    write_json(args.output/'replay-summary.json',results)
    print('Replayed',len(results),'sealed reports; no original files changed')
