"""Classify observed operator keys using the run's torch registration tables."""
import argparse
import json
from pathlib import Path


def classify(name, overload=''):
    import torch
    operation = getattr(getattr(torch.ops.aten, name.removeprefix('aten::')), overload or 'default')
    schema = operation._schema
    aliases = set().union(*(a.alias_info.after_set for a in schema.arguments if a.alias_info))
    if any(r.alias_info and not r.alias_info.is_write and r.alias_info.after_set & aliases for r in schema.returns) and not any(
            a.alias_info and a.alias_info.is_write for a in schema.arguments):
        return 'view'
    key = schema.name + ('.'+schema.overload_name if schema.overload_name else '')
    return 'composite' if torch._C._dispatch_has_kernel_for_dispatch_key(key, 'CompositeImplicitAutograd') else 'kernel'


def build(keys):
    import torch
    rows = {}
    for name, overload in sorted(set(keys)):
        key = name + ('.'+overload if overload else '')
        try:
            rows[key] = dict(kind=classify(name, overload))
        except (AttributeError, RuntimeError) as error:
            rows[key] = dict(kind='unknown', reason=str(error))
    return dict(torch_version=torch.__version__, entries=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--env',help='registered historical image package provenance, when the old Store lacks torch version')
    args = parser.parse_args()
    from flagprof.store import StoreView
    with StoreView(args.store) as view:
        from flagprof.analyzers.workspace import AnalysisSet
        sources=AnalysisSet.open(view)
        try:
            table=build((s.record['name'],s.record['overload']) for source in sources.views() for s in source.iter_events() if s.record['t']=='sig')
        finally:sources.close()
    table['run_id']=Path(args.store).name
    if args.env:
        env=json.loads(Path(args.env).read_text())
        version=env['packages']['torch']
        if version!=table['torch_version']:raise ValueError('registered image torch version mismatch')
        import hashlib
        table['external_run_environment']=dict(torch_version=version,source=str(args.env),sha256=hashlib.sha256(Path(args.env).read_bytes()).hexdigest())
    Path(args.out).write_text(json.dumps(table, sort_keys=True, indent=2)+'\n')


if __name__ == '__main__':
    main()
