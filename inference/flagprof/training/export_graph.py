# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Training-side model graph export: the CTR task model captured on CPU.

Builds the exact model the training engine trains (same builder, same sealed
package, same seed and batch shape), then reuses the inference graph export
machinery (ExportSpec + export_into) so the structure/full bundles, ONNX/FX
formats and the HTML reading page are identical. Export is a standalone CPU
process like the inference entry: it does not attach a profiler and does not
change training semantics.

CLI: python -m flagprof.training.export_graph --ctr-arch <arch> --package <dir>
     --out <dir> [--formats fx,onnx] [--bundle structure|full] [--verify-onnx]
"""
import argparse
import json
from pathlib import Path


def export_ctr(*, arch, package, out, seed=531, batch_size=4, formats=('fx', 'onnx'),
               bundle='structure', verify_onnx=False):
    import torch
    from flagprof.graph import ExportSpec
    from flagprof.graph.core import export_into
    from .ctr import prepare_ctr
    net, batches, _loss_fn, protocol = prepare_ctr(arch=arch, package=package, seed=seed,
                                                   batch_size=batch_size)
    if not batches:
        raise ValueError('package produced no batches')
    batch = {k: (v.detach().to('cpu') if isinstance(v, torch.Tensor) else v) for k, v in batches[0].items()}
    tensors = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
    if not tensors:
        raise ValueError('batch carries no tensors')
    model = net.to('cpu').float().eval()
    # Derived label tensors feed the loss, not the model; keep only the
    # parameters the forward signature accepts.
    import inspect
    accepted = set(inspect.signature(model.forward).parameters)
    tensors = {k: v for k, v in tensors.items() if k in accepted}
    if not tensors:
        raise ValueError('no batch tensors match the model forward signature')
    outputs = model(**tensors)
    if isinstance(outputs, torch.Tensor):
        output_names = ('output',)
    elif isinstance(outputs, (tuple, list)):
        output_names = tuple('output'+str(i) for i in range(len(outputs)))
    else:
        raise ValueError('unsupported model output type: '+type(outputs).__name__)
    spec = ExportSpec(model=model, args=(), kwargs=tensors,
                      input_names=tuple(tensors.keys()), output_names=output_names,
                      identity=dict(kind='training-graph', task='ctr', arch=arch,
                                    package=protocol.get('ctr_package_oracle_id'),
                                    weight_sha256=protocol.get('ctr_weight_sha256'),
                                    attention='n/a', components='n/a'))
    root = Path(out)
    if root.exists():
        raise ValueError('export destination already exists: '+str(root))
    root.mkdir(parents=True)
    with torch.no_grad():
        result = export_into(spec, root, formats=tuple(formats), bundle=bundle, verify_onnx=verify_onnx)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ctr-arch', required=True,
                        help='CTR architecture (dlrm din widedeep deepfm dcn sasrec lightgcn bst dien dcnmix esmm mmoe ple dssm)')
    parser.add_argument('--package', required=True, help='sealed ctr package directory')
    parser.add_argument('--seed', type=int, default=531)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--formats', default='fx,onnx')
    parser.add_argument('--bundle', choices=('structure', 'full'), default='structure')
    parser.add_argument('--verify-onnx', action='store_true')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    result = export_ctr(arch=args.ctr_arch, package=args.package, out=args.out, seed=args.seed,
                        batch_size=args.batch_size, formats=tuple(p.strip() for p in args.formats.split(',') if p.strip()),
                        bundle=args.bundle, verify_onnx=args.verify_onnx)
    print(json.dumps(dict(output=str(args.out), status=result.get('status'),
                          formats=sorted(result.get('formats', {}).keys())), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
