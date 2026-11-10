# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Single-sample public-entry accuracy fixture, reusing the existing CPU model."""
import argparse
import torch
from flagprof import current_attachment
from flagprof.runner import run_prepared
from tests.probes.cpu_target import Model


def main():
    p=argparse.ArgumentParser();p.add_argument('--outputs',required=True);p.add_argument('--delta',type=float,default=0.)
    args=p.parse_args();torch.set_num_threads(1);torch.manual_seed(531)
    model=Model().eval();inputs=torch.arange(2).reshape(1,2);handle=current_attachment()
    if handle:
        handle.register_model(model)
        handle.bind_tensor_inputs(['fixture-sample'],{'input_ids':inputs},identity=dict(model_files={'fixture':'seed531'},oracle_id='cpu-fixture',
            inputs_sha256='cpu-arange2',samples=['fixture-sample'],batches=[[1,2]],dtype='float32',attention='fixture',pooling='mean',probe_alg='p4v1',
            device='cpu',software={},numerics={},flaggems='off',include=None))
    def transform(output):
        pooled=output.mean(1);result=pooled.clone();result[0,0]+=args.delta
        if handle:handle.record_tensor('pooled',pooled);handle.record_tensor('output',result)
        return result
    run_prepared(model,inputs,forwards=1,outputs=args.outputs,transform=transform)


if __name__=='__main__':main()
