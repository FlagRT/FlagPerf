# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Runtime collective capture. Offline interpretation lives in communication_analysis."""
from pathlib import Path
from runtime.common import write_json


class Capture:
    """Diagnostic-only module scopes and collective payload observations."""
    def __init__(self,model,root,rank):
        self.model,self.root,self.rank=model,Path(root),rank
        self.active=False
        self.batch_index=self.cycle=0
        self.calls=[];self.handles=[];self.stack=[]

    def __enter__(self):
        import torch
        import torch.distributed as dist
        self.native=dist.all_reduce
        for name,module in self.model.named_modules():
            if name.endswith('self_attn.o_proj') or name.endswith('mlp.down_proj'):
                def enter(_m,_i,name=name): self.stack.append(name)
                def leave(_m,_i,_o): self.stack.pop()
                self.handles.append(module.register_forward_pre_hook(enter))
                # Appended after Transformers' TP output hook, which performs AllReduce.
                self.handles.append(module.register_forward_hook(leave))
        def observed(tensor,*args,**kwargs):
            if not self.active: return self.native(tensor,*args,**kwargs)
            name=self.stack[-1] if self.stack else None
            category=('attention_output' if name and 'self_attn' in name else
                      'mlp_output' if name and 'mlp' in name else 'unattributed')
            seq=len(self.calls)
            marker=f'flagperf/collective/{seq}/{category}'
            row={'rank':self.rank,'sequence':seq,'batch_index':self.batch_index,'cycle':self.cycle,
                 'module':name,'category':category,'collective':'all_reduce','marker':marker,
                 'backend':dist.get_backend(kwargs.get('group',args[1] if len(args)>1 else None)),
                 'shape':list(tensor.shape),'dtype':str(tensor.dtype),'elements':tensor.numel(),
                 'logical_bytes':tensor.numel()*tensor.element_size()}
            self.calls.append(row)
            with torch.profiler.record_function(marker): return self.native(tensor,*args,**kwargs)
        dist.all_reduce=observed
        return self

    def __exit__(self,*exc):
        import torch.distributed as dist
        dist.all_reduce=self.native
        for handle in self.handles: handle.remove()
        write_json(self.root/'collectives.json',self.calls)
