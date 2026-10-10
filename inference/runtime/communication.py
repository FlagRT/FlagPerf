"""Runtime collective capture. Offline interpretation lives in communication_analysis."""
from pathlib import Path
from runtime.common import write_json


class Capture:
    """Diagnostic-only module scopes and collective payload observations."""
    def __init__(self,model,root,rank,roles=None):
        self.model,self.root,self.rank=model,Path(root),rank
        self.roles = roles or {}
        self.active=False
        self.batch_index=self.cycle=0
        self.calls=[];self.handles=[];self.stack=[]
        self.step_context={}

    def set_step(self, context):
        self.step_context={k:context[k] for k in ('phase','generation_step','kv_length_before') if k in context}

    def __enter__(self):
        import torch
        import torch.distributed as dist
        self.native={n:getattr(dist,n) for n in ('all_reduce','all_gather','all_gather_into_tensor')}
        for name,module in self.model.named_modules():
            if name in self.roles:
                def enter(_m,_i,name=name): self.stack.append(name)
                def leave(_m,_i,_o):
                    if self.stack: self.stack.pop()
                self.handles.append(module.register_forward_pre_hook(enter))
                # Appended after Transformers' TP output hook, which performs AllReduce.
                self.handles.append(module.register_forward_hook(leave,always_call=True))
        def wrap(collective):
            native=self.native[collective]
            def observed(*args,**kwargs):
                if not self.active: return native(*args,**kwargs)
                tensor=(args[0] if args else kwargs['tensor']) if collective=='all_reduce' else (
                    args[1] if len(args)>1 else kwargs['tensor' if collective=='all_gather' else 'input_tensor'])
                name=self.stack[-1] if self.stack else None
                category=self.roles.get(name,'unattributed')
                seq=len(self.calls)
                marker=f'flagperf/collective/{seq}/{category}'
                outputs=[tensor] if collective=='all_reduce' else (args[0] if args else kwargs['tensor_list']) if collective=='all_gather' else [args[0] if args else kwargs['output_tensor']]
                size=tensor.numel()*tensor.element_size()
                row={'rank':self.rank,'sequence':seq,'batch_index':self.batch_index,'cycle':self.cycle,
                     'module':name,'category':category,'collective':collective,'marker':marker,**self.step_context,
                     'backend':dist.get_backend(kwargs.get('group',args[2] if len(args)>2 else None)),
                     'shape':list(tensor.shape),'dtype':str(tensor.dtype),'elements':tensor.numel(),
                     'output_tensors':[{'shape':list(t.shape),'dtype':str(t.dtype),'elements':t.numel()} for t in outputs],
                     'logical_bytes':size,'input_logical_bytes':size,
                     'output_logical_bytes':sum(t.numel()*t.element_size() for t in outputs)}
                self.calls.append(row)
                with torch.profiler.record_function(marker): return native(*args,**kwargs)
            return observed
        for name in self.native: setattr(dist,name,wrap(name))
        return self

    def __exit__(self,*exc):
        import torch.distributed as dist
        for name,native in self.native.items(): setattr(dist,name,native)
        for handle in self.handles: handle.remove()
        write_json(self.root/'collectives.json',self.calls)
