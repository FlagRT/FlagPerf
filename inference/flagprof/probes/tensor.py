# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Opt-in accuracy snapshots; device work is limited to selected boundaries."""
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import weakref
from .base import Probe


def leaves(value, path='0'):
    import torch
    if isinstance(value,torch.Tensor):yield path,value
    elif isinstance(value,(tuple,list)):
        for i,v in enumerate(value):yield from leaves(v,path+'['+str(i)+']')
    elif isinstance(value,dict):
        for k,v in sorted(value.items()):yield from leaves(v,path+'{'+str(k)+'}')


def arguments(args,kwargs):
    for i,value in enumerate(args):yield from leaves(value,str(i))
    for key,value in sorted(kwargs.items()):yield from leaves(value,'kw:'+key)


class TensorProbe(Probe):
    id='tensor'
    produces=('tensor','sig')
    requires=('torch',)

    def __init__(self,manager):
        super().__init__(manager)
        self.handles=[];self.model=None;self.paths={};self.boundaries=set()
        self.samples=[];self.mask=None;self.inputs={};self.passes=defaultdict(int);self.frames=[]
        self.sources={};self.parameters={};self.external={};self.calls=defaultdict(int);self.storage_epochs=defaultdict(int)
        self.mode=None;self.stream=None;self.busy=False;self.drilling=False;self.blob_exhausted=False
        self.metrics=dict(events=0,blob_bytes=0,blob_skipped=0,first_skipped=None)
        self.manager.state['tensor']=self.metrics

    def install(self):
        self.installed=True
        if self.config.get('tensor_ops_in'):
            from torch.utils._python_dispatch import TorchDispatchMode
            owner=self
            class Mode(TorchDispatchMode):
                def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                    if not owner.active() or owner.busy or not owner.drilling:return func(*args,**(kwargs or {}))
                    return owner.invoke(func,args,kwargs or {})
            self.mode=Mode();self.mode.__enter__()

    def bind(self,sample_ids,inputs):
        if self.config.get('tensor_ops_in') and len(sample_ids)!=1:
            raise ValueError('tensor-ops-in requires batch 1')
        self.samples=list(sample_ids);self.inputs=inputs
        mask=inputs.get('attention_mask') if isinstance(inputs,dict) else None
        self.mask=mask.detach().to('cpu') if mask is not None else None

    def register_model(self,model):
        import torch
        if self.model is model:return
        self.model=model
        self.paths={id(m):'<root>'+('.'+name if name else '') for name,m in model.named_modules()}
        self.parameters={id(t):('param:'+name,t) for name,t in model.named_parameters()}
        self.parameters.update({id(t):('buffer:'+name,t) for name,t in model.named_buffers()})
        configured=self.config.get('tensor_boundaries')
        if configured is None:configured=getattr(model,'_flagprof_tensor_boundaries',None)
        if configured is not None:
            self.boundaries={p if p.startswith('<root>') else '<root>.'+p for p in configured}
        else:
            embed=model.get_input_embeddings() if hasattr(model,'get_input_embeddings') else getattr(model,'embedding',None)
            if embed is not None:self.boundaries.add(self.paths[id(embed)])
            groups=[m for m in model.modules() if isinstance(m,torch.nn.ModuleList) and len(m) and len({type(v) for v in m})==1]
            if groups:self.boundaries.update(self.paths[id(v)] for v in max(groups,key=len))
        if self.config.get('tensor_ops_in') and self.config['tensor_ops_in'] not in self.paths.values():
            raise ValueError('tensor scope is not a registered module path')
        outputs=getattr(model,'_flagprof_output_boundaries',['pooled','output'])
        self.manager.state['tensor_boundaries']=sorted(self.boundaries|{'<root>#'+name for name in outputs})
        for module in model.modules():
            self.handles.append(module.register_forward_pre_hook(lambda m,a,k:self.safe(self.enter,m,a,k),with_kwargs=True))
            self.handles.append(module.register_forward_hook(lambda m,a,k,o:self.safe(self.leave,m,o),with_kwargs=True,always_call=True))

    def enter(self,module,args,kwargs):
        if not self.samples:return
        path=self.paths[id(module)]
        counts=[self.passes[(path,s)] for s in self.samples]
        for s in self.samples:self.passes[(path,s)]+=1
        item=dict(path=path,counts=counts,drill=path==self.config.get('tensor_ops_in') and counts==[0])
        self.frames.append(item)
        if item['drill']:
            self.drilling=True;self.sources={};self.external={};self.calls.clear()
            for argpath,tensor in arguments(args,kwargs):
                key=self.record(path,0,'in/'+argpath,self.samples[0],tensor,role='in',whole=True)
                self.remember(tensor,key)

    def leave(self,module,output):
        if not self.samples or not self.frames:return
        item=self.frames.pop()
        if item['path']!=self.paths[id(module)]:raise RuntimeError('tensor module hook ownership mismatch')
        if item['path'] in self.boundaries and output is not None:
            values=list(leaves(output))
            if values:self.capture(item['path'],'out',values[0][1],counts=item['counts'])
        if item['drill']:self.drilling=False;self.sources.clear();self.external.clear()

    def capture(self,path,boundary,tensor,counts=None):
        previous_busy=self.busy;self.busy=True
        try:
            if not self.samples:return
            if counts is None:
                counts=[max(0,self.passes[('<root>',s)]-1) for s in self.samples]
            for index,sample in enumerate(self.samples):
                value=tensor[index]
                mask=None
                if self.mask is not None and tensor.ndim>=2 and tuple(tensor.shape[:2])==tuple(self.mask.shape):
                    # Index on CPU after a single snapshot; avoid device-side mask operations.
                    value=value.detach().to('cpu')[self.mask[index].bool()]
                    mask=dict(**{'from':'attention_mask'},valid=int(self.mask[index].sum()))
                self.record(path,counts[index],boundary,sample,value,mask=mask)
        finally:self.busy=previous_busy

    def remember(self,tensor,key):
        try:version=tensor._version
        except RuntimeError:version=None
        storage=tensor.untyped_storage().data_ptr()
        self.sources[id(tensor)]=(weakref.ref(tensor),key,version,storage,self.storage_epochs[storage])

    def source(self,tensor):
        previous=self.sources.get(id(tensor))
        if previous and previous[0]() is tensor:
            try:version=tensor._version
            except RuntimeError:version=None
            if version==previous[2] and self.storage_epochs[previous[3]]==previous[4]:return previous[1]
            return 'unknown'
        parameter=self.parameters.get(id(tensor))
        if parameter and parameter[1] is tensor:
            boundary=parameter[0]
            if boundary not in self.external:
                self.external[boundary]=self.record(self.config['tensor_ops_in'],0,boundary,self.samples[0],tensor,role='in',whole=True)
                self.remember(tensor,self.external[boundary])
            return self.external[boundary]
        return 'unknown'

    def invoke(self,func,args,kwargs):
        from .dispatch import devices
        from flagprof.devices import accelerator
        from .argtree import describe_args
        from flagprof.store.sig import compute_sid
        # Producer metadata failures must never execute the target operation twice.
        self.busy=True
        try:
            schema=func._schema;name=schema.name;overload=schema.overload_name
            sources=[self.source(t) for _,t in arguments(args,kwargs)]
            tree=describe_args(args,kwargs);sid=compute_sid(name,overload,'in',tree)
            self.context.writer.register_signature(dict(name=name,overload=overload,role='in',tree=tree,sid=sid))
            config=[]
            for i,argument in enumerate(schema.arguments):
                present=i<len(args) or argument.name in kwargs
                value=args[i] if i<len(args) else kwargs.get(argument.name,argument.default_value)
                config.append(dict(name=argument.name,value=describe_args([value]),present=present))
            # Explicit versus implicit default arguments have equal semantic configuration.
            for row in config:row.pop('present')
        except Exception as error:
            self.manager.failure(self.id,error);self.busy=False
            return func(*args,**kwargs)
        finally:
            self.busy=False
        with self.context.probe_op(True,op_key=name+('.'+overload if overload else '')):
            output=func(*args,**kwargs)
        for i,argument in enumerate(schema.arguments):
            if argument.alias_info and argument.alias_info.is_write:
                value=args[i] if i<len(args) else kwargs.get(argument.name)
                for _,tensor in leaves(value):self.storage_epochs[tensor.untyped_storage().data_ptr()]+=1
        if not accelerator(devices(args,kwargs,output)):return output
        self.busy=True
        try:
            scope=self.context.current_scope()
            path=scope.name if scope else self.config['tensor_ops_in']
            key=(path,name,overload);index=self.calls[key];self.calls[key]+=1
            op=name+('.'+overload if overload else '')
            for i,(_,tensor) in enumerate(leaves(output)):
                call=self.record(path,0,'op:'+op+'/'+str(index)+'/'+str(i),self.samples[0],tensor,
                    whole=True,sid=sid,src=sources,ext={'call_config':config})
                self.remember(tensor,call)
        except Exception as error:self.manager.failure(self.id,error)
        finally:self.busy=False
        return output

    def record(self,path,iteration,boundary,sample,tensor,*,role='out',whole=False,mask=None,**fields):
        import torch
        busy=self.busy;self.busy=True
        try:
            key=path+'#'+str(iteration)+'#'+boundary+'#'+sample
            cpu=tensor.detach().to('cpu').contiguous()
            raw=cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest='sha256:'+hashlib.sha256(raw).hexdigest()
            flat=cpu.reshape(-1).double()
            def statistics(values):
                finite=values[torch.isfinite(values)]
                return dict(count=values.numel(),nan=int(torch.isnan(values).sum()),inf=int(torch.isinf(values).sum()),
                    min=float(finite.min()) if finite.numel() else None,max=float(finite.max()) if finite.numel() else None,
                    mean=float(finite.mean()) if finite.numel() else None,absmean=float(finite.abs().mean()) if finite.numel() else None,
                    l2=float(torch.linalg.vector_norm(finite)) if finite.numel() else None)
            n=flat.numel();k=min(64,n);width=n//k if k else 0
            chunks=[statistics(flat[i*width:n if i==k-1 else (i+1)*width]) for i in range(k)]
            m=min(256,n);step=n//m if m else 1
            phase=int(hashlib.sha256(key.encode()).hexdigest()[:16],16)%step
            probe=[]
            for v in flat[phase:phase+step*m:step].tolist():
                probe.append(v if __import__('math').isfinite(v) else 'nan' if __import__('math').isnan(v) else '+inf' if v>0 else '-inf')
            event=dict(t='tensor',call_key=key,role=role,boundary=boundary,shape=list(cpu.shape),dtype=str(cpu.dtype),
                stats=statistics(flat),chunks=chunks,probe=probe,probe_alg='p4v1',digest=digest,**fields)
            generation=self.context.probe_state().get('generation')
            if generation:
                event.setdefault('ext',{})['generation']={k:generation[k] for k in ('request_id','generation_step','phase','mode')}
            if mask is not None:event['mask']=mask
            blob=self.config.get('tensor_blob','none')
            selected=blob=='all' or isinstance(blob,list) and any(key.startswith(prefix) for prefix in blob)
            if selected:
                if not self.blob_exhausted and self.metrics['blob_bytes']+len(raw)<=self.config.get('tensor_blob_bytes',1<<30):
                    if self.stream is None:
                        folder=self.context.writer.run_dir/'blobs';folder.mkdir(exist_ok=True)
                        self.stream=(folder/('tensor-'+str(os.getpid())+'-'+self.manager.state['id']+'.bin')).open('xb')
                    offset=self.stream.tell();self.stream.write(raw)
                    event['blob']=dict(path=str(Path(self.stream.name).relative_to(self.context.writer.run_dir)),offset=offset,nbytes=len(raw),hash=digest,
                        fmt='raw-le:'+str(cpu.dtype).removeprefix('torch.')+':'+str(list(cpu.shape)))
                    self.metrics['blob_bytes']+=len(raw)
                else:
                    self.blob_exhausted=True
                    self.metrics['blob_skipped']+=1
                    if self.metrics['first_skipped'] is None:self.metrics['first_skipped']=key
            self.context.writer.emit(event);self.metrics['events']+=1
            return key
        finally:self.busy=busy

    def uninstall(self):
        for handle in reversed(self.handles):handle.remove()
        self.handles.clear()
        if self.mode:self.mode.__exit__(None,None,None);self.mode=None
        if self.stream:self.stream.flush();os.fsync(self.stream.fileno());self.stream.close();self.stream=None
        self.sources.clear();self.parameters.clear();super().uninstall()
