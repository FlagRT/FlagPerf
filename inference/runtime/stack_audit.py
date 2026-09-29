# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Untimed component dispatch evidence; never equated with hardware coverage."""
from collections import Counter
from pathlib import Path
import os
from runtime.common import write_json
from vendors.stack import selection, compiler_identity, communication_backend


class StackAudit:
    def __init__(self,cfg,root,compiler=None):
        self.cfg,self.root=cfg,root
        self.checked_compiler=compiler
        self.calls=Counter(); self.collectives=[]; self.restore=[]

    def __enter__(self):
        import triton
        import torch.distributed as dist
        from triton.runtime.jit import JITFunction
        if self.checked_compiler is None:
            self.compiler=compiler_identity(self.cfg)
        else:
            expected=self.checked_compiler
            provider='flagtree' if selection(self.cfg)['flagtree']=='on' else 'vendor'
            if (str(Path(triton.__file__).resolve()) != expected['module_path'] or
                    triton.__version__ != expected['import_version'] or provider != expected['provider']):
                raise ValueError('compiler import changed after worker identity check')
            self.compiler=dict(expected)
        self.cache_groups={'lookups':0,'complete_reads':0,'writes':0}
        if self.cfg.get('_preview_trial'):
            from triton.runtime.cache import FileCacheManager
            owner=self
            get_group=FileCacheManager.get_group
            put_group=FileCacheManager.put_group
            def observed_get(manager,*args,**kwargs):
                value=get_group(manager,*args,**kwargs)
                owner.cache_groups['lookups']+=1
                if value:
                    try:
                        from runtime.common import read_json
                        filename=args[0] if args else kwargs['filename']
                        children=read_json(Path(manager.cache_dir)/('__grp__'+filename))['child_paths']
                        if isinstance(children,dict) and children == value and all(os.path.isfile(p) for p in value.values()):
                            owner.cache_groups['complete_reads']+=1
                    except (OSError,ValueError,KeyError,AttributeError):
                        pass
                return value
            def observed_put(manager,*args,**kwargs):
                value=put_group(manager,*args,**kwargs)
                owner.cache_groups['writes']+=1
                return value
            FileCacheManager.get_group=observed_get
            FileCacheManager.put_group=observed_put
            self.restore.extend([lambda:setattr(FileCacheManager,'get_group',get_group),
                                 lambda:setattr(FileCacheManager,'put_group',put_group)])
        original=JITFunction.run
        owner=self
        def run(kernel,*args,**kwargs):
            value=original(kernel,*args,**kwargs)
            owner.calls['jit_warmup' if kwargs.get('warmup',False) else 'jit_run']+=1
            return value
        JITFunction.run=run
        self.restore.append(lambda:setattr(JITFunction,'run',original))
        runtime=getattr(getattr(triton,'knobs',None),'runtime',None)
        self.launch_hook_available=runtime is not None and hasattr(runtime,'launch_enter_hook')
        if self.launch_hook_available:
            previous=runtime.launch_enter_hook
            def launch(*args,**kwargs):
                owner.calls['launch_hook']+=1
                if previous: return previous(*args,**kwargs)
            runtime.launch_enter_hook=launch
            self.restore.append(lambda:setattr(runtime,'launch_enter_hook',previous))
        for name in ('all_reduce','all_gather','all_gather_into_tensor','reduce_scatter_tensor'):
            native=getattr(dist,name)
            def observed(*args,_name=name,_native=native,**kwargs):
                # Known public signatures: all_reduce uses group position 2; gather/scatter position 2.
                group=kwargs.get('group',args[2] if len(args)>2 and _name!='reduce_scatter_tensor' else None)
                if _name=='reduce_scatter_tensor': group=kwargs.get('group',args[3] if len(args)>3 else None)
                backend=dist.get_backend(group)
                tensor=args[0] if args else kwargs.get('tensor')
                row={'collective':_name,'backend':backend}
                if hasattr(tensor,'numel'): row.update(shape=list(tensor.shape),dtype=str(tensor.dtype),logical_bytes=tensor.numel()*tensor.element_size())
                value=_native(*args,**kwargs)
                owner.collectives.append(row)
                return value
            setattr(dist,name,observed)
            self.restore.append(lambda name=name,native=native:setattr(dist,name,native))
        return self

    def __exit__(self,*exc):
        for restore in reversed(self.restore): restore()
        requested=selection(self.cfg)
        tp=self.cfg['runtime'].get('parallelism')=='tp'
        expected=communication_backend(self.cfg) if tp else None
        mismatch=any(c['backend']!=expected for c in self.collectives)
        self.result={'requested':requested,'compiler':dict(self.compiler,
            jit_run_calls=self.calls['jit_run'],jit_warmup_calls=self.calls['jit_warmup'],
            launch_hook_calls=self.calls['launch_hook'] if self.launch_hook_available else None,
            observation='observed' if self.calls['jit_run'] else 'not_observed',
            evidence_level='Python JIT dispatch and optional launch hook; not hardware kernel coverage'),
            'communication':{'expected_backend':expected,'observed_backends':sorted({c['backend'] for c in self.collectives}),
                'status':'not_applicable' if not tp else ('mismatch' if mismatch else 'observed' if self.collectives else 'not_observed'),
                'model_collective_calls':len(self.collectives),'collectives':self.collectives,
                'scope':'model forward only; control-group synchronization excluded'}}
        if self.cfg.get('_preview_trial'):
            self.result['cache_groups']=dict(self.cache_groups,
                scope='Python file cache group reads/writes; not hardware kernel coverage')
        write_json(self.root/'components.json',self.result)
        if mismatch and exc[0] is None: raise ValueError('model collective used an unexpected backend')
