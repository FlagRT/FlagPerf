"""Untimed component dispatch evidence; never equated with hardware coverage."""
from collections import Counter
from pathlib import Path
import os
from runtime.common import write_json
from vendors.stack import selection, compiler_identity, communication_backend


class CompilerAudit:
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
        return self

    def __exit__(self,*exc):
        for restore in reversed(self.restore): restore()
        self.result = dict(self.compiler,
            jit_run_calls=self.calls['jit_run'], jit_warmup_calls=self.calls['jit_warmup'],
            launch_hook_calls=self.calls['launch_hook'] if self.launch_hook_available else None,
            observation='observed' if self.calls['jit_run'] else 'not_observed',
            evidence_level='Python JIT dispatch and optional launch hook; not hardware kernel coverage')
