"""Generic signatures for operator inventory and optional route evidence."""
import torch

def signature(value):
    if isinstance(value, torch.Tensor):
        return {'shape':list(value.shape), 'dtype':str(value.dtype), 'device':str(value.device),
                'stride':list(value.stride())}
    if isinstance(value,(list,tuple)):
        return [signature(v) for v in value]
    if isinstance(value,dict):
        return {str(k):signature(v) for k,v in value.items()}
    return value if isinstance(value,(int,float,str,bool,type(None))) else str(value)
