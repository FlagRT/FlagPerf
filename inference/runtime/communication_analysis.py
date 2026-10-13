"""Offline aggregation of vendor-decoded communication evidence."""
from pathlib import Path
from copy import deepcopy
from runtime.common import write_json
from runtime.trace_utils import ns, union_ns, overlap_ns
from runtime.trace_dispatch import decoder


def flatten_communication(data):
    from vendors.ascend.communication import flatten_communication as impl
    return impl(data)


def matrix_totals(data, rank):
    from vendors.ascend.communication import matrix_totals as impl
    return impl(data, rank)


def analyze_rank(folder, rank, *, timeout=900):
    selected = decoder(folder)
    if selected.__name__ == 'vendors.nvidia.trace':
        from runtime.cuda_analysis_job import run
        return run(folder, rank=rank, communication_only=True, timeout=timeout)
    return selected.communication(Path(folder), rank)


def analyze(root,world_size, *, rank_results=None, timeout=900):
    root=Path(root);ranks={}
    for rank in range(world_size):
        try:
            ranks[str(rank)] = (deepcopy(rank_results[str(rank)]) if rank_results is not None
                                else analyze_rank(root/f'rank-{rank}',rank,timeout=timeout))
        except Exception as error:ranks[str(rank)]={'status':'partial','rank':rank,'issues':[str(error)]}
    rank_skew=[]
    if all(r['status']=='completed' for r in ranks.values()):
        lists=[r['events'] for r in ranks.values()]
        if len({len(v) for v in lists})!=1:
            ranks['0']['status']='partial';ranks['0']['issues'].append('rank collective counts differ')
        else:
            for rows in zip(*lists):
                keys=['sequence','cycle','batch_index','module','logical_bytes','collective','phase','generation_step','kv_length_before']
                if any(any(r.get(k)!=rows[0].get(k) for k in keys) for r in rows):
                    ranks['0']['status']='partial';ranks['0']['issues'].append('rank collective pairing differs');break
                starts=[r['host_api_start_ns'] for r in rows]
                rank_skew.append({'sequence':rows[0]['sequence'],'module':rows[0]['module'],
                                  'host_entry_skew_ns':max(starts)-min(starts),
                                  'scope':'same-host trace CPU markers; diagnostic, not causal proof'})
    result={'status':'completed' if all(r['status']=='completed' for r in ranks.values()) else 'partial',
            'rank_entry_skew':rank_skew,
            'ranks':ranks,'scope':'independent profiled replay; never added to unprofiled latency',
            'aggregation_rules':{'logical_payload':'per rank; replicated TP inputs counted once for model throughput',
                'link_volume':'source-owned directed matrix total rows; LOCAL separate; do not add bandwidth aliases',
                'time':'wait/synchronization overlap; do not add them or sum ranks as wall latency',
                'attribution':'vendor decoder with explicit correlation, payload and coverage checks; timing sources retain vendor semantics'}}
    write_json(root/'communication.json',result)
    return result
