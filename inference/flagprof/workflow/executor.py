# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Execute the compiled argv and follow receipts emitted by the actual producer."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

from flagprof.store.manifest import atomic_json
from flagprof._process import IMPORT_ROOT, module_environment as execution_environment
from flagprof.progress import phase
from .loader import snapshot
from .planner import CONTROL, RUN

def error_detail(stream):
    stream.seek(0, os.SEEK_END)
    stream.seek(max(0, stream.tell() - 16384))
    lines = stream.read().decode(errors='replace').splitlines()
    reasons = [line for line in lines if line and not line[0].isspace() and
               ('Error:' in line or 'Exception:' in line or line.startswith('SystemExit:'))]
    return ' / '.join(dict.fromkeys(reasons[-3:] or lines[-1:]))[:2000]


def reserve(job, automatic):
    job.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            job.mkdir()
        except FileExistsError:
            if automatic:
                job = job.with_name(job.name+'-'+uuid.uuid4().hex[:8])
                continue
            if not job.is_dir() or any(job.iterdir()):
                raise ValueError('任务目录已被使用 ' + str(job))
        # Atomic ownership claim also covers a user-supplied empty directory.
        try:
            (job/'logs').mkdir()
        except FileExistsError as error:
            raise ValueError('任务目录已被使用 ' + str(job)) from error
        return job


def replay_script(job):
    root = shlex.quote(IMPORT_ROOT)
    tool = ' '.join(map(shlex.quote, CONTROL))
    script = ('#!/bin/sh\nset -eu\n'
              'test "$#" -eq 1 || { echo "用法: bash commands.sh <新的任务目录>" >&2; exit 2; }\n'
              'here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)\n'
              'export PYTHONPATH=' + root + '"${PYTHONPATH:+:$PYTHONPATH}"\n'
              'exec ' + tool + ' run "$here/resolved-config.json" --out "$1"\n')
    path = job/'commands.sh'
    path.write_text(script, encoding='utf-8')
    path.chmod(0o755)


def record_receipt(stdout, store, final, cwd):
    # Targets can print arbitrary text before the wrapper's final one-line JSON.
    with stdout.open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - (1<<20)))
        lines = stream.read().splitlines()
    if lines:
        try:
            data = json.loads(lines[-1])
            if isinstance(data,dict) and data.get('status') and (data.get('run') or data.get('capture')):
                return data, Path(cwd), stdout
        except (ValueError, UnicodeError):
            pass
    pattern = 'capture-*/record-status.json' if final else '*/record-status.json'
    receipts = list(store.glob(pattern))
    if len(receipts) != 1:
        raise ValueError('record 缺少唯一的本次回执，详情见 ' + str(stdout))
    path = receipts[0]
    return json.loads(path.read_text()), path.parent, path


def evidence(summary, subject, status, reasons, source, **details):
    summary['evidence'].append(dict(subject=subject,status=status,reasons=reasons,source=str(source),**details))


def artifact(summary, kind, path, **tags):
    if path.exists():
        value = dict(type=kind,path=str(path),**tags)
        if value not in summary['artifacts']:
            summary['artifacts'].append(value)
        return True
    return False



def report_evidence(summary, directory, limits, **tags):
    from flagprof.analyzers.workspace import Limits
    from .receipts import report_rows
    from .report_input import report_info
    result = directory/'result.json'
    for name, path in [('result',result),('report',directory/'report.html')]:
        if not artifact(summary,name,path,**tags):
            raise ValueError('报告缺少 '+str(path))
    budgets=json.loads(Path(limits).read_text()) if isinstance(limits,str) else limits or {}
    doc=report_info(result,record_bytes=Limits(**budgets).record_bytes,details=True)
    summary['evidence'].extend(dict(row,**tags) for row in report_rows(doc,result))
    return doc


def comparison_steps(spec, variants):
    from .comparison import report_metadata
    alternatives=spec.get('if_runner_missing')
    if not alternatives:
        return [spec]
    metadata=[report_metadata(Path(side['report'])) for side in variants.values()]
    if all(m['runner'] for m in metadata):
        return [{k:v for k,v in spec.items() if k!='if_runner_missing'}]
    if all(m['children'] for m in metadata):
        return alternatives
    raise ValueError('性能对照缺少共同的 runner 摘要或同 rank 子报告')


class Execution:
    def __init__(self,config,source,origins,job,plan,progress=None):
        self.config,self.job,self.plan,self.progress=config,job,plan,progress
        self.sides=plan.get('variants') or {config.get('mode','single'):plan}
        self.comparing=config.get('mode')=='compare'
        self.summary=dict(schema='flagprof-workflow-summary/1',mode=config.get('mode','single'),status='failed',exit_code=2,
                          job_dir=str(job),config_source=str(source),value_sources=origins,output_type=plan['output_type'],
                          component_owner=plan['component_owner'],vendor=plan['vendor'],artifacts=[],evidence=[],steps=[],
                          skipped=dict(plan['skipped']))
        if self.comparing:
            self.summary['variants']={r:dict(source=p['source'],output_type=p['output_type'],directory=p['directory'],
                         label=config['compare'][r].get('label',r),vendor=p['vendor']) for r,p in self.sides.items()}
            self.summary['declaration']=plan['declaration']
            self.summary['declaration_source']=plan.get('declaration_source','auto')
        self.locations={}

    def initialize(self,assets):
        atomic_json(self.job/'resolved-config.json',snapshot(self.config))
        replay_script(self.job)
        for role,side in self.sides.items():
            directory=Path(side['directory'])
            directory.mkdir(parents=True,exist_ok=True)
            if 'external' in side:
                artifact(self.summary,side['input_type'],Path(side['external']),variant=role,source=side['source'])
                meta=side.get('report_metadata',{})
                if meta.get('group'):
                    evidence(self.summary,'flagprof-group-report/1',meta['status'],meta['issues'],
                             Path(meta['report'])/'result.json',variant=role)
        for filename,value in assets.items():
            path=Path(filename)
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.suffix=='.jsonl':
                path.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in value),encoding='utf-8')
            elif path.suffix=='.yaml':
                import yaml
                path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding='utf-8')
            else:
                atomic_json(path,value)
            artifact(self.summary,'declaration' if path.name=='declare.json' else 'input',path)

    def record(self,role,side,spec,stdout,code):
        directory=Path(side['directory'])
        store=directory/'store'
        cfg=self.config['compare'][role] if self.comparing else self.config
        receipt,base,source=record_receipt(stdout,store,cfg['collection']['capture_mode']=='final',spec['cwd'])
        found={}
        for key,kind in [('capture','capture'),('run','store')]:
            if receipt.get(key):
                path=(base/receipt[key]).resolve()
                if store not in path.parents or not path.is_dir():
                    raise ValueError('record 回执路径无效 '+str(path))
                if key=='capture' and not (path/'capture-input.json').is_file():
                    kind='capture-attempt'
                artifact(self.summary,kind,path,variant=role)
                artifact(self.summary,'record-status',path/'record-status.json',variant=role)
                if key=='run':
                    found[0]=path
                    self.locations[RUN if role=='single' else f'<{role}.run>']=path
        reasons = [receipt['error']] if receipt.get('error') else []
        reasons.extend(receipt.get('recovery_commands', []))
        evidence(self.summary,'record',receipt['status'],reasons,source,variant=role)
        want='sealed' if side['output_type']=='capture' else 'complete'
        if not code and (receipt['status']!=want or not receipt.get('capture' if want=='sealed' else 'run')):
            raise ValueError('record 回执与预期产物不匹配 '+str(source))
        if not code and want=='sealed':
            evidence(self.summary,'import','deferred',['原始采集已封存'],source,variant=role)
        return found

    def command(self,spec,log):
        role=spec['variant']
        action=spec.get('action',spec['id'])
        argv=spec['argv'].copy()
        # Substitute only FlagPerf-owned input positions, never the target's literal argv.
        if action.startswith('inspect') or action=='report':
            pos=len(CONTROL)+1
            argv[pos]=str(self.locations.get(argv[pos],argv[pos]))
        elif action=='oracle':
            for flag in ('--ref','--cmp','--baseline'):
                if flag in argv:
                    pos=argv.index(flag)+1
                    argv[pos]=str(self.locations.get(argv[pos],argv[pos]))
            for flag in ('--replay-ref-run', '--replay-cmp-run'):
                if flag in argv:
                    pos = argv.index(flag) + 1
                    argv[pos] = str(self.locations[argv[pos]])
        if action=='report' and self.sides[role]['attach_runner']:
            missing=[i for i,p in enumerate(self.sides[role]['runners']) if not Path(p).is_file()]
            if missing:
                reason=f'{role} 缺少 rank {missing} 的 runner.json'
                evidence(self.summary,'runner.attachment','unavailable',[reason],self.sides[role]['directory'],variant=role)
                if self.config['output']['attach_runner_results'] is True:
                    raise ValueError('显式挂接失败 '+reason)
                pos=argv.index('--runner-results')
                del argv[pos:pos+1+len(self.sides[role]['runners'])]
        errors=log.with_suffix('.stderr')
        step=dict(spec,argv=argv,log=str(log),stderr=str(errors),exit_code=None)
        self.summary['steps'].append(step)
        if self.progress:
            self.progress(f"[运行] {len(self.summary['steps'])}. {spec['id']}  {log}")
        started=time.monotonic()
        try:
            with log.open('wb') as output,errors.open('wb') as err:
                try:
                    proc=subprocess.run(argv,cwd=spec['cwd'],env=execution_environment(),stdout=output,stderr=err)
                    code=proc.returncode if proc.returncode>=0 else 128-proc.returncode
                except KeyboardInterrupt:
                    code=130
            step['exit_code']=code
        finally:
            step['elapsed_s']=round(time.monotonic()-started,6)
        tags={'variant':role}
        if 'rank' in spec:
            tags['rank']=spec['rank']
        stores={}
        side=self.sides.get(role,{})
        if action in ('replay.prepare', 'replay.run') and (not code or log.stat().st_size):
            from .replay import consume_receipt
            try:
                consume_receipt(self, spec, log)
            except (ValueError, OSError, KeyError, TypeError) as error:
                step['receipt_error'] = str(error)
                if not code:
                    step['error'] = str(error)
                    raise
        if action in ('export','batch'):
            from .suite_receipts import graph_receipt, batch_receipt
            try:
                if action == 'export':
                    graph_receipt(self.summary, self.job, self.config, code)
                else:
                    batch_receipt(self.summary, self.job, self.plan, code)
            except (OSError,ValueError,KeyError,TypeError) as error:
                if not code:
                    raise
                step['receipt_error'] = str(error)
        if action in ('record','launch'):
            try:
                if action=='record':
                    stores=self.record(role,side,spec,log,code)
                else:
                    from .receipts import launch_result
                    group,stores,items=launch_result(side,code)
                    self.summary['artifacts'].extend(dict(a,variant=role) for a in items)
                    evidence(self.summary,'launch',group['status'],[group['error']] if group.get('error') else [],
                             items[0]['path'],variant=role,backend=group['backend'],group_id=group['group_id'])
                    for row in group['ranks']:
                        rank=row['rank']
                        rank_log=Path(side['directory'])/'store'/f'rank-{rank}.log'
                        artifact(self.summary,'rank-log',rank_log,variant=role,rank=rank)
                        evidence(self.summary,'launch.rank',row['status'],[row['error']] if row.get('error') else [],
                                 items[0]['path'],variant=role,rank=rank,returncode=row.get('returncode'),
                                 termination_reason=row.get('termination_reason'))
                    for rank,run in stores.items():
                        self.locations[f'<{role}.rank-{rank}.run>']=run
                    if group['status']=='captured':
                        evidence(self.summary,'import','deferred',['逐 rank 原始采集已封存'],items[0]['path'],variant=role)
            except (ValueError,OSError,KeyError) as error:
                if not code:
                    raise
                step['receipt_error']=str(error)
        if code:
            reasons = [reason for row in self.summary['evidence']
                       if row.get('variant') == role and row['subject'] in ('record', 'launch.rank', 'replay.run')
                       and row['status'] not in ('ok','complete') for reason in row['reasons']]
            if not reasons:
                with errors.open('rb') as stream:
                    detail = error_detail(stream)
                if detail:
                    reasons.append(detail)
            step['error'] = self.summary['error'] = '；'.join([f"{spec['id']} 返回 {code}", *reasons, str(errors)])
            return code
        if action in ('record','launch','runner'):
            if action=='runner' and not Path(side['runners'][0]).is_file():
                raise ValueError('内置 runner 未生成 '+side['runners'][0])
            if (self.comparing or action=='launch') and side['output_type'] not in ('capture','capture-group','execution'):
                from .receipts import component_evidence
                cfg=self.config['compare'][role] if self.comparing else self.config
                rows=component_evidence(cfg,side,role,stores,required=self.comparing)
                self.summary['evidence'].extend(dict(row,variant=role) for row in rows)
                problems=[reason for row in rows for reason in row['reasons']]
                if problems:
                    raise ValueError(' / '.join(problems))
        elif action=='make-package':
            if not artifact(self.summary,'package',Path(side['directory'])/'inputs/ctr-package/manifest.json',**tags):
                raise ValueError('自动输入包缺少 manifest.json')
        elif action.startswith('inspect'):
            inspection=json.loads(log.read_text())
            evidence(self.summary,'inspect',inspection['run']['manifest_status'],[],log,
                     run_state=inspection['run']['state'],traces=len(inspection['traces']),
                     verified=sum(t.get('integrity')=='verified' for t in inspection['traces']),**tags)
            artifact(self.summary,'inspection',log,**tags)
        elif action in ('report','compare','oracle'):
            directory = Path(argv[argv.index('--out')+1])
            doc = report_evidence(self.summary, directory, self.config['output'].get('limits'), **tags)
            if self.config.get('mode') == 'replay' and doc['results'].get('oracle.replay', {}).get('status') != 'ok':
                raise ValueError('Oracle 报告缺少有效的同输入重放结果 ' + str(directory/'result.json'))
        return 0

    def run(self,assets):
        total = len(self.plan['steps'])
        with phase('工作流', total=total, unit='步') as workflow:
            summary = self._run(assets, workflow, total)
            workflow.finish(summary['exit_code'])
            return summary

    def _run(self,assets,workflow,total):
        summary=self.summary
        code=0
        completed = 0
        try:
            self.initialize(assets)
            for spec in self.plan['steps']:
                actual = comparison_steps(spec, self.sides) if spec.get('action') == 'compare' else [spec]
                if len(actual) != 1:
                    total += len(actual) - 1
                    workflow.update(completed, total=total)
                for step in actual:
                    log = self.job/'logs'/f"{completed + 1:02d}-{step['id']}.stdout"
                    with phase(step['id'], variant=step['variant'], rank=step.get('rank'),
                               detail=str(log)) as current:
                        code = self.command(step, log)
                        current.finish(code)
                    if code:
                        break
                    completed += 1
                    if completed < total:
                        workflow.update(completed)
                if code:
                    break
        except KeyboardInterrupt:
            code=130
            summary['error']='用户中断了工作流'
        except (ValueError,OSError,KeyError,TypeError,RuntimeError) as error:
            code=code or 2
            summary['error']=str(error)
        done={s['id'] for s in summary['steps']}
        if code:
            summary['skipped'].update({s['id']:'前序步骤失败' for s in self.plan['steps'] if s['id'] not in done})
        for role,side in self.sides.items():
            for rank,filename in enumerate(side['runners']):
                artifact(summary,'runner',Path(filename),rank=rank,variant=role)
            out=Path(side['directory'])/'report'
            for kind,name in [('report','report.html'),('result','result.json')]:
                artifact(summary,kind,out/name,variant=role)
        for kind,path in [('config',self.job/'resolved-config.json'),('replay',self.job/'commands.sh')]:
            artifact(summary,kind,path)
        summary.update(status='failed' if code else 'complete',exit_code=code)
        atomic_json(self.job/'summary.json',summary)
        if not code:
            workflow.update(completed)
        return summary


def execute(config,source,origins,job,plan,assets,*,progress=None):
    return Execution(config,source,origins,job,plan,progress).run(assets)
