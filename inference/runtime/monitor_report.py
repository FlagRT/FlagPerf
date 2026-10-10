"""Portable, offline telemetry panels; raw evidence is never rewritten."""
from collections import defaultdict
import html
import json
from pathlib import Path

from runtime.common import read_json, write_json


def safe_render(root, result):
    if result.get('command') != 'performance': return
    try:
        render(root, result)
        status = {'status': 'completed' if (Path(root)/'monitor.html').is_file() else 'not_collected'}
    except Exception as error:
        # A telemetry visualization error must not fail performance analysis.
        import sys
        print('Monitor report unavailable: '+str(error), file=sys.stderr)
        status = {'status': 'failed', 'error': str(error)}
    try:
        write_json(Path(root)/'monitor-report-status.json', status)
    except OSError:
        pass


def render(root, result):
    if result.get('command') != 'performance': return
    root = Path(root)
    state = result.get('monitoring', {'status': 'not_collected'})
    summary_file = root/'monitor/summary.json'
    lines = ['<!-- monitor:start -->', '## 设备监控旁路', '',
             f"监控状态：**{state.get('status', 'not_collected')}**。监控状态独立于性能执行状态。", '']
    if summary_file.is_file() and (root/'monitor/inventory.json').is_file() and (root/'monitor/samples.jsonl').is_file():
        state = read_json(summary_file)
        from runtime.monitor_analysis import timeline_rows
        inventory, samples = timeline_rows(root, state)
        grouped = defaultdict(list)
        aligned_phases = state['phases'] if state['clock_aligned'] else []
        origin = min([r['read_start_ns'] for r in samples]+[p['start_ns'] for p in aligned_phases]+[inventory['clock_anchor_ns']])
        for row in samples:
            grouped[(row['entity'], row['metric'])].append([
                (row['read_end_ns']-origin)/1e9, row['value'] if row['status']=='ok' else None, row.get('reason')])
        definitions = {m['name']:m for m in inventory['metrics']}
        panels = [dict(entity=e, metric=m, unit=definitions[m]['unit'], points=points,
                       resolution='源刷新周期未知',
                       ranks=inventory['entities'][e].get('ranks', [])) for (e,m),points in sorted(grouped.items())]
        phases = [dict(p, start=(p['start_ns']-origin)/1e9, end=(p['end_ns']-origin)/1e9) for p in aligned_phases]
        batches = []
        # Worker files are retained in ordinary compact results. Profiler batches are excluded.
        for worker in sorted({p['worker'] for p in phases if p['action']=='performance'}):
            path = root/worker/'batches.jsonl'
            if not path.is_file(): continue
            for line in path.read_text().splitlines():
                try: batch = json.loads(line)
                except ValueError: continue
                if 'start_ns' in batch and 'end_ns' in batch:
                    batches.append(dict(worker=worker, start=(batch['start_ns']-origin)/1e9,
                        end=(batch['end_ns']-origin)/1e9, latency_ms=batch['latency_ns']/1e6,
                        cycle=batch['cycle'], batch_index=batch['batch_index']))
        for worker in sorted({b['worker'] for b in batches}):
            rank = next(p['rank'] for p in phases if p['worker']==worker)
            panels.append(dict(entity='batch:'+worker, metric='batch.latency', unit='ms',
                ranks=[rank], resolution='原始同步主机计时',
                points=[[b['end'],b['latency_ms'],None] for b in batches if b['worker']==worker]))
        payload = json.dumps(dict(panels=panels, phases=phases, batches=batches,
                                  interval_ms=state['interval_ms']), ensure_ascii=False).replace('<', '\\u003c')
        title = html.escape(str(state['status']))
        page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Inference Monitor</title>
<style>body{font:15px system-ui;margin:24px;background:#f5f7fa;color:#182635}section{background:white;padding:16px;margin:12px 0;border-radius:8px}svg{width:100%;height:180px}label{margin-right:20px}select,input{margin:8px}small{color:#596775}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ddd}</style>
<h1>设备监控时间轴</h1><p>状态：STATUS。各面板共享时间轴；缺失数据断线显示。秒级遥测不能归因到单个短 batch。</p>
<p>功率域可能包含未选中的兄弟芯片；设备内存与框架 allocated/reserved 分开解释。曲线只用于运行趋势，不表示峰值算力达成率。</p>
<label>实体<select id="entity"><option value="">全部</option></select></label>
<label>阶段 / worker<select id="phase"><option value="">完整运行</option></select></label>
<label>起点(s)<input id="lo" type="number" step="0.1"></label><label>终点(s)<input id="hi" type="number" step="0.1"></label>
<p><a href="monitor/summary.json">统计与质量</a> · <a href="monitor/inventory.json">指标定义和能力</a> · <a href="monitor/samples.jsonl">完整原始样本</a></p>
<div id="charts"></div><h2>阶段与批次窗口</h2><div id="details"></div>
<script>const D=PAYLOAD;const $=id=>document.getElementById(id);const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const end=D.panels.reduce((v,p)=>p.points.reduce((v,x)=>Math.max(v,x[0]),v),D.phases.reduce((v,p)=>Math.max(v,p.end),1));$('lo').value=0;$('hi').value=end.toFixed(3);
[...new Set(D.panels.map(p=>p.entity))].forEach(e=>$('entity').add(new Option(e,e)));
D.phases.forEach((p,i)=>$('phase').add(new Option(`${p.worker} · ${p.stage} · ${p.start.toFixed(2)}–${p.end.toFixed(2)}s`,i)));
$('phase').onchange=()=>{const p=D.phases[$('phase').value];$('lo').value=p?p.start:0;$('hi').value=p?p.end:end;draw()};
for(const id of ['entity','lo','hi'])$(id).onchange=draw;
function draw(){let lo=Number($('lo').value),hi=Number($('hi').value);if(!(hi>lo))return;let out='';
const phase=D.phases[$('phase').value];
for(const p of D.panels){if($('entity').value&&p.entity!==$('entity').value)continue;if(phase&&p.ranks.length&&!p.ranks.includes(phase.rank))continue;
const pts=p.points.filter(v=>v[0]>=lo&&v[0]<=hi),vs=pts.filter(v=>v[1]!==null).map(v=>v[1]);let low=vs.reduce((v,x)=>Math.min(v,x),0),high=vs.reduce((v,x)=>Math.max(v,x),1);const x=t=>55+880*(t-lo)/(hi-lo),y=v=>145-115*(v-low)/(high-low);let path='',last=null;
for(const v of pts){if(v[1]===null){last=null;continue;}const gap=last!==null&&v[0]-last>2.5*D.interval_ms/1000;path+=(last===null||gap?'M':'L')+x(v[0])+','+y(v[1])+' ';last=v[0];}
let bands='';for(const s of D.phases){if(s.end<lo||s.start>hi||p.ranks.length&&!p.ranks.includes(s.rank))continue;bands+=`<rect x="${x(Math.max(lo,s.start))}" y="20" width="${Math.max(0,x(Math.min(hi,s.end))-x(Math.max(lo,s.start)))}" height="130" fill="${s.action==='profile'||s.action==='layer_profile'?'#ffe2c6':s.stage==='measurement'?'#e0f1ea':'#e9edf5'}" opacity=".5"><title>${esc(s.worker+' '+s.stage)}</title></rect>`;}
out+=`<section><b>${esc(p.entity+' · '+p.metric+' ('+p.unit+')')}</b><small> 有效点 ${vs.length}/${pts.length}；${esc(p.resolution)}</small><svg viewBox="0 0 960 180">${bands}<text x="2" y="30">${high.toPrecision(4)}</text><text x="2" y="145">${low.toPrecision(4)}</text><path d="${path}" fill="none" stroke="#087e8b" stroke-width="2"/>${pts.filter(v=>v[1]!==null).map(v=>`<circle cx="${x(v[0])}" cy="${y(v[1])}" r="2" fill="#087e8b"><title>${v[0].toFixed(3)}s: ${v[1]}</title></circle>`).join('')}<text x="55" y="172">${lo.toFixed(3)}s</text><text x="870" y="172">${hi.toFixed(3)}s</text></svg></section>`;}
$('charts').innerHTML=out;const b=D.batches.filter(b=>b.end>=lo&&b.start<=hi&&(!phase||b.worker===phase.worker));$('details').innerHTML='<p>绿色：正式测量阶段；橙色：profiler worker（保守屏蔽冲突利用率）；其他：加载/预热/审计等。阶段墙钟包含批次间隙。</p><table><tr><th>worker</th><th>cycle / batch</th><th>起止(s)</th><th>主计时(ms)</th></tr>'+b.map(v=>`<tr><td>${esc(v.worker)}</td><td>${v.cycle} / ${v.batch_index}</td><td>${v.start.toFixed(4)}–${v.end.toFixed(4)}</td><td>${v.latency_ms.toFixed(3)}</td></tr>`).join('')+'</table>';}
draw();</script></html>'''.replace('STATUS', title).replace('PAYLOAD', payload)
        (root/'monitor.html').write_text(page)
        lines += [f"采样请求间隔：{state['interval_ms']} ms；有效性及时间对齐详见摘要。采样周期不等于硬件分辨率。", '',
                  '[交互时间轴](monitor.html) · [质量与分阶段统计](monitor/summary.json) · '
                  '[设备和指标定义](monitor/inventory.json) · [原始样本](monitor/samples.jsonl) · [证据校验](monitor/manifest.json)', '',
                  '阶段墙钟包含批次间隙；统计是有效采样点统计，不与累计 batch latency 混用。'
                  'Profiler 冲突值不可解释成设备空闲；无效点和采样空洞断线显示。', '']
        lines += ['- '+str(error) for error in state.get('errors', [])]
    else:
        lines += ['本次连续监控证据未采集；离线重绘仅展示已经保存的遥测。', '']
    lines.append('<!-- monitor:end -->')
    report = root/'report.md'
    if report.is_file():
        text = report.read_text()
        if '<!-- monitor:start -->' in text:
            start = text.index('<!-- monitor:start -->')
            finish = text.index('<!-- monitor:end -->', start)+len('<!-- monitor:end -->')
            text = text[:start]+text[finish:]
        report.write_text(text.rstrip()+'\n\n'+'\n'.join(lines)+'\n')
