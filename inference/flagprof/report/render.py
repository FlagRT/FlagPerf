# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Offline HTML and Markdown from the same disk-backed presentation model."""
from html import escape
from heapq import nsmallest
import itertools
import json
import hashlib
import re
from pathlib import Path
from urllib.parse import urlencode

from flagprof.progress import phase
from .fields import LABELS, VALUES, REASON_TEXT, PANEL_LABELS, MISSING_TEXT, RAW_CLOCK_FIELDS, cell, display, field_text
from .fields import value_field
from .layout import MODES, decorate, excerpts


def safe_json(value):
    return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e').replace('\u2028','\\u2028').replace('\u2029','\\u2029')


def reader_text(value, field=''):
    """Present local file references by basename in portable reports."""
    value=str(value)
    if field in ('sample','samples','model','call_key'):
        return value
    for code,label in REASON_TEXT.items():
        value=value.replace(code,label)
    return re.sub(r"""(?<![\w/])/(?:[^/\s，；。：、'"<>]+/)+([^/\s，；。：、'"<>]+)""", r"\1", str(value))


def _web_value(value, field=''):
    if isinstance(value,str):
        return reader_text(value,field)
    if isinstance(value,int) and not isinstance(value,bool) and abs(value)>9007199254740991:
        return str(value)
    if isinstance(value,dict):
        return {k:_web_value(v,value_field(value,k)) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_web_value(v,field) for v in value]
    return value


def records_json(rows):
    """Share repeated object keys within one bounded, self-describing batch."""
    layouts = {}
    def pack(value):
        if isinstance(value,dict):
            keys = tuple(value)
            index = layouts.setdefault(keys,len(layouts))
            return [index]+[pack(item) for item in value.values()]
        if isinstance(value,list):
            return [None]+[pack(item) for item in value]
        return value
    packed = [pack(row) for row in rows]
    return safe_json(dict(keys=list(layouts),rows=packed))


def _row(stream, section, index, row):
    attention = section['id']=='oracle.identity_diff' and row.get('equal') is False and (
        row.get('requirement')=='must_equal' or not row.get('declared'))
    stream.write('<tr data-row="'+str(index)+'"'+(' class="condition-blocked"' if attention else '')+'>')
    for key in section['columns']:
        value=cell(row,key)
        shown=reader_text(field_text(row,key),value_field(row,key))
        brief=shown if len(shown)<=140 else shown[:137]+'…'
        content = escape(brief)
        if key == 'first_observed':
            content = '<a href="#oracle.first_divergence">'+escape(value)+'</a>' if value else ''
        if key == 'report' and isinstance(value,str) and not value.startswith('/') and ':' not in value and '..' not in Path(value).parts:
            content = '<a href="'+escape(value,quote=True)+'">打开 rank 结果</a>'
        stream.write('<td data-panel="'+escape(section['id'],quote=True)+'" data-row="'+str(index)+
                     '" data-field="'+escape(key,quote=True)+'" data-unit="'+('ns' if key.endswith('_ns') else '')+
                     '"'+(' title="'+escape(shown,quote=True)+'"' if brief!=shown else '')+'>'+content+'</td>')
    stream.write('<td><button class="detail" data-section="'+escape(section['id'],quote=True)+
                 '" data-index="'+str(index)+'">查看详情与证据</button>')
    if section['id']=='oracle.op_sensitivity' and row.get('replay_status')=='replay_ready':
        stream.write('<button class="replay" data-index="'+str(index)+'">复制回放命令</button>')
    stream.write('</td></tr>')


def evidence_links(model, row):
    links=[]
    keys=[row.get('call_key')]+list(row.get('input_sources_ref') or [])+list(row.get('input_sources_cmp') or [])
    for key in dict.fromkeys(k for k in keys if isinstance(k,str) and k!='unknown'):
        for title,href in model.db.execute('select title,href from evidence_lookup where key=? order by href limit 64',(key,)):
            links.append(dict(label=('当前调用：' if key==row.get('call_key') else '输入生产者：')+key+' · '+title,href=href))
    return dict(row,_evidence_links=links) if links else row


def finding_href(finding, filename):
    query = ''
    if finding.get('row_index') is not None:
        query = '?'+urlencode(dict(section=finding['section'], row=finding['row_index']))
    return filename+query+'#'+finding['section']


@phase('渲染 HTML 和 Markdown 报告')
def render(model, out, *, filename='report.html'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assets=Path(__file__).parent/'assets'
    meta=model.meta
    if 'mode' not in meta:
        decorate(model)
    sections=[dict(s,count=len(model.rows(s['id']))) for s in model.sections]
    model.db.execute('create temp table evidence_lookup(key text,title text,href text)')
    for section in sections:
        if section.get('kind')!='evidence':continue
        stem=section['id'].replace('.','-')
        for ordinal,row in enumerate(model.rows(section['id'])):
            if row.get('call_key'):
                href='evidence/'+stem+'-'+str(ordinal//500+1)+'.html#row-'+str(ordinal)
                model.db.execute('insert into evidence_lookup values(?,?,?)',(row['call_key'],section['title'],href))
    model.db.execute('create index evidence_key on evidence_lookup(key)')
    for section in sections:
        if section.get('kind')=='evidence':
            section['evidence_index']=evidence_pages(model,out,section,filename)
    fingerprint=hashlib.sha256()
    for name in ('render.py','fields.py','product.py','layout.py','curves.py','diagnosis.py','oracle_view.py','assets/report.css','assets/report.js','assets/workspaces.js','assets/charts.js','assets/monitor.js','assets/resource.js'):
        fingerprint.update((Path(__file__).parent/name).read_bytes())
    (out/'renderer.json').write_text(json.dumps(dict(schema='report-renderer/1',sha256=fingerprint.hexdigest(),
        presentation_schema=meta['schema']),indent=2)+'\n')
    with (out/filename).open('w',encoding='utf-8') as stream:
        stream.write('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">')
        stream.write('<title>'+escape(meta['title'])+' · FlagProf</title><style>'+ (assets/'report.css').read_text()+'</style></head><body>')
        stream.write('<a class="skip" href="#overview">跳到结果概览</a><aside class="sidebar"><div class="brand">FlagProf<span>推理结果报告</span></div><nav aria-label="报告目录"><a href="#overview">结果概览</a>')
        for _,members in itertools.groupby(sections,key=lambda s:s['group']):
            section=next(members)
            stream.write('<a href="#'+escape(section['id'],quote=True)+'">'+escape(section['group_title'])+'<small>'+str(section['count'])+'</small></a>')
        stream.write('<a href="#availability">观测范围与未提供内容</a><a href="#sources">来源与导出</a></nav></aside><main>')
        stream.write('<header id="overview"><h1><small>'+MODES[meta['mode']]+'</small> '+escape(meta['title'])+'</h1>')
        overview = meta.get('oracle_overview')
        if overview:
            stream.write('<article class="oracle-verdict"><h2>'+escape(overview['title'])+'</h2><p>'+
                         escape(overview['detail'])+'</p><a href="#'+escape(overview['section'])+'">查看判断依据</a></article>')
        for notice in meta['banners'][:1]:
            stream.write('<aside class="fault-banner"><strong>'+escape(notice['title'])+'</strong> '+escape(reader_text(notice['detail'])))
            if notice.get('section')=='diagnosis':
                stream.write(' <a href="#diagnosis">查看各任务的事件</a>')
            stream.write('</aside>')
        if meta['context_summary']:
            stream.write('<div class="context">')
            priority=('输入内容摘要','精度对照身份','模型','模型文件身份','设备','精度','样本标识','FlagGems','并行配置','通信后端','采集窗口与配置')
            front=sorted(meta['context_summary'],key=lambda f:(not f.get('different'),priority.index(f['label']) if f['label'] in priority else len(priority)))
            for fact in front[:7]:
                value=fact.get('value',fact.get('left','')+(' → '+fact['right'] if fact.get('different') else ''))
                stream.write('<span class="'+('changed' if fact.get('different') else '')+'">'+escape(fact['label'])+' <b>'+escape(reader_text(value,fact.get('field','')))+'</b></span>')
            stream.write('</div>')
        if meta['mode']=='group':
            stream.write('<div class="rank-cards">')
            for row in model.rows('group.ranks'):
                href=row.get('report','')
                content='<b>rank '+escape(str(row['rank']))+'</b> '+escape(row.get('execution_summary','查看 rank 报告'))
                if href and not href.startswith('/') and ':' not in href and '..' not in Path(href).parts:
                    stream.write('<a href="'+escape(href,quote=True)+'">'+content+'</a>')
                else:
                    stream.write('<span>'+content+'</span>')
            stream.write('</div>')
        if meta['facts']:
            stream.write('<details class="identity"><summary>本次任务与比较条件（展开核对）</summary><dl>')
            for fact in meta['facts']:
                value = fact['value']
                field = fact.get('field','')
                if isinstance(value,dict) and 'ref' in value and 'cmp' in value:
                    shown = '参考侧 '+display(value['ref'], field)+' · 候选侧 '+display(value['cmp'], field)
                else:
                    shown = display(value, field)
                stream.write('<div><dt>'+escape(fact['label'])+'</dt><dd>'+escape(reader_text(shown,field))+'</dd></div>')
            stream.write('</dl></details>')
        if meta['credibility']:
            stream.write('<div class="credibility">')
            for fact in meta['credibility']:
                stream.write('<a href="#'+escape(fact.get('section') or 'availability')+'" title="'+escape(reader_text(fact['detail']))+'">'+escape(reader_text(fact['label']))+'</a>')
            stream.write('</div>')
        stream.write('<div class="findings">')
        for pos,finding in enumerate(meta['kpis']):
            if pos==3:
                stream.write('</div><details class="more-findings"><summary>其余 '+str(len(meta['kpis'])-3)+' 项发现</summary><div class="findings">')
            stream.write('<article class="finding '+escape(finding['level'],quote=True)+'"><small>'+escape(finding['eyebrow'])+'</small><h2>'+escape(finding['title'])+'</h2><strong>'+escape(finding['metric_value'])+'</strong><p>'+escape(reader_text(finding['detail']))+'</p>')
            if finding.get('section') and any(s['id']==finding['section'] for s in sections):
                focus = ''
                if finding.get('row_index') is not None:
                    focus = ' data-focus-section="'+escape(finding['section'],quote=True)+'" data-focus-row="'+str(finding['row_index'])+'"'
                stream.write('<a href="'+escape(finding_href(finding,filename),quote=True)+'"'+focus+'>查看对应结果 →</a>')
            stream.write('</article>')
        if not meta['findings'] and not overview:
            stream.write('<article class="finding"><h2>查看本次可用结果</h2><p>下方按功能整理已保存的观测；缺少的记录及原因在观测范围中说明。</p></article>')
        if len(meta['kpis'])>3:stream.write('</div></details>')
        else:stream.write('</div>')
        stream.write('</header><section class="filters" aria-label="结果筛选"><label>搜索<input id="search" type="search" placeholder="名称、签名、样本或原因"></label>')
        stream.write('<span id="dimensions"></span>')
        stream.write('<button id="reset">清除筛选</button><button id="print">打印 / 保存 PDF</button></section><p id="print-scope" class="print-scope">打印保留当前筛选及当前页，完整明细见 HTML 与导出。</p>')
        for section in sections:
            ident=section['id'];fold=section['kind']=='quality'
            stream.write(('<details' if fold else '<section')+' class="panel" id="'+escape(ident,quote=True)+'" data-section="'+escape(ident,quote=True)+'">')
            stream.write(('<summary>' if fold else '<h2>')+escape(section['title'])+('</summary>' if fold else '</h2>'))
            if section['description']:
                stream.write('<p class="description">'+escape(reader_text(section['description']))+'</p>')
            if section.get('message'):
                stream.write('<p class="oracle-message">'+escape(section['message'])+'</p>')
                if any(s['id']=='oracle.layer' for s in sections):
                    stream.write('<a href="#oracle.layer">查看逐层数值差异</a> · ')
                stream.write('<a href="'+escape(section['evidence_href'])+'">查看原始证据</a></section>')
                continue
            if section.get('evidence_index'):
                stream.write('<p>本页提供前 '+str(model.limits.html_rows)+' 条证据样本，筛选仅作用于这些样本。<a href="'+section['evidence_index']+'">阅读全部 '+str(section['count'])+' 条证据（独立分页）</a>。</p>')
            if section.get('status') not in ('ok','complete','available'):
                stream.write('<p class="status">'+escape(display(section['status']))+' · 此状态描述数据可用范围。</p>')
            if ident in ('resource.series','memory.series'):
                stream.write('<div class="chart-controls"><label>曲线<select class="series-select"></select></label>'
                             '<details class="resource-time-tools"><summary>时间范围与峰值</summary><div>'
                             '<label>起点（秒）<input class="time-low" type="number" step="any"></label>'
                             '<label>终点（秒）<input class="time-high" type="number" step="any"></label>'
                             '<button class="chart-peak">定位峰值</button></div></details></div><div class="timeline"></div>'
                             '<button type="button" class="resource-records-toggle" aria-expanded="false" aria-controls="'+
                             ident+'-rows">全部指标采样明细（'+format(section['count'],',')+' 条）</button>')
            elif section['kind'] in ('memory','monitor'):
                stream.write('<div class="chart-controls"><label>曲线<select class="series-select"></select></label><label>起点（秒）<input class="time-low" type="number" step="any"></label><label>终点（秒）<input class="time-high" type="number" step="any"></label><button class="chart-peak">定位峰值</button></div><div class="timeline"></div>')
            elif section.get('chart'):
                stream.write('<div class="plot" aria-label="结果图"></div>')
            priority_sort = ident in ('oracle.identity_diff','oracle.op_sensitivity')
            default_label = '优先显示待核对项' if priority_sort else '原始记录顺序'
            stream.write('<div class="table-tools"><p class="row-count">共 '+str(section['count'])+' 条，默认每页 '+str(model.limits.html_rows)+' 条。</p><label>排序<select class="sort"><option value="">'+default_label+'</option>')
            for key in section['columns']:
                stream.write('<option value="'+escape(key,quote=True)+'">'+escape(PANEL_LABELS.get(ident,{}).get(key,LABELS.get(key,key)))+'</option>')
            stream.write('</select></label><button class="direction" title="切换排序方向">降序</button></div>')
            stream.write('<div class="table-scroll" id="'+escape(ident,quote=True)+'-rows"><table data-panel="'+escape(ident,quote=True)+'"><thead><tr>')
            for key in section['columns']:
                stream.write('<th scope="col" data-field="'+escape(key,quote=True)+'">'+escape(PANEL_LABELS.get(ident,{}).get(key,LABELS.get(key,key)))+'</th>')
            stream.write('<th scope="col">证据与关联</th></tr></thead><tbody>')
            indexed = enumerate(model.rows(ident))
            first_page = (nsmallest(model.limits.html_rows,indexed,key=lambda entry:entry[1].get('review_priority',3))
                          if priority_sort else itertools.islice(indexed,model.limits.html_rows))
            for index,row in first_page:
                _row(stream,section,index,row)
            stream.write('</tbody></table></div><div class="pager"><button class="previous">上一页</button><span class="page-number"></span><button class="next">下一页</button></div>')
            stream.write('</details>' if fold else '</section>')
        stream.write('<section class="panel" id="availability"><h2>观测范围与未提供内容</h2><ul>')
        for note in dict.fromkeys(meta['notices']):
            stream.write('<li>'+escape(reader_text(note))+'</li>')
        if not meta['notices']:stream.write('<li>本次结果的范围说明随各指标展示。</li>')
        stream.write('</ul></section><section class="panel" id="sources"><h2>来源与导出</h2><p>复制整个报告目录可保留下钻与导出。机器导出保留原始单位与字段。</p><ul>')
        for link in meta['links']+[dict(label='Markdown 结果摘要',href='report.md')]:
            href=link['href']
            if not href.startswith(('/','http:','https:','javascript:')) and '..' not in Path(href).parts:
                stream.write('<li><a href="'+escape(href,quote=True)+'">'+escape(link['label'])+'</a></li>')
        stream.write('</ul></section></main><dialog id="row-detail"><button id="close-detail">关闭</button><div id="detail-body"></div></dialog>')
        stream.write('<script id="report-config" type="application/json">'+safe_json(_web_value(dict(meta={k:v for k,v in meta.items() if k not in ('kpis','findings','facts','banners','context_summary','notices','rendered_from')},sections=sections,labels=LABELS,panel_labels=PANEL_LABELS,values=VALUES,missing_text=MISSING_TEXT,raw_clock_fields=sorted(RAW_CLOCK_FIELDS),page_size=model.limits.html_rows)))+'</script>')
        # Batches bound Python memory; only aggregate/curated records are delivered.
        for section in sections:
            iterator=iter(itertools.islice(model.rows(section['id']),model.limits.html_rows) if section.get('evidence_index') else model.rows(section['id']))
            while True:
                batch=list(itertools.islice(iterator,100))
                if not batch:break
                stream.write('<script class="report-records" data-section="'+escape(section['id'],quote=True)+'" type="application/json">'+records_json(_web_value([evidence_links(model,row) for row in batch]))+'</script>')
        scripts=['workspaces.js','charts.js']
        if any(s['group']=='resource' for s in sections):
            scripts.append('resource.js')
        if any(s['id']=='monitor.series' for s in sections):
            scripts.append('monitor.js')
        scripts.append('report.js')
        script='\n'.join((assets/name).read_text() for name in scripts)
        stream.write("<script>(function(){'use strict';\n"+script+"\n})();</script></body></html>")
    write_markdown(model,out,filename)


def evidence_pages(model,out,section,filename):
    """All event-level records remain readable without loading them into the browser."""
    folder=out/'evidence';folder.mkdir(exist_ok=True)
    stem=section['id'].replace('.','-');size=500;count=len(model.rows(section['id']))
    pages=(count+size-1)//size;iterator=iter(model.rows(section['id']))
    for page in range(pages):
        path=folder/(stem+'-'+str(page+1)+'.html')
        with path.open('w') as stream:
            stream.write('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+escape(section['title'])+'</title><style>body{font:15px system-ui;max-width:1100px;margin:24px auto;padding:0 16px;color:#18354b}dd{overflow-wrap:anywhere}details{padding:12px;border-bottom:1px solid #ccd}dt{color:#526879}nav{display:flex;gap:20px}pre{white-space:pre-wrap}</style><h1>'+escape(section['title'])+'</h1>')
            def nav():
                stream.write('<nav><a href="../'+filename+'#'+section['id']+'">返回结果</a>')
                if page:stream.write('<a href="'+stem+'-'+str(page)+'.html">上一页</a>')
                if page+1<pages:stream.write('<a href="'+stem+'-'+str(page+2)+'.html">下一页</a>')
                stream.write('<span>'+str(page+1)+' / '+str(pages)+' 页 · 共 '+str(count)+' 条</span></nav>')
            nav()
            for n,row in enumerate(itertools.islice(iterator,size),page*size):
                stream.write('<details id="row-'+str(n)+'"><summary>'+str(n+1)+' · '+escape(str(row.get('call_key') or row.get('name') or row.get('scope') or '证据记录'))+'</summary><dl>')
                for key,value in row.items():
                    if key != '_field_notes':
                        unit = value_field(row,key)
                        shown = field_text(row,key) if value is None else display(value,unit)
                        stream.write('<dt>'+escape(PANEL_LABELS.get(section['id'], {}).get(key, LABELS.get(key,key)))+'</dt><dd>'+escape(reader_text(shown,unit))+'</dd>')
                stream.write('</dl>')
                for link in evidence_links(model,row).get('_evidence_links',[]):
                    stream.write('<p><a href="'+escape(link['href'].removeprefix('evidence/'),quote=True)+'">'+escape(link['label'])+'</a></p>')
                stream.write('</details>')
            nav();stream.write('</html>')
    return 'evidence/'+stem+'-1.html'


def write_markdown(model,out,filename):
    def text(value, field=''):
        shown = reader_text(value,field)
        if field in ('sample','samples','model','call_key'):
            shown = escape(shown,quote=False)
        return shown.replace('|','\\|').replace('\n',' ')
    with (out/'report.md').open('w',encoding='utf-8') as stream:
        stream.write('# '+model.meta['title']+'\n\n[打开完整交互报告]('+filename+')\n\n')
        summary = model.meta.get('oracle_overview')
        if summary:
            stream.write('**'+text(summary['title'])+'。** '+text(summary['detail'])+' [查看判断依据]('+filename+'#'+summary['section']+')\n\n')
        facts=model.meta['context_summary']
        if facts:
            stream.write('## 任务与条件\n\n')
            paired=any('left' in f for f in facts)
            stream.write('| 条件 | 基准 / 参考 | 候选 / 比较 | 状态 |\n|---|---|---|---|\n' if paired else '| 条件 | 记录值 |\n|---|---|\n')
            for f in facts:
                fields=[f['label'],f.get('left',f.get('value',''))]
                if paired:fields += [f.get('right',''),(f.get('status_text') or ('**不同**' if f.get('different') else '一致')) if 'left' in f else '来源信息']
                stream.write('| '+' | '.join(text(v,f.get('field','') if i in (1,2) else '') for i,v in enumerate(fields))+' |\n')
        if model.meta['credibility']:
            stream.write('\n## 数据范围与口径\n\n')
            for f in model.meta['credibility']:stream.write('- '+text(f['detail'])+'\n')
        if model.meta['findings'] or not summary:
            stream.write('\n## 本次发现\n\n')
        for finding in model.meta['findings']:
            stream.write('- **'+text(finding['title'])+'**：'+text(finding['detail']))
            if finding.get('section'):
                stream.write(' [查看结果]('+finding_href(finding,filename)+')')
            stream.write('\n')
        if not model.meta['findings'] and not summary:
            stream.write('本次已保存结果见下方功能索引；尚无可形成总览的派生结论。\n')
        for section,columns,rows in excerpts(model):
            panel_labels = PANEL_LABELS.get(section['id'],{})
            stream.write('\n### '+text(section['title'])+'\n\n| '+' | '.join(panel_labels.get(k,LABELS.get(k,k)) for k in columns)+' |\n|'+ '---|'*len(columns)+'\n')
            for row in rows:
                values=[text(field_text(row,k),value_field(row,k)) for k in columns]
                stream.write('| '+' | '.join(values)+' |\n')
        stream.write('\n## 结果入口\n\n| 功能 | 记录数 | 统计范围 |\n|---|---:|---|\n')
        for section in model.sections:
            if section['kind']=='quality':continue
            stream.write('| ['+text(section['title'])+']('+filename+'#'+section['id']+') | '+
                         str(len(model.rows(section['id'])))+' | '+text(section.get('message') or section['description'])+' |\n')
        if model.meta['notices']:
            stream.write('\n## 观测范围\n\n')
            for notice in dict.fromkeys(model.meta['notices']):
                stream.write('- '+text(notice)+'\n')
        stream.write('\n[完整机器结果](result.json)。指标原始单位及字段保留在导出中。\n')
        for link in model.meta['links']:
            if link['href'].startswith('exports/'):
                stream.write('- ['+text(link['label'])+']('+link['href']+')\n')
