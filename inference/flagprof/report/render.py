# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Offline HTML and Markdown from the same disk-backed presentation model."""
from html import escape
import itertools
import json
import hashlib
from pathlib import Path
from urllib.parse import urlencode

from .fields import LABELS, VALUES, cell, display


def safe_json(value):
    return json.dumps(value,ensure_ascii=False,allow_nan=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e').replace('\u2028','\\u2028').replace('\u2029','\\u2029')


def _web_value(value):
    if isinstance(value,int) and not isinstance(value,bool) and abs(value)>9007199254740991:
        return str(value)
    if isinstance(value,dict):
        return {k:_web_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_web_value(v) for v in value]
    return value


def _row(stream, section, index, row):
    stream.write('<tr data-row="'+str(index)+'">')
    for key in section['columns']:
        value=cell(row,key)
        shown=display(value,'duration_ns' if row.get('unit')=='ns' and key in ('baseline','candidate','delta') else key)
        brief=shown if len(shown)<=140 else shown[:137]+'…'
        content = escape(brief)
        if key == 'report' and isinstance(value,str) and not value.startswith('/') and ':' not in value and '..' not in Path(value).parts:
            content = '<a href="'+escape(value,quote=True)+'">打开 rank 结果</a>'
        stream.write('<td data-panel="'+escape(section['id'],quote=True)+'" data-row="'+str(index)+
                     '" data-field="'+escape(key,quote=True)+'" data-unit="'+('ns' if key.endswith('_ns') else '')+
                     '" data-value="'+escape(safe_json(value),quote=True)+'" title="'+escape(shown,quote=True)+'">'+content+'</td>')
    stream.write('<td><button class="detail" data-section="'+escape(section['id'],quote=True)+
                 '" data-index="'+str(index)+'">查看详情与证据</button></td></tr>')


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


def render(model, out, *, filename='report.html'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assets=Path(__file__).parent/'assets'
    meta=model.meta
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
    for name in ('render.py','fields.py','assets/report.css','assets/report.js'):
        fingerprint.update((Path(__file__).parent/name).read_bytes())
    (out/'renderer.json').write_text(json.dumps(dict(schema='report-renderer/1',sha256=fingerprint.hexdigest(),
        presentation_schema=meta['schema']),indent=2)+'\n')
    with (out/filename).open('w',encoding='utf-8') as stream:
        stream.write('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">')
        stream.write('<title>'+escape(meta['title'])+' · FlagProf</title><style>'+ (assets/'report.css').read_text()+'</style></head><body>')
        stream.write('<a class="skip" href="#overview">跳到结果概览</a><aside class="sidebar"><div class="brand">FlagProf<span>推理结果报告</span></div><nav aria-label="报告目录"><a href="#overview">结果概览</a>')
        for section in sections:
            if section['kind']!='quality':
                stream.write('<a href="#'+escape(section['id'],quote=True)+'">'+escape(section['title'])+'<small>'+str(section['count'])+'</small></a>')
        stream.write('<a href="#availability">观测范围与未提供内容</a><a href="#sources">来源与导出</a></nav></aside><main>')
        stream.write('<header id="overview"><p class="eyebrow">运行结果 · 可离线阅读</p><h1>'+escape(meta['title'])+'</h1>')
        stream.write('<p class="intro">先阅读本次发现，再沿对应结果查看范围和证据。总览保留原运行范围；筛选作用于下方明细。</p>')
        if meta['facts']:
            compact = [fact for fact in meta['facts'] if fact['label'] in ('设备','精度','输入样本','参考侧','基线侧','候选侧','来源 1','来源 2')]
            stream.write('<p class="task-brief">'+escape(' · '.join(fact['label']+'：'+display(fact['value']) for fact in compact))+'</p>')
            stream.write('<details class="identity"><summary>本次任务与比较条件（展开核对）</summary><dl>')
            for fact in meta['facts']:
                stream.write('<div><dt>'+escape(fact['label'])+'</dt><dd>'+escape(display(fact['value']))+'</dd></div>')
            stream.write('</dl></details>')
        stream.write('<div class="findings">')
        for finding in meta['findings']:
            stream.write('<article class="finding '+escape(finding['level'],quote=True)+'"><h2>'+escape(finding['title'])+'</h2><p>'+escape(finding['detail'])+'</p>')
            if finding.get('section') and any(s['id']==finding['section'] for s in sections):
                focus = ''
                if finding.get('row_index') is not None:
                    focus = ' data-focus-section="'+escape(finding['section'],quote=True)+'" data-focus-row="'+str(finding['row_index'])+'"'
                stream.write('<a href="'+escape(finding_href(finding,filename),quote=True)+'"'+focus+'>查看对应结果 →</a>')
            stream.write('</article>')
        if not meta['findings']:
            stream.write('<article class="finding"><h2>查看本次可用结果</h2><p>下方按功能整理已保存的观测；缺少的记录及原因在观测范围中说明。</p></article>')
        stream.write('</div></header><section class="filters" aria-label="结果筛选"><label>搜索<input id="search" type="search" placeholder="名称、签名、样本或原因"></label>')
        for key,label in [('rank','rank'),('layer','层 / 模块'),('phase','阶段'),('shape','输入 shape'),('sid','签名'),('step','生成步')]:
            stream.write('<label>'+label+'<input id="filter-'+key+'" placeholder="全部" list="options-'+key+'"><datalist id="options-'+key+'"></datalist></label>')
        stream.write('<button id="reset">清除筛选</button><button id="print">打印 / 保存 PDF</button></section><p id="print-scope" class="print-scope">打印保留当前筛选及当前页，完整明细见 HTML 与导出。</p>')
        for section in sections:
            ident=section['id'];fold=section['kind']=='quality'
            stream.write(('<details' if fold else '<section')+' class="panel" id="'+escape(ident,quote=True)+'" data-section="'+escape(ident,quote=True)+'">')
            stream.write(('<summary>' if fold else '<h2>')+escape(section['title'])+('</summary>' if fold else '</h2>'))
            stream.write('<p class="description">'+escape(section['description'])+'</p>')
            if section.get('evidence_index'):
                stream.write('<p>本页提供前 '+str(model.limits.html_rows)+' 条证据样本，筛选仅作用于这些样本。<a href="'+section['evidence_index']+'">阅读全部 '+str(section['count'])+' 条证据（独立分页）</a>。</p>')
            if section.get('status') not in ('ok','complete','available'):
                stream.write('<p class="status">'+escape(display(section['status']))+' · 此状态描述数据可用范围。</p>')
            if section['kind'] in ('memory','monitor'):
                stream.write('<div class="chart-controls"><label>曲线<select class="series-select"></select></label><label>起点（秒）<input class="time-low" type="number" step="any"></label><label>终点（秒）<input class="time-high" type="number" step="any"></label><button class="chart-peak">定位峰值</button></div><div class="timeline"></div>')
            elif ident in ('perf.device','perf.op','perf.layer','oracle.layer','perf.gap','communication.aligned','performance.baseline'):
                stream.write('<div class="plot" aria-label="结果图"></div>')
            stream.write('<div class="table-tools"><p class="row-count">共 '+str(section['count'])+' 条，默认每页 '+str(model.limits.html_rows)+' 条。</p><label>排序<select class="sort">')
            for key in section['columns']:
                stream.write('<option value="'+escape(key,quote=True)+'">'+escape(LABELS.get(key,key))+'</option>')
            stream.write('</select></label><button class="direction" title="切换排序方向">降序</button></div>')
            stream.write('<div class="table-scroll"><table data-panel="'+escape(ident,quote=True)+'"><thead><tr>')
            for key in section['columns']:
                stream.write('<th scope="col">'+escape(LABELS.get(key,key))+'</th>')
            stream.write('<th scope="col">证据与关联</th></tr></thead><tbody>')
            for index,row in enumerate(itertools.islice(model.rows(ident),model.limits.html_rows)):
                _row(stream,section,index,row)
            stream.write('</tbody></table></div><div class="pager"><button class="previous">上一页</button><span class="page-number"></span><button class="next">下一页</button></div>')
            if ident=='diagnosis.evidence':
                for row in model.rows(ident):
                    stream.write('<details class="evidence-record" id="'+escape(row['anchor'],quote=True)+'"><summary>触发事件 '+escape(display(row['identity']))+'</summary><dl>')
                    for key,value in row.get('record',{}).items():
                        stream.write('<dt>'+escape(LABELS.get(key,key))+'</dt><dd>'+escape(display(value,key))+'</dd>')
                    stream.write('</dl></details>')
            stream.write('</details>' if fold else '</section>')
        stream.write('<section class="panel" id="availability"><h2>观测范围与未提供内容</h2><ul>')
        for note in dict.fromkeys(meta['notices']):
            stream.write('<li>'+escape(str(note))+'</li>')
        if not meta['notices']:stream.write('<li>本次结果的范围说明随各指标展示。</li>')
        stream.write('</ul></section><section class="panel" id="sources"><h2>来源与导出</h2><p>复制整个报告目录可保留下钻与导出。机器导出保留原始单位与字段。</p><ul>')
        for link in meta['links']+[dict(label='Markdown 结果摘要',href='report.md')]:
            href=link['href']
            if not href.startswith(('/','http:','https:','javascript:')) and '..' not in Path(href).parts:
                stream.write('<li><a href="'+escape(href,quote=True)+'">'+escape(link['label'])+'</a></li>')
        stream.write('</ul></section></main><dialog id="row-detail"><button id="close-detail">关闭</button><div id="detail-body"></div></dialog>')
        stream.write('<script id="report-config" type="application/json">'+safe_json(dict(meta=meta,sections=sections,labels=LABELS,values=VALUES,page_size=model.limits.html_rows))+'</script>')
        # Batches bound Python memory; only aggregate/curated records are delivered.
        for section in sections:
            iterator=iter(itertools.islice(model.rows(section['id']),model.limits.html_rows) if section.get('evidence_index') else model.rows(section['id']))
            while True:
                batch=list(itertools.islice(iterator,100))
                if not batch:break
                stream.write('<script class="report-records" data-section="'+escape(section['id'],quote=True)+'" type="application/json">'+safe_json(_web_value([evidence_links(model,row) for row in batch]))+'</script>')
        stream.write('<script>'+(assets/'report.js').read_text()+'</script></body></html>')
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
                for key,value in row.items():stream.write('<dt>'+escape(LABELS.get(key,key))+'</dt><dd>'+escape(display(value,key))+'</dd>')
                stream.write('</dl>')
                for link in evidence_links(model,row).get('_evidence_links',[]):
                    stream.write('<p><a href="'+escape(link['href'].removeprefix('evidence/'),quote=True)+'">'+escape(link['label'])+'</a></p>')
                stream.write('</details>')
            nav();stream.write('</html>')
    return 'evidence/'+stem+'-1.html'


def write_markdown(model,out,filename):
    def text(value):
        return str(value).replace('|','\\|').replace('\n',' ')
    with (out/'report.md').open('w',encoding='utf-8') as stream:
        stream.write('# '+model.meta['title']+'\n\n[打开完整交互报告]('+filename+')\n\n')
        for fact in model.meta['facts']:
            stream.write('- **'+text(fact['label'])+'**：'+text(display(fact['value']))+'\n')
        stream.write('\n## 本次发现\n\n')
        for finding in model.meta['findings']:
            stream.write('- **'+text(finding['title'])+'**：'+text(finding['detail']))
            if finding.get('section'):
                stream.write(' [查看结果]('+finding_href(finding,filename)+')')
            stream.write('\n')
        if not model.meta['findings']:
            stream.write('本次已保存结果见下方功能索引；尚无可形成总览的派生结论。\n')
        stream.write('\n## 结果入口\n\n| 功能 | 记录数 | 统计范围 |\n|---|---:|---|\n')
        for section in model.sections:
            if section['kind']=='quality':continue
            stream.write('| ['+text(section['title'])+']('+filename+'#'+section['id']+') | '+
                         str(len(model.rows(section['id'])))+' | '+text(section['description'])+' |\n')
        if model.meta['notices']:
            stream.write('\n## 观测范围\n\n')
            for notice in dict.fromkeys(model.meta['notices']):
                stream.write('- '+text(notice)+'\n')
        stream.write('\n[完整机器结果](result.json)。指标原始单位及字段保留在导出中。\n')
