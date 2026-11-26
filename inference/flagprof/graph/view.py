# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Self-contained structure navigation, using only the standard library."""
from html import escape
import json
from pathlib import Path
from .data import read_manifest


def write_view(out, manifest, graphs):
    # JSON is data, never HTML; escaping '<' also closes the script-end injection path.
    data = json.dumps(graphs, ensure_ascii=False).replace('<', '\\u003c').replace('&','\\u0026')
    info = {k:manifest.get(k) for k in ('status','bundle','identity','inputs','outputs','software','settings','formats','error')}
    summary = [str(manifest.get('bundle','unknown'))+' · '+str(manifest.get('status','unknown'))]
    for fmt,row in manifest.get('formats',{}).items():
        validation = row.get('roundtrip',{}).get('status', row.get('structural_check','not_run'))
        summary.append(fmt.upper()+': '+str(row['status'])+'；回读/结构 '+validation+
                       ('；ORT '+row.get('runtime',{}).get('status','not_run') if fmt=='onnx' else ''))
    body = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>FlagProf · 模型计算图</title><style>
body{font:15px system-ui,-apple-system,"Segoe UI","PingFang SC","Noto Sans CJK SC","Microsoft YaHei",sans-serif;margin:0;color:#172b39;background:#f5f7fa}header,main{padding:22px;max-width:1300px;margin:auto}
h1{margin:0 0 10px}p{line-height:1.65}input,select,button{font:inherit;padding:7px;border:1px solid #afbac6;border-radius:5px}
button{cursor:pointer;background:#fff;color:#18547a;margin:3px}#layout{display:grid;grid-template-columns:minmax(280px,1fr) 2fr;gap:18px}
#nodes{max-height:70vh;overflow:auto}details,article{background:white;padding:14px;border-radius:8px;margin:8px 0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px ui-monospace,SFMono-Regular,Consolas,"Liberation Mono","Noto Sans Mono CJK SC","Noto Sans CJK SC",monospace}summary{cursor:pointer}#nodes button{display:block;text-align:left;width:98%}
.muted{color:#566673}@media(max-width:750px){#layout{grid-template-columns:1fr}}.selected{background:#def1ff}
</style><header><h1>模型计算图</h1><p>阅读路径：选择格式 → 展开模块 → 选择节点 → 沿输入与使用者查看数据依赖。</p>
<p class="muted">这是固定输入形状下的 CPU 原生计算结构。图中节点编号只在本图有效；设备执行耗时与 profiling 关联将在后续提供。</p>
<p>SUMMARY</p><details><summary>导出身份、输入输出与逐格式验证</summary><pre>INFO</pre></details></header>
<main><label>格式 <select id="format"></select></label> <label>搜索 <input id="search" placeholder="节点、算子或模块"></label>
<p id="count"></p><div id="layout"><section id="nodes"></section><article id="detail">选择节点，查看张量形状、来源和上下游依赖。</article></div></main>
<script type="application/json" id="graph-data">DATA</script><script>
const graphs=JSON.parse(document.getElementById('graph-data').textContent), select=document.getElementById('format');
for(const fmt of Object.keys(graphs)){const o=document.createElement('option');o.textContent=fmt;o.value=fmt;select.append(o)}
function button(text,fn){const b=document.createElement('button');b.textContent=text;b.onclick=fn;return b}
function show(id){const g=graphs[select.value], n=g.nodes.find(x=>x.id===id), d=document.getElementById('detail');if(!n)return;d.replaceChildren();
 const h=document.createElement('h2');h.textContent=n.id+' · '+n.target;d.append(h);
 for(const [title,key] of [['输入节点','inputs'],['使用者','users']]){const p=document.createElement('p');p.textContent=title+': ';for(const v of n[key]||[])p.append(button(v,()=>show(v)));d.append(p)}
 const pre=document.createElement('pre');pre.textContent=JSON.stringify(n,null,2);d.append(pre)}
function draw(){const g=graphs[select.value], root=document.getElementById('nodes');root.replaceChildren();if(!g)return;
 const query=document.getElementById('search').value.toLowerCase(), groups=new Map();let visible=0;
 for(const n of g.nodes){const module=n.modules?.length?n.modules[n.modules.length-1].path:'未记录模块 / 图级节点';
 if(query&&!JSON.stringify([n.id,n.target,module]).toLowerCase().includes(query))continue;visible++;
 if(!groups.has(module)){const d=document.createElement('details'),s=document.createElement('summary');s.textContent=module;d.append(s);d.open=!!query;root.append(d);groups.set(module,d)}
 groups.get(module).append(button(n.id+' · '+n.target,()=>show(n.id)))}
 document.getElementById('count').textContent=`显示 ${visible} / ${g.nodes.length} 个节点；参数项 ${Object.keys(g.parameters||{}).length}`;
 const first=root.querySelector('details');if(first)first.open=true}
select.onchange=draw;document.getElementById('search').oninput=draw;draw();
</script></html>'''
    body = body.replace('>DATA</script>', '>'+data+'</script>').replace('>INFO</pre>', '>'+escape(json.dumps(info, ensure_ascii=False, indent=2))+'</pre>')
    body = body.replace('>SUMMARY</p>', '>'+escape(' | '.join(summary))+'</p>')
    Path(out).write_text(body)


def render_graph(path, out=None):
    root = Path(path)
    manifest = read_manifest(root)
    graphs = {fmt:json.loads((root/(fmt+'.json')).read_text())
              for fmt in manifest['formats'] if (root/(fmt+'.json')).is_file()}
    # Re-render outside a sealed bundle so its checksums stay valid.
    if out is None:
        raise ValueError('provide an output HTML path outside the graph bundle')
    destination = Path(out).resolve()
    if root.resolve() == destination or root.resolve() in destination.parents:
        raise ValueError('render output must be outside the sealed graph bundle')
    write_view(destination, manifest, graphs)
    return str(destination)
