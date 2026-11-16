(function(){
'use strict';
var cfg=JSON.parse(document.getElementById('report-config').textContent), data={}, states={}, timers;
var $=function(id){return document.getElementById(id);};
var esc=function(s){return String(s).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];});};
document.querySelectorAll('.report-records').forEach(function(s){var id=s.dataset.section;data[id]=(data[id]||[]).concat(JSON.parse(s.textContent));s.remove();});
function num(n){if(!Number.isFinite(n))return '无有限值';return n===0?'0':Number(n.toPrecision(6)).toString();}
function fmt(v,k){
 if(v===null||v===undefined)return '未记录 / 不适用';
 if(typeof v==='boolean')return v?'是':'否';
 if(typeof v==='number'){
  if(/_ns$/.test(k||'')){if(Math.abs(v)>=1e9)return num(v/1e9)+' s';if(Math.abs(v)>=1e6)return num(v/1e6)+' ms';if(Math.abs(v)>=1e3)return num(v/1e3)+' μs';return num(v)+' ns';}
  if(/_bytes$/.test(k||'')||k==='bytes'){for(var pair of [[1073741824,'GiB'],[1048576,'MiB'],[1024,'KiB']])if(Math.abs(v)>=pair[0])return num(v/pair[0])+' '+pair[1];return num(v)+' B';}
  if(/_(ratio|share|fraction)$/.test(k||'')||k==='relative_change'||k==='repeat_cv')return num(v*100)+'%';
  return num(v);
 }
 if(typeof v==='string'){if(k==='instrumentation'&&v==='none')return '无插桩';if(k==='field'||k==='metric')return v.split('.').map(function(p){return cfg.labels[p]||cfg.values[p]||p;}).join(' / ');return cfg.values[v]||v;}
 if(Array.isArray(v))return v.length?v.map(function(x){return fmt(x);}).join(/shape/.test(k||'')?' × ':'、'):'无';
 if('value' in v){var t=fmt(v.value,k);[['sample','样本'],['index','位置'],['ulp','ULP']].forEach(function(p){if(v[p[0]]!=null)t+=' · '+p[1]+' '+v[p[0]];});return t;}
 return Object.keys(v).map(function(x){return (cfg.labels[x]||x)+'：'+fmt(v[x],x);}).join('；');
}
function brief(v,k){var text=fmt(v,k);return text.length<=140?text:text.slice(0,137)+'…';}
function val(row,k){return row[k]===undefined&&row.worst?row.worst[k]:row[k];}
function scalar(v){return v&&typeof v==='object'&&!Array.isArray(v)&&'value' in v?v.value:v;}
function tokens(row,key){
 if(key==='sid')return [row.sid,row.sid_ref,row.sid_cmp].filter(Boolean).join(' ');
 if(key==='layer')return [row.scope,row.name].concat(row.scopes||[]).filter(Boolean).join(' ');
 if(key==='phase')return [row.phase,row.kind,(cfg.meta.phase_bands||[]).some(function(b){return b.name===row.name;})?row.name:null].filter(Boolean).join(' ');
 if(key==='shape')return JSON.stringify(row.input_shape||row.shape||row.signature||'');
 if(key==='step')return String(row.generation_step===undefined?(row.step===undefined?'':row.step):row.generation_step);
 return String(row[key]===undefined?'':row[key]);
}
function accepts(row){
 var query=$('search').value.trim().toLowerCase();
 if(query&&JSON.stringify(row).toLowerCase().indexOf(query)<0)return false;
 return ['rank','layer','phase','shape','sid','step'].every(function(k){var q=$('filter-'+k).value.trim();if(!q)return true;var v=tokens(row,k);return k==='rank'||k==='step'?v===q:v.toLowerCase().indexOf(q.toLowerCase())>=0;});
}
function detail(id,index){
 var row=data[id][index],body=$('detail-body');
 var section=cfg.sections.find(function(s){return s.id===id;});
 body.innerHTML='<h2>'+esc(section.title)+' · 详情与证据</h2><p>'+esc(section.description)+'</p>';
 var links=document.createElement('p');
 [['sid','查看同签名记录'],['sid_ref','查看参考侧签名'],['sid_cmp','查看比较侧签名'],['call_key','查看同调用记录']].forEach(function(p){if(row[p[0]]){var b=document.createElement('button');b.textContent=p[1];b.onclick=function(){related(row[p[0]],p[0]==='call_key'?'call_key':'sid',id,index);};links.appendChild(b);}});
 (Array.isArray(row.evidence)?row.evidence:[]).forEach(function(ref){if(!Array.isArray(ref))return;var target=(data['diagnosis.evidence']||[]).find(function(r){return JSON.stringify(r.identity)===JSON.stringify(ref);});if(target){var a=document.createElement('a');a.href='#'+target.anchor;a.textContent='查看触发事件 '+ref.join(' / ');a.onclick=function(){$('row-detail').close();var node=$(target.anchor);if(node)node.open=true;};links.appendChild(a);}});
 (row.input_sources_ref||[]).concat(row.input_sources_cmp||[]).filter(function(v,i,a){return v!=='unknown'&&a.indexOf(v)===i;}).forEach(function(key){var b=document.createElement('button');b.textContent='输入生产者：'+key;b.onclick=function(){related(key,'call_key',id,index);};links.appendChild(b);});
 (row._evidence_links||[]).forEach(function(link){var a=document.createElement('a');a.href=link.href;a.textContent=link.label;links.appendChild(a);});
 body.appendChild(links);
 var dl=document.createElement('dl');
 Object.keys(row).filter(function(k){return k!=='source_result'&&k!=='_evidence_links';}).forEach(function(k){var dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=cfg.labels[k]||k;dd.textContent=fmt(row[k],k);dl.append(dt,dd);});
 body.appendChild(dl);
 var numeric=row.metrics||row,hist=numeric.ulp_histogram,quantiles=numeric.absolute_error_quantiles;
 function mini(values,title){var entries=Object.keys(values).map(function(k){return [k,values[k]];}).filter(function(p){return typeof p[1]==='number';});if(!entries.length)return;var max=Math.max.apply(null,entries.map(function(p){return p[1];}).concat([1e-300])),chart=document.createElement('div');chart.className='plot';chart.innerHTML='<h3>'+esc(title)+'</h3><div class=bars>'+entries.map(function(p){return '<div class=bar><span>'+esc(p[0])+'</span><span class=bar-track><span class=bar-fill style="display:block;width:'+100*p[1]/max+'%"></span></span><span>'+esc(fmt(p[1]))+'</span></div>';}).join('')+'</div>';body.appendChild(chart);}
 if(hist)mini(hist,'当前边界不同元素的 ULP 分布；相同元素不在桶中');
 if(quantiles)mini(quantiles,'当前有限有效元素的绝对误差分位数');

 var raw=document.createElement('details');raw.innerHTML='<summary>核对原始字段</summary>';var pre=document.createElement('pre');pre.textContent=JSON.stringify(row,null,2);raw.appendChild(pre);body.appendChild(raw);
 $('row-detail').showModal();
}
document.addEventListener('click',function(e){var b=e.target.closest('button.detail');if(b)detail(b.dataset.section,Number(b.dataset.index));var link=e.target.closest('[data-focus-section]');if(link){e.preventDefault();focusRow(link.dataset.focusSection,Number(link.dataset.focusRow),true);}});
$('close-detail').onclick=function(){$('row-detail').close();};
function bars(panel,section,selected){
 var plot=panel.querySelector('.plot');if(!plot)return;
 var key=section.id==='oracle.layer'?'max_abs':section.id==='perf.gap'?'window_ns':section.id==='communication.aligned'?'cross_rank_difference_ns':section.id==='performance.baseline'?'mean_ns':section.id==='perf.layer'?'inclusive_ns':'device_work_ns';
 if(!selected.some(function(i){return typeof scalar(val(data[section.id][i],key))==='number';}))key='host_ns';
 var items=selected.map(function(i){return data[section.id][i];}).filter(function(r){return typeof scalar(val(r,key))==='number';}).sort(function(a,b){return scalar(val(b,key))-scalar(val(a,key));}).slice(0,10);
 if(!items.length){plot.innerHTML='';return;}
 if(section.id==='perf.gap'){plot.innerHTML='<p class=description>当前筛选 · 活动并集与空白；同一窗口两者之和等于窗口长度。</p><div class=bars>'+items.map(function(r){var active=100*r.all_activity_union_ns/r.window_ns;return '<div class=bar><span>'+esc(r.kind)+' '+esc(r.step==null?'':r.step)+'</span><span class=bar-track title="'+esc('活动 '+fmt(r.all_activity_union_ns,'all_activity_union_ns')+'；空白 '+fmt(r.blank_ns,'blank_ns'))+'"><span class=bar-fill style="display:block;width:'+active+'%"></span></span><span>'+esc(fmt(r.window_ns,'window_ns'))+'</span></div>';}).join('')+'</div>';return;}
 var max=Math.max.apply(null,items.map(function(r){return scalar(val(r,key));}).concat([1e-300]));
 plot.innerHTML='<p class="description">当前筛选 · '+esc(cfg.labels[key]||key)+' Top '+items.length+'；条长按最大观测值缩放。</p><div class="bars">'+items.map(function(r){var v=scalar(val(r,key)),label=r.name||r.scope||r.request_id||r.group_seq||r.boundary||r.phase||'观测';return '<div class="bar"><span class="bar-label" title="'+esc(label)+'">'+esc(label)+'</span><span class="bar-track"><span class="bar-fill" style="display:block;width:'+Math.max(0,100*v/max)+'%"></span></span><span>'+esc(fmt(v,key))+'</span></div>';}).join('')+'</div>';
}
function draw(section){
 var panel=$(section.id),state=states[section.id],rows=data[section.id]||[];
 var selected=[];rows.forEach(function(r,i){if(accepts(r))selected.push(i);});
 selected.sort(function(i,j){var a=scalar(val(rows[i],state.sort)),b=scalar(val(rows[j],state.sort));if(a==null)return b==null?i-j:1;if(b==null)return -1;var d=typeof a==='number'&&typeof b==='number'?a-b:String(a).localeCompare(String(b),undefined,{numeric:true});return (state.desc?-d:d)||i-j;});
 state.selected=selected;
 var pages=Math.max(1,Math.ceil(selected.length/cfg.page_size));state.page=Math.min(state.page,pages-1);
 var start=state.page*cfg.page_size,visible=selected.slice(start,start+cfg.page_size),tbody=panel.querySelector('tbody');
 tbody.innerHTML=visible.map(function(i){var r=rows[i];return '<tr data-row="'+i+'">'+section.columns.map(function(k){var v=val(r,k),content=esc(brief(v,r.unit==='ns'&&['baseline','candidate','delta'].includes(k)?'duration_ns':k));if(k==='report'&&typeof v==='string'&&!v.startsWith('/')&&!v.includes(':')&&!v.split('/').includes('..'))content='<a href="'+esc(v)+'">打开 rank 结果</a>';return '<td data-panel="'+esc(section.id)+'" data-row="'+i+'" data-field="'+esc(k)+'" data-unit="'+(/_ns$/.test(k)?'ns':'')+'" data-value="'+esc(JSON.stringify(v===undefined?null:v))+'">'+content+'</td>';}).join('')+'<td><button class="detail" data-section="'+esc(section.id)+'" data-index="'+i+'">查看详情与证据</button></td></tr>';}).join('');
 panel.querySelector('.row-count').textContent=(section.evidence_index?'证据样本筛选 ':'当前筛选 ')+selected.length+' / '+(section.evidence_index?'本页样本 ':'全表 ')+rows.length+' 条；显示 '+(visible.length?start+1:0)+'–'+(start+visible.length)+'。';
 panel.querySelector('.page-number').textContent=(state.page+1)+' / '+pages;
 panel.querySelector('.previous').disabled=state.page===0;panel.querySelector('.next').disabled=state.page+1>=pages;
 bars(panel,section,selected);
 if(section.kind==='memory'||section.kind==='monitor')drawCurve(section);
}
function seriesFor(section){
 var result={},rows=data[section.id]||[];
 if(section.kind==='memory'){
  rows.forEach(function(r){if(r.source_time_ns==null)return;Object.keys(r.values||{}).forEach(function(metric){var id=[r.dev,r.component,metric].join(' / ');(result[id]||(result[id]={unit:r.unit===undefined?'B':r.unit,points:[]})).points.push([Number(r.source_time_ns),r.values[metric],r]);});});
 }else{
  var window=cfg.meta.monitor_window||[0,256e9],span=window[1]-window[0];
  rows.forEach(function(r){var id=r.entity+' / '+r.metric,points=r.samples_points||Array.from({length:256},function(_,k){return [Number(window[0])+k*span/256,r.points&&Object.prototype.hasOwnProperty.call(r.points,k)?r.points[k]:null];});result[id]={unit:r.unit||'',points:points.map(function(p){return [Number(p[0]),p[1],r];})};});
 }
 Object.keys(result).forEach(function(k){result[k].points.sort(function(a,b){return a[0]-b[0];});});
 return result;
}
function drawCurve(section){
 var panel=$(section.id),state=states[section.id],select=panel.querySelector('.series-select');
 if(!state.series){state.series=seriesFor(section);Object.keys(state.series).forEach(function(k){var o=document.createElement('option');o.value=k;o.textContent=k;select.appendChild(o);});}
 var s=state.series[select.value];if(!s||!s.points.length)return;
 var points=s.points,origin=points[0][0],end=(points[points.length-1][0]-origin)/1e9;
 var loInput=panel.querySelector('.time-low'),hiInput=panel.querySelector('.time-high');
 if(state.curveKey!==select.value){loInput.value=0;hiInput.value=end||1;state.curveKey=select.value;}
 var lo=Number(loInput.value),hi=Number(hiInput.value);if(!(hi>lo))hi=lo+1;
 var phase=$('filter-phase').value.trim().toLowerCase(),phaseBands=section.id==='resource.series'?(cfg.meta.phase_bands||[]).filter(function(b){return !phase||b.name.toLowerCase().includes(phase);}):[];
 var chosen=points.filter(function(p){var t=(p[0]-origin)/1e9;return t>=lo&&t<=hi;}).map(function(p){return phase&&section.id==='resource.series'&&!phaseBands.some(function(b){return p[0]>=b.start_ns&&p[0]<=b.end_ns;})?[p[0],null,p[2]]:p;}),valid=chosen.filter(function(p){return p[1]!=null&&Number.isFinite(p[1]);});
 var low=valid.reduce(function(v,p){return Math.min(v,p[1]);},0),high=valid.reduce(function(v,p){return Math.max(v,p[1]);},0);if(high===low)high=low+1;
 var x=function(p){return 65+850*((p[0]-origin)/1e9-lo)/(hi-lo);},y=function(p){return 210-170*(p[1]-low)/(high-low);},path='',on=false;
 chosen.forEach(function(p){if(p[1]==null){on=false;return;}path+=(on?'L':'M')+x(p)+','+y(p)+' ';on=true;});
 var bands=section.id==='resource.series'?(cfg.meta.phase_bands||[]).map(function(b){var a=Math.max(lo,(b.start_ns-origin)/1e9),z=Math.min(hi,(b.end_ns-origin)/1e9);return z>a?'<rect x="'+(65+850*(a-lo)/(hi-lo))+'" y="35" width="'+(850*(z-a)/(hi-lo))+'" height="180" fill="#eaf0f7"><title>'+esc(b.name)+'</title></rect>':'';}).join(''):'';
 var svg='<svg viewBox="0 0 950 260" role="img" aria-label="'+esc(select.value)+'">'+bands+'<text x="8" y="22">'+esc(fmt(high))+' '+esc(s.unit)+'</text><text x="8" y="216">'+esc(fmt(low))+'</text><path d="M65 35 V215 H920" stroke="#c5d3df" fill="none"/><path class="curve" d="'+path+'"/>';
 // Points selected for the plot retain their actual timestamps and values.
 var shown=valid.length<=2048?valid:valid.filter(function(p,i){return i%Math.ceil(valid.length/2048)===0||p[1]===high;});
 shown.forEach(function(p){svg+='<circle class="sample" cx="'+x(p)+'" cy="'+y(p)+'" r="2"><title>'+esc(num((p[0]-origin)/1e9)+' s：'+fmt(p[1])+' '+s.unit)+'</title></circle>';});
 svg+='<text x="65" y="245">'+esc(num(lo))+' s</text><text x="835" y="245">'+esc(num(hi))+' s</text></svg><p class="chart-readout">来源时间相对首点；当前窗口 '+valid.length+' 个有效显示点。悬停查看原始观测；精确统计以统计表为准。</p>';
 panel.querySelector('.timeline').innerHTML=svg;
 state.peak=function(){var p=points.filter(function(x){return x[1]!=null&&(!phase||section.id!=='resource.series'||phaseBands.some(function(b){return x[0]>=b.start_ns&&x[0]<=b.end_ns;}));}).reduce(function(a,b){return !a||b[1]>a[1]?b:a;},null);if(!p)return;var t=(p[0]-origin)/1e9,span=Math.max(end*.05,1e-6);loInput.value=Math.max(0,t-span);hiInput.value=Math.min(end||1,t+span);drawCurve(section);};
}
function layerTree(){
 var panel=$('perf.layer');if(!panel)return;var rows=data['perf.layer']||[],box=document.createElement('div');box.className='layer-tree';
 var byName={};rows.forEach(function(r){byName[r.name]=r;});
 var children={},roots=[];rows.forEach(function(r){var parents=(r.parent_names||[]).filter(function(n){return n!==r.name&&byName[n];});if(parents.length!==1)roots.push(r);else(children[parents[0]]||(children[parents[0]]=[])).push(r);});
 var seen=new Set();function node(r,parent){if(seen.has(r.name))return;seen.add(r.name);var details=document.createElement('details'),summary=document.createElement('summary');summary.textContent=r.name+' · '+fmt(r.inclusive_ns==null?r.host_ns:r.inclusive_ns,'inclusive_ns');details.appendChild(summary);var b=document.createElement('button');b.textContent='筛选此层';b.onclick=function(){$('filter-layer').value=r.name;apply();};details.appendChild(b);(children[r.name]||[]).forEach(function(c){node(c,details);});parent.appendChild(details);}
 var label=document.createElement('p');label.textContent='层次来自已记录的父层关系；未提供关系的层独立列出。父子耗时不能相加。';box.appendChild(label);roots.forEach(function(r){node(r,box);});rows.forEach(function(r){node(r,box);});panel.insertBefore(box,panel.querySelector('.table-tools'));
}
function filterCaption(){var filters=Array.from(document.querySelectorAll('.filters input')).filter(function(e){return e.value.trim();}).map(function(e){return e.parentNode.textContent.trim()+'='+e.value;});$('print-scope').textContent='打印保留当前筛选及各表当前页；'+(filters.length?filters.join('；'):'未设置筛选')+'。每表显示范围与全表条数见表前说明；完整明细见 HTML 与导出。';}
function clearFilters(){document.querySelectorAll('.filters input').forEach(function(e){e.value='';});}
function apply(){filterCaption();cfg.sections.forEach(function(s){states[s.id].page=0;draw(s);});}
function focusRow(id,index,clear){
 var section=cfg.sections.find(function(s){return s.id===id;});if(!section||!data[id]||!data[id][index])return false;
 if(clear){clearFilters();apply();}var panel=$(id),state=states[id],position=state.selected.indexOf(index);if(position<0)return false;
 state.page=Math.floor(position/cfg.page_size);draw(section);if(panel.tagName==='DETAILS')panel.open=true;
 document.querySelectorAll('tr.focused').forEach(function(r){r.classList.remove('focused');});var node=panel.querySelector('tr[data-row="'+index+'"]');node.classList.add('focused');node.scrollIntoView({block:'center'});detail(id,index);return true;
}
function related(key,kind,currentId,currentIndex){
 $('row-detail').close();clearFilters();$('search').value=key;apply();
 var candidates=[];cfg.sections.forEach(function(s){(data[s.id]||[]).forEach(function(r,i){if(s.id===currentId&&i===currentIndex)return;if(kind==='call_key'?r.call_key===key:[r.sid,r.sid_ref,r.sid_cmp].includes(key))candidates.push([s.id,i]);});});
 candidates.sort(function(a,b){return (a[0]===currentId?1:0)-(b[0]===currentId?1:0);});
 if(candidates.length)focusRow(candidates[0][0],candidates[0][1],false);else{var panel=cfg.sections.find(function(s){return states[s.id].selected.length;});if(panel)$(panel.id).scrollIntoView({block:'start'});}
}
cfg.sections.forEach(function(section){
 var panel=$(section.id),preferred=['device_work_ns','inclusive_ns','host_ns','cross_rank_difference_ns','mean_ns'].find(function(k){return section.columns.indexOf(k)>=0;});
 states[section.id]={page:0,sort:preferred||section.columns[0],desc:!!preferred};
 if(section.id==='diagnosis')states[section.id].sort=section.columns.includes('occurrence_task_relative_ns')?'occurrence_task_relative_ns':'task_relative_ns';
 var sort=panel.querySelector('.sort');sort.value=states[section.id].sort||'';
 var direction=panel.querySelector('.direction');direction.textContent=states[section.id].desc?'降序':'升序';
 sort.onchange=function(){states[section.id].sort=sort.value;states[section.id].page=0;draw(section);};
 direction.onclick=function(){var s=states[section.id];s.desc=!s.desc;direction.textContent=s.desc?'降序':'升序';draw(section);};
 panel.querySelector('.previous').onclick=function(){states[section.id].page--;draw(section);};
 panel.querySelector('.next').onclick=function(){states[section.id].page++;draw(section);};
 if(section.kind==='memory'||section.kind==='monitor'){
  panel.querySelectorAll('.series-select,.time-low,.time-high').forEach(function(e){e.onchange=function(){drawCurve(section);};});
  panel.querySelector('.chart-peak').onclick=function(){if(states[section.id].peak)states[section.id].peak();};
 }
 draw(section);
});
['rank','layer','phase','shape','sid','step'].forEach(function(k){
 var options=new Set();Object.keys(data).forEach(function(id){data[id].forEach(function(row){var v=tokens(row,k);if(v&&v.length<160&&options.size<1000)options.add(v);});});
 options.forEach(function(v){var o=document.createElement('option');o.value=v;$('options-'+k).appendChild(o);});
});
document.querySelectorAll('.filters input').forEach(function(e){e.oninput=function(){clearTimeout(timers);timers=setTimeout(apply,120);};});
$('reset').onclick=function(){clearFilters();apply();};
$('print').onclick=function(){window.print();};
layerTree();
filterCaption();
var target=new URLSearchParams(location.search);if(target.has('section')&&/^\d+$/.test(target.get('row')||''))focusRow(target.get('section'),Number(target.get('row')),true);
document.documentElement.dataset.reportReady='true';
})();
