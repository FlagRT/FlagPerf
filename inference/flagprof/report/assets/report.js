var cfg=JSON.parse(document.getElementById('report-config').textContent), data={}, states={}, timers;
var $=function(id){return document.getElementById(id);};
var esc=function(s){return String(s).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];});};
document.querySelectorAll('.report-records').forEach(script=>{
 const id=script.dataset.section,batch=JSON.parse(script.textContent);
 function unpack(value){
   if(!Array.isArray(value))return value;
   const [layout,...fields]=value;
   if(layout===null)return fields.map(unpack);
   return Object.fromEntries(batch.keys[layout].map((key,i)=>[key,unpack(fields[i])]));
 }
 (data[id]??=[]).push(...batch.rows.map(unpack));script.remove();
});
function num(n){if(!Number.isFinite(n))return '无有限值';return n===0?'0':Number(n.toPrecision(6)).toString();}
function fmt(v,k){
 if(v===null||v===undefined)return cfg.missing_text[k]||('缺少'+(cfg.labels[k]||k||'该字段')+'记录');
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
function fieldText(row,key,unit=key){
 const value=val(row,key);
 if(value==null){
   if(row._field_notes?.[key])return row._field_notes[key];
   if(key==='generation_step'&&!(key in row))return '按完整计时范围汇总';
   return fmt(value,key);
 }
 return fmt(value,unit);
}
function val(row,k){return row[k]===undefined&&row.worst?row.worst[k]:row[k];}
function scalar(v){return v&&typeof v==='object'&&!Array.isArray(v)&&'value' in v?v.value:v;}
function detail(id,index){
 var row=data[id][index],body=$('detail-body');
 var section=cfg.sections.find(function(s){return s.id===id;});
 body.innerHTML='<h2>'+esc(section.title)+' · 详情与证据</h2><p>'+esc(section.description)+'</p>';
 var links=document.createElement('p');
 [['sid','查看同签名记录'],['sid_ref','查看参考侧签名'],['sid_cmp','查看比较侧签名'],['call_key','查看同调用记录']].forEach(function(p){if(row[p[0]]){var b=document.createElement('button');b.textContent=p[1];b.onclick=function(){related(row[p[0]],p[0]==='call_key'?'call_key':'sid',id,index);};links.appendChild(b);}});
 (Array.isArray(row.evidence)?row.evidence:[]).forEach(function(ref){if(!Array.isArray(ref))return;var target=(data['diagnosis.evidence']||[]).find(function(r){return JSON.stringify(r.identity)===JSON.stringify(ref);});if(target){var a=document.createElement('a');a.href='#diagnosis.evidence';a.textContent='查看触发事件 '+ref.join(' / ');a.onclick=function(e){e.preventDefault();focusRow('diagnosis.evidence',data['diagnosis.evidence'].indexOf(target),true);};links.appendChild(a);}});
 (row.input_sources_ref||[]).concat(row.input_sources_cmp||[]).filter(function(v,i,a){return v!=='unknown'&&a.indexOf(v)===i;}).forEach(function(key){var b=document.createElement('button');b.textContent='输入生产者：'+key;b.onclick=function(){related(key,'call_key',id,index);};links.appendChild(b);});
 (row._evidence_links||[]).forEach(function(link){var a=document.createElement('a');a.href=link.href;a.textContent=link.label;links.appendChild(a);});
 const copyBar=document.createElement('div');copyBar.className='copy-tools';
 const copies=[['复制全称',row.name],['复制 call_key',row.call_key]];
 if(id.startsWith('oracle.')&&row.call_key){
   const quoted="'"+row.call_key.replaceAll("'", "'\"'\"'")+"'";
   copies.push(['复制回放命令','flagprof replay prepare --ref "$REF_RUN" --cmp "$CMP_RUN" --call-key '+quoted+' --out ./replay-input']);
 }
 for(const [title,value] of copies)if(typeof value==='string'){
   const button=document.createElement('button');button.textContent=title;
   button.onclick=async()=>{try{await navigator.clipboard.writeText(value);button.textContent='已复制';}
     catch{const field=document.createElement('textarea');field.value=value;copyBar.append(field);field.select();if(document.execCommand('copy')){field.remove();button.textContent='已复制';}}};
   copyBar.append(button);
 }
 links.className='related-tools';
 const summary=document.createElement('div');summary.className='detail-metrics';
 for(const k of section.columns.filter(k=>Number.isFinite(scalar(val(row,k)))).slice(0,4)){
   const metric=document.createElement('div');metric.textContent=(cfg.labels[k]||k)+' · '+fmt(val(row,k),k);summary.append(metric);
 }
 body.append(summary,copyBar);
 body.appendChild(links);
 var dl=document.createElement('dl');
 Object.keys(row).filter(function(k){return k!=='source_result'&&k!=='_evidence_links'&&k!=='_field_notes';}).forEach(function(k){var dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=cfg.labels[k]||k;dd.textContent=row[k]==null?fieldText(row,k):fmt(row[k],k);dl.append(dt,dd);});
 body.appendChild(dl);
 var numeric=row.metrics||row,hist=numeric.ulp_histogram,quantiles=numeric.absolute_error_quantiles;
 function mini(values,title){var entries=Object.keys(values).map(function(k){return [k,values[k]];}).filter(function(p){return typeof p[1]==='number';});if(!entries.length)return;var max=Math.max.apply(null,entries.map(function(p){return p[1];}).concat([1e-300])),chart=document.createElement('div');chart.className='plot';chart.innerHTML='<h3>'+esc(title)+'</h3><div class=bars>'+entries.map(function(p){return '<div class=bar><span>'+esc(p[0])+'</span><span class=bar-track><span class=bar-fill style="display:block;width:'+100*p[1]/max+'%"></span></span><span>'+esc(fmt(p[1]))+'</span></div>';}).join('')+'</div>';body.appendChild(chart);}
 if(hist)mini(hist,'当前边界不同元素的 ULP 分布；相同元素不在桶中');
 if(quantiles)mini(quantiles,'当前有限有效元素的绝对误差分位数');

 var raw=document.createElement('details');raw.innerHTML='<summary>核对原始字段</summary>';var pre=document.createElement('pre');pre.textContent=JSON.stringify(row,null,2);raw.appendChild(pre);body.appendChild(raw);
 if(!$('row-detail').open)$('row-detail').show();
 $('row-detail').scrollTop=0;
}
document.addEventListener('click',function(e){var b=e.target.closest('button.detail');if(b)detail(b.dataset.section,Number(b.dataset.index));var link=e.target.closest('[data-focus-section]');if(link){e.preventDefault();focusRow(link.dataset.focusSection,Number(link.dataset.focusRow),true);}});
document.addEventListener('keydown',event=>{if(event.key==='Escape')$('row-detail').close();});
document.addEventListener('pointerdown',event=>{if($('row-detail').open&&!event.target.closest('#row-detail,button.detail,[data-point]'))$('row-detail').close();});
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
 var selected=[];rows.forEach(function(r,i){if(accepts(section,r))selected.push(i);});
 if(state.sort)selected.sort(function(i,j){var a=scalar(val(rows[i],state.sort)),b=scalar(val(rows[j],state.sort));if(a==null)return b==null?i-j:1;if(b==null)return -1;var d=typeof a==='number'&&typeof b==='number'?a-b:String(a).localeCompare(String(b),undefined,{numeric:true});return (state.desc?-d:d)||i-j;});
 state.selected=selected;
 var pages=Math.max(1,Math.ceil(selected.length/cfg.page_size));state.page=Math.min(state.page,pages-1);
 var start=state.page*cfg.page_size,visible=selected.slice(start,start+cfg.page_size),tbody=panel.querySelector(':scope > .table-scroll tbody');
 state.barKey=section.default_sort||(['max_abs','median_ns'].find(k=>section.columns.includes(k)));
 state.barMax=selected.reduce((max,i)=>Math.max(max,Math.abs(Number(scalar(val(rows[i],state.barKey)))||0)),0);
 tbody.innerHTML=visible.map(i=>{
   const fields=section.columns.filter(k=>state.expanded||!state.omit.has(k));
   return '<tr data-row="'+i+'">'+fields.map(k=>tableCell(section,state,rows[i],k,i)).join('')+
     '<td><button class="detail" data-section="'+esc(section.id)+'" data-index="'+i+'">查看详情与证据</button></td></tr>';
 }).join('');
 panel.querySelector('.row-count').textContent=(section.evidence_index?'证据样本筛选 ':'当前筛选 ')+selected.length+' / '+(section.evidence_index?'本页样本 ':'全表 ')+rows.length+' 条；显示 '+(visible.length?start+1:0)+'–'+(start+visible.length)+'。';
 panel.querySelectorAll(':scope > .table-scroll thead th').forEach((th,i)=>{th.hidden=state.omit.has(section.columns[i])&&!state.expanded;th.setAttribute('aria-sort',section.columns[i]===state.sort?(state.desc?'descending':'ascending'):'none');});
 const ignored=Object.keys(dimensions).filter(k=>selectedValue(k)&&!(section.filter_scope||[]).includes(k));
 if(ignored.length)panel.querySelector('.row-count').textContent+=' · '+ignored.map(k=>dimensions[k]).join('、')+'筛选未作用于本表';
 panel.querySelector('.page-number').textContent=(state.page+1)+' / '+pages;
 panel.querySelector('.previous').disabled=state.page===0;panel.querySelector('.next').disabled=state.page+1>=pages;
 relationship(section,selected);
 if(section.kind==='memory'||section.kind==='monitor')drawCurve(section);
}
function seriesFor(section){
 var result={},rows=data[section.id]||[];
 if(section.kind==='memory'){
  rows.forEach(function(r){if(r.source_time_ns==null)return;Object.keys(r.values||{}).forEach(function(metric){var id=[r.dev,r.component,metric].join(' / ');(result[id]||(result[id]={unit:r.unit===undefined?'B':r.unit,points:[]})).points.push([Number(r.source_time_ns),r.values[metric],r]);});});
  Object.keys(result).forEach(function(k){result[k].points.sort(function(a,b){return a[0]-b[0];});});
 }else{
  for(const r of rows)result[r.entity+' / '+r.metric]=monitorSeries(r,cfg.meta.monitor_window);
 }
 return result;
}
function axisValue(value,unit){return unit==='B'?fmt(value,'bytes'):fmt(value)+' '+unit;}
function drawCurve(section){
 if(section.id==='monitor.series')return monitorGrid(section);
 var panel=$(section.id),state=states[section.id],select=panel.querySelector('.series-select');
 if(!state.series){state.series=seriesFor(section);Object.keys(state.series).forEach(function(k){var o=document.createElement('option');o.value=k;o.textContent=k;select.appendChild(o);});}
 const allowed=new Set(state.selected.map(i=>data[section.id][i]));
 for(const opt of select.options)opt.hidden=!state.series[opt.value].points.some(p=>allowed.has(p[2]));
 if(select.selectedOptions[0]?.hidden)select.value=Array.from(select.options).find(o=>!o.hidden)?.value||'';
 var s=state.series[select.value];if(!s||!s.points.length){panel.querySelector('.timeline').replaceChildren();return;}
 var points=s.points.filter(p=>allowed.has(p[2])),origin=s.points[0][0],end=(points[points.length-1][0]-origin)/1e9;
 var loInput=panel.querySelector('.time-low'),hiInput=panel.querySelector('.time-high');
 if(state.curveKey!==select.value){loInput.value=0;hiInput.value=end||1;state.curveKey=select.value;}
 var lo=Number(loInput.value),hi=Number(hiInput.value);if(!(hi>lo))hi=lo+1;
 var phase=selectedValue('phase').trim().toLowerCase(),phaseBands=section.id==='resource.series'?(cfg.meta.phase_bands||[]).filter(function(b){return !phase||b.name.toLowerCase().includes(phase);}):[];
 var chosen=points.filter(function(p){var t=(p[0]-origin)/1e9;return t>=lo&&t<=hi;}).map(function(p){return phase&&section.id==='resource.series'&&!phaseBands.some(function(b){return p[0]>=b.start_ns&&p[0]<=b.end_ns;})?[p[0],null,p[2]]:p;}),valid=chosen.filter(function(p){return p[1]!=null&&Number.isFinite(p[1]);});
 var low=valid.reduce(function(v,p){return Math.min(v,p[1]);},0),high=valid.reduce(function(v,p){return Math.max(v,p[1]);},0);if(high===low)high=low+1;
 var x=function(p){return 65+850*((p[0]-origin)/1e9-lo)/(hi-lo);},y=function(p){return 210-170*(p[1]-low)/(high-low);},path='',on=false;
 chosen.forEach(function(p){if(p[1]==null){on=false;return;}path+=(on?'L':'M')+x(p)+','+y(p)+' ';on=true;});
 var bands=section.id==='resource.series'?(cfg.meta.phase_bands||[]).map(function(b){var a=Math.max(lo,(b.start_ns-origin)/1e9),z=Math.min(hi,(b.end_ns-origin)/1e9);return z>a?'<rect x="'+(65+850*(a-lo)/(hi-lo))+'" y="35" width="'+(850*(z-a)/(hi-lo))+'" height="180" fill="#eaf0f7"><title>'+esc(b.name)+'</title></rect>':'';}).join(''):'';
 var svg='<svg viewBox="0 0 950 260" role="img" aria-label="'+esc(select.value)+'">'+bands+'<text x="8" y="22">'+esc(axisValue(high,s.unit))+'</text><text x="8" y="216">'+esc(axisValue(low,s.unit))+'</text><path d="M65 35 V215 H920" stroke="#c5d3df" fill="none"/><path class="curve" d="'+path+'"/>';
 // Points selected for the plot retain their actual timestamps and values.
 var shown=valid.length<=2048?valid:valid.filter(function(p,i){return i%Math.ceil(valid.length/2048)===0||p[1]===high;});
 shown.forEach(function(p){const i=(data[section.id]||[]).indexOf(p[2]),tip=num((p[0]-origin)/1e9)+' s · '+axisValue(p[1],s.unit);
   svg+=mark(i,tip,'<circle class="sample" cx="'+x(p)+'" cy="'+y(p)+'" r="2"/>');});
 const marker=section.id==='resource.series'?cfg.meta.resource_fault_ns:null;
 if(Number.isFinite(marker)){
   const t=(marker-origin)/1e9;
   if(t>=lo&&t<=hi)svg+=line(x([marker]),32,x([marker]),215,'#b83227',2,'stroke-dasharray="4 3"')+label(x([marker]),28,'异常触发','middle');
   if(!panel.querySelector('.fault-focus')){
     const btn=document.createElement('button');btn.className='fault-focus';btn.textContent='聚焦故障前窗口';
     btn.onclick=()=>{const series=state.series[select.value];const end=(marker-series.points[0][0])/1e9;hiInput.value=end;loInput.value=Math.max(0,end-Math.max(1,end*0.15));drawCurve(section);};panel.querySelector('.chart-controls').append(btn);
   }
 }
 svg+='<text x="65" y="245">'+esc(num(lo))+' s</text><text x="835" y="245">'+esc(num(hi))+' s</text></svg><p class="chart-readout">来源时间相对首点；当前窗口 '+valid.length+' 个有效显示点。悬停查看原始观测；精确统计以统计表为准。</p>';
 panel.querySelector('.timeline').innerHTML=svg;
 state.peak=function(){var p=points.filter(function(x){return x[1]!=null&&(!phase||section.id!=='resource.series'||phaseBands.some(function(b){return x[0]>=b.start_ns&&x[0]<=b.end_ns;}));}).reduce(function(a,b){return !a||b[1]>a[1]?b:a;},null);if(!p)return;var t=(p[0]-origin)/1e9,span=Math.max(end*.05,1e-6);loInput.value=Math.max(0,t-span);hiInput.value=Math.min(end||1,t+span);drawCurve(section);};
}
function layerTree(){
 var panel=$('perf.layer');if(!panel)return;var rows=data['perf.layer']||[],box=document.createElement('div');box.className='layer-tree';
 var byName={};rows.forEach(function(r){byName[r.name]=r;});
 var children={},roots=[];rows.forEach(function(r){var parents=(r.parent_names||[]).filter(function(n){return n!==r.name&&byName[n];});if(parents.length!==1)roots.push(r);else(children[parents[0]]||(children[parents[0]]=[])).push(r);});
 var seen=new Set();function node(r,parent){if(seen.has(r.name))return;seen.add(r.name);var details=document.createElement('details'),summary=document.createElement('summary');summary.textContent=r.name+' · '+fmt(r.inclusive_ns==null?r.host_ns:r.inclusive_ns,'inclusive_ns');if(Number.isFinite(r.exclusive_ns)&&r.inclusive_ns>0){const share=document.createElement('meter');share.max=r.inclusive_ns;share.value=r.exclusive_ns;summary.append(' ',share,' 独占 '+fmt(r.exclusive_ns,'exclusive_ns')+' · 设备工作量 '+fmt(r.device_work_ns,'device_work_ns'));}details.appendChild(summary);var b=document.createElement('button');b.textContent='筛选此层';b.onclick=function(){if($('filter-layer')){$('filter-layer').value=r.name;apply();}};if($('filter-layer'))details.appendChild(b);(children[r.name]||[]).forEach(function(c){node(c,details);});parent.appendChild(details);}
 var label=document.createElement('p');label.textContent='层次来自已记录的父层关系；未提供关系的层独立列出。父子耗时不能相加。';box.appendChild(label);roots.forEach(function(r){node(r,box);});rows.forEach(function(r){node(r,box);});panel.insertBefore(box,panel.querySelector('.table-tools'));
}
function filterCaption(){var filters=Array.from(document.querySelectorAll('.filters input,.filters select')).filter(function(e){return e.value.trim();}).map(function(e){return e.parentNode.textContent.trim()+'='+e.value;});$('print-scope').textContent='打印保留当前筛选及各表当前页；'+(filters.length?filters.join('；'):'未设置筛选')+'。每表显示范围与全表条数见表前说明；完整明细见 HTML 与导出。';}
function clearFilters(){document.querySelectorAll('.filters input,.filters select').forEach(function(e){e.value='';});}
function updateNavigation(){
 for(const {box,items} of workspaceTabs.values()){
   const hits=items.reduce((n,s)=>n+states[s.id].selected.length,0),total=items.reduce((n,s)=>n+(data[s.id]||[]).length,0);
   box.hidden=!hits&&total>0;
   const link=document.querySelector('.sidebar a[href="#'+CSS.escape(items[0].id)+'"] small');
   if(link)link.textContent=hits===total?total:hits+' / '+total;
 }
}
function apply(){filterCaption();cfg.sections.forEach(function(s){states[s.id].page=0;draw(s);});updateNavigation();}
function focusRow(id,index,clear){
 var section=cfg.sections.find(function(s){return s.id===id;});if(!section||!data[id]||!data[id][index])return false;
 activate(id);if(clear){clearFilters();apply();}var panel=$(id),state=states[id],position=state.selected.indexOf(index);if(position<0)return false;
 state.page=Math.floor(position/cfg.page_size);draw(section);if(panel.tagName==='DETAILS')panel.open=true;
 document.querySelectorAll('tr.focused').forEach(function(r){r.classList.remove('focused');});var node=panel.querySelector(':scope > .table-scroll tr[data-row="'+index+'"]');node.classList.add('focused');node.scrollIntoView({block:'center'});detail(id,index);return true;
}
function related(key,kind,currentId,currentIndex){
 $('row-detail').close();clearFilters();$('search').value=key;apply();
 var candidates=[];cfg.sections.forEach(function(s){(data[s.id]||[]).forEach(function(r,i){if(s.id===currentId&&i===currentIndex)return;if(kind==='call_key'?r.call_key===key:[r.sid,r.sid_ref,r.sid_cmp].includes(key))candidates.push([s.id,i]);});});
 candidates.sort(function(a,b){return (a[0]===currentId?1:0)-(b[0]===currentId?1:0);});
 if(candidates.length)focusRow(candidates[0][0],candidates[0][1],false);else{var panel=cfg.sections.find(function(s){return states[s.id].selected.length;});if(panel)$(panel.id).scrollIntoView({block:'start'});}
}
initChartEvents();
initFilters();
setupWorkspaces();
window.addEventListener('hashchange',revealHash);
document.querySelectorAll('a[href^="#"]').forEach(a=>a.addEventListener('click',()=>activate(a.hash.slice(1))));
cfg.sections.forEach(function(section){
 var panel=$(section.id);
 states[section.id]={page:0,sort:section.default_sort,desc:section.default_desc};
 columnTools(section,states[section.id]);
 var sort=panel.querySelector('.sort');sort.value=states[section.id].sort||'';
 var direction=panel.querySelector('.direction');direction.textContent=states[section.id].desc?'降序':'升序';
 const restore=document.createElement('button');restore.textContent='恢复默认顺序';direction.after(restore);
 restore.onclick=()=>{Object.assign(states[section.id],{sort:section.default_sort,desc:section.default_desc,page:0});sort.value=section.default_sort||'';direction.textContent=section.default_desc?'降序':'升序';draw(section);};
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
document.querySelectorAll('.filters input,.filters select').forEach(function(e){e.oninput=function(){clearTimeout(timers);timers=setTimeout(apply,120);};});
$('reset').onclick=function(){clearFilters();apply();};
$('print').onclick=function(){window.print();};
layerTree();
updateNavigation();
revealHash();
filterCaption();
var target=new URLSearchParams(location.search);if(target.has('section')&&/^\d+$/.test(target.get('row')||''))focusRow(target.get('section'),Number(target.get('row')),true);
document.documentElement.dataset.reportReady = 'true';
