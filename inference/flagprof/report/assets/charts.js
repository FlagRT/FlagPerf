// Each datum keeps its source row ordinal. Axes use the units of that row.
const palette={compute:'#1D6FA5',memcpy:'#D97706',comm:'#0F766E',wait:'#7C6F9F'};
function picker(state,key,values,label,display=v=>cfg.labels[v]||fmt(v)){
 if(!values.includes(state[key]))state[key]=values[0];
 if(values.length<2)return '';
 return '<label>'+esc(label)+' <select data-chart-choice="'+key+'">'+values.map(v=>'<option '+(state[key]===v?'selected ':'')+'value="'+esc(v)+'">'+esc(display(v))+'</option>').join('')+'</select></label> ';
}
function svgFrame(body,height,label){
 return '<svg class="relationship" viewBox="0 0 840 '+height+'" role="img" aria-label="'+esc(label)+'">'+body+'</svg>';
}
function label(x,y,value,anchor='start'){
 return '<text x="'+x+'" y="'+y+'" text-anchor="'+anchor+'">'+esc(value)+'</text>';
}
function line(x,y,end,bottom,color,width=1,extra=''){
 return '<line x1="'+x+'" y1="'+y+'" x2="'+end+'" y2="'+bottom+'" stroke="'+color+'" stroke-width="'+width+'" '+extra+'/>';
}
function mark(index,tip,shape){
 return '<g data-point="'+index+'" data-tip="'+esc(tip)+'" tabindex="0">'+shape+'</g>';
}
function metricChartRow(index,name,lines,reading,tip){
 return '<button type="button" class="metric-chart-row" data-point="'+index+'" data-tip="'+esc(tip)+'" aria-label="'+esc(name+' · '+tip)+'">'+
   '<span class="metric-chart-label">'+esc(name)+'</span><svg aria-hidden="true">'+lines+'</svg>'+
   '<span class="metric-chart-reading">'+esc(reading)+'</span></button>';
}
function metricChart(title,rows,axis=''){
 return '<div class="metric-chart" role="group" aria-label="'+esc(title)+'">'+rows+axis+'</div>';
}
function baselineLabel(r){
 const parts=['rank '+(r.rank??'—'),fmt(r.phase)];
 if(r.instrumentation!=null)parts.push(fmt(r.instrumentation,'instrumentation'));
 if(r.generation_step!=null)parts.push('step '+r.generation_step);
 const shape=r.shape,ids=shape?.samples?.samples??shape?.samples;
 if(shape?.batch!=null)parts.push('批量 '+fmt(shape.batch));
 if(shape?.input_tokens!=null)parts.push('输入 token 数 '+fmt(shape.input_tokens));
 if(Array.isArray(ids)&&ids.length&&ids.every(v=>typeof v==='string'||typeof v==='number')){
   const first=Number(ids[0]);
   const consecutive=ids.length>2&&Number.isSafeInteger(first+ids.length-1)&&ids.every((v,i)=>String(v)===String(first+i));
   const samples=consecutive?ids[0]+'–'+ids[ids.length-1]:
     ids.slice(0,4).join('、')+(ids.length>4?'…（共 '+ids.length+' 个）':'');
   parts.push('样本 '+samples);
 }else if(shape!=null&&shape.batch==null&&shape.input_tokens==null){
   parts.push(brief(shape,'shape'));
 }
 return parts.join(' · ');
}
function trend(rows,series,options={}){
 if(!rows.length)return '';
 const values=[];
 for(const item of rows)for(const s of series){
   const value=s.get(item.r);if(Number.isFinite(value)&&(!options.log||value>0))values.push(value);
 }
 const convert=v=>options.log?Math.log10(v):v;
 let min=values.reduce((n,v)=>Math.min(n,convert(v)),Infinity),max=values.reduce((n,v)=>Math.max(n,convert(v)),-Infinity);
 if(!Number.isFinite(min)){min=0;max=1;}
 if(!options.log)min=Math.min(0,min);
 const y=v=>options.log&&v<=0?178:158-133*(convert(v)-min)/(max-min||1);
 const xValues=rows.map((item,i)=>options.x?options.x(item.r):i);
 const xMin=xValues.reduce((n,v)=>Math.min(n,v),Infinity),xMax=xValues.reduce((n,v)=>Math.max(n,v),-Infinity);
 const x=i=>72+735*(xValues[i]-xMin)/(xMax-xMin||1);
 let body='';
 for(let k=0;k<=3;k++){
   const v=min+(max-min)*k/3;
   body+=line(70,158-133*k/3,815,158-133*k/3,'#e2e8f0')+label(65,162-133*k/3,fmt(options.log?10**v:v,options.unit),'end');
 }
 if(options.log)body+=line(72,178,815,178,'#94a3b8',1,'stroke-dasharray="3 4"')+label(65,181,'≤0','end');
 for(const s of series){
   let path='',active=false,dots='';
   rows.forEach(({r,i},j)=>{
     const v=s.get(r);
     if(!Number.isFinite(v)){active=false;return;}
     path+=(active?'L':'M')+x(j)+' '+y(v)+' ';active=true;
     dots+=mark(i,options.name(r)+' · '+s.name+' '+fmt(v,options.unit),'<circle cx="'+x(j)+'" cy="'+y(v)+'" r="3.4" fill="'+s.color+'"/>');
   });
   body+='<path d="'+path+'" fill="none" stroke="'+s.color+'" stroke-width="2"/>'+dots;
 }
 rows.forEach(({r},j)=>{
   if(j%Math.max(1,Math.ceil(rows.length/7))===0)body+=label(x(j),200,options.name(r).slice(-19),'middle');
   if(options.marker?.(r))body+=line(x(j),24,x(j),178,'#b83227',1,'stroke-dasharray="4 3"')+label(x(j),17,'首个差异','middle');
   if(options.jump?.(r))body+=line(x(Math.max(0,j-1)),168,x(j),168,'#d97706',4);
 });
 return svgFrame(body,212,series.map(s=>s.name).join(' / '));
}
function comparisonPlot(items,state){
 const controls=picker(state,'view',['条件透视','原始长表'],'表格');
 const keys=['source_label','rank','scope','phase','generation_step','shape'];
 const groups=new Map(),metrics=new Set();
 for(const {r,i} of items){
   const key=JSON.stringify(keys.map(k=>r[k]));
   if(!groups.has(key))groups.set(key,{r,values:new Map()});
   groups.get(key).values.set(r.metric,{r,i});metrics.add(r.metric);
 }
 let pivot='';
 if(state.view==='条件透视'){
   pivot='<div class="table-scroll"><table><thead><tr><th>来源 · rank · 阶段 · 规模</th>'+Array.from(metrics,m=>'<th>'+esc(fmt(m,'metric'))+'</th>').join('')+'</tr></thead><tbody>';
   for(const {r,values} of Array.from(groups.values()).slice(state.page*cfg.page_size,(state.page+1)*cfg.page_size)){
     pivot+='<tr><th>'+esc(keys.filter(k=>r[k]!=null).map(k=>fmt(r[k],k)).join(' · '))+'</th>';
     for(const m of metrics){
       const entry=values.get(m);
       if(!entry){pivot+='<td>未记录</td>';continue;}
       const item=entry.r;
       pivot+='<td><button data-point="'+entry.i+'">'+esc(fmt(item.baseline,m)+' → '+fmt(item.candidate,m)+' · '+fmt(item.relative_change,'relative_change'))+'</button></td>';
     }
     pivot+='</tr>';
   }
   pivot+='</tbody></table></div>';
 }
 const selected=items.filter(({r})=>Number.isFinite(r.relative_change)).slice(0,12);
 const scale=selected.reduce((m,{r})=>Math.max(m,Math.abs(r.relative_change)),0)||1;
 const rows=selected.map(({r,i})=>{
   const v=r.relative_change,end=(50+50*v/scale)+'%';
   const throughput=/per_second|throughput/.test(r.metric),latency=/_ns$/.test(r.metric);
   const color=(!throughput&&!latency)||v===0?'#64748b':(throughput?v>0:v<0)?'#18794e':'#b83227';
   const reading=fmt(v,'relative_change')+' · '+fmt(r.baseline,r.metric)+' → '+fmt(r.candidate,r.metric);
   return metricChartRow(i,'rank '+(r.rank??'—')+' · '+fmt(r.metric,'metric'),
     line('50%',0,'50%',18,'#94a3b8')+line('50%',9,end,9,color,9),reading,reading);
 }).join('');
 return controls+(rows?metricChart('变化率，以 0% 为中心',rows):'')+pivot;
}
function baselineDistribution(items){
 const keys=['minimum_ns','median_ns','p95_ns','maximum_ns'];
 const rows=items.filter(({r})=>keys.every(k=>Number.isFinite(r[k]))).slice(0,12);
 if(!rows.length)return '';
 const end=rows.reduce((max,{r})=>Math.max(max,r.maximum_ns),0);
 const x=v=>(end?100*v/end:0)+'%';
 const body=rows.map(({r,i})=>{
   const stats=keys.map(k=>cfg.labels[k]+' '+fmt(r[k],k));
   const lines=line(0,9,'100%',9,'#e2e8f0')+line(x(r.minimum_ns),9,x(r.maximum_ns),9,'#64748b',2)+
     line(x(r.median_ns),9,x(r.p95_ns),9,'#155e8a',7)+line(x(r.median_ns),2,x(r.median_ns),16,'#10263a',2);
   return metricChartRow(i,baselineLabel(r),lines,stats[1],stats.join(' · '));
 }).join('');
 const axis='<div class="metric-chart-axis"><div><span>0 ns</span><span>'+esc(fmt(end,'duration_ns'))+'</span></div></div>';
 return metricChart('最小值、中位数、p95 与最大值',body,axis);
}
function forwardLayerRanking(items,state,sampled){
 const metricPicker=picker(state,'layerMetric',['exclusive_ns','inclusive_ns'],'耗时口径');
 const metric=state.layerMetric,metricName=cfg.labels[metric];
 const groups=new Map();
 for(const item of items){
   const {run_id,forward,root_corr,phase}=item.r;
   const key=JSON.stringify([run_id,forward,root_corr,phase]);
   if(!groups.has(key))groups.set(key,[]);
   groups.get(key).push(item);
 }
 function groupLabel(key){
   const [run,forward,root,phase]=JSON.parse(key);
   const label=forward==null||forward==='unknown'?'前向序号未记录':'前向 '+forward;
   return [label,phase&&phase!=='unknown'?fmt(phase):null,run,root!=null?'根调用 '+root:null].filter(Boolean).join(' · ');
 }
 const groupPicker=picker(state,'layerForward',[...groups.keys()],'前向范围',groupLabel);
 const group=groups.get(state.layerForward)||[];
 const ranked=group.filter(({r})=>r.name!=='<root>'&&Number.isFinite(r[metric])&&r[metric]>=0)
   .sort((a,b)=>b.r[metric]-a.r[metric]||a.i-b.i).slice(0,10);
 const max=ranked[0]?.r[metric]||1;
 const bars=ranked.map(({i,r})=>'<button type="button" class="bar" data-point="'+i+'">'+
   '<span class="bar-label" title="'+esc(r.name)+'">'+esc(r.name)+'</span>'+
   '<span class="bar-track"><span class="bar-fill" style="width:'+100*(r[metric]/max)+'%"></span></span>'+
   '<span>'+esc(fmt(r[metric],metric))+'</span></button>').join('');
 const roots=group.filter(({r})=>r.name==='<root>').map(({i,r})=>
   '<button type="button" data-point="'+i+'">整模型 &lt;root&gt; · '+esc(metricName)+' '+esc(fmt(r[metric],metric))+'</button>');
 const scope=groups.size===1?'<p class="description">'+esc(groupLabel(state.layerForward))+'</p>':'';
 const definition=metric==='exclusive_ns'?'扣除与已记录直接子层重叠的区间。':'包含子层执行时间，父子层的数值不能相加。';
 return '<div class="layer-ranking"><h3>'+esc(metricName)+' Top 10</h3><div class="chart-controls">'+metricPicker+groupPicker+'</div>'+scope+
   '<p class="description">'+(sampled?'范围为本页证据样本。':'')+'按当前筛选内各层的累计时间排列。'+definition+'选择一行可查看对应明细。</p>'+
   (bars?'<div class="bars">'+bars+'</div>':'<p>当前范围没有可排名的子层耗时记录。</p>')+
   (roots.length?'<p>'+roots.join(' ')+'</p>':'')+'</div>';
}
function relationship(section,indices){
 const id=section.id,panel=$(id),plot=panel.querySelector('.plot');if(!plot)return;
 const state=states[id],items=indices.map(i=>({i,r:data[id][i]}));
 let controls='',graphic='',note='';
 if(id==='perf.layer.forwards'){
   graphic=forwardLayerRanking(items,state,section.evidence_index);
 }else if(id==='comparison.metrics'){
   graphic=comparisonPlot(items,state);note='变化率以基准值为分母。绿色表示时延下降或吞吐上升。';
   panel.querySelector(':scope > .table-scroll').hidden=state.view==='条件透视';
 }else if(id==='oracle.layer'||id==='oracle.three_way'){
   const three=id==='oracle.three_way';
   const metrics=three?[...new Set(items.map(o=>o.r.metric))]:['relative_l2','max_abs','cosine_similarity','differing_fraction'];
   controls=picker(state,'metric',metrics,'指标')+picker(state,'axis',['线性','对数'],'纵轴');
   let rows=items,series;
   if(three){
     rows=items.filter(o=>o.r.metric===state.metric);
     controls+=picker(state,'sample',[...new Set(rows.map(o=>o.r.sample))],'样本');
     rows=rows.filter(o=>o.r.sample===state.sample);
     const boundary=r=>JSON.stringify([r.scope,r.boundary]);
     const layerOrder=new Map((cfg.meta.boundary_order||[]).map(r=>[boundary(r),r.order]));
     if(layerOrder.size)rows.sort((a,b)=>(layerOrder.get(boundary(a.r))??Infinity)-(layerOrder.get(boundary(b.r))??Infinity));
     series=[{name:'参考→基线',color:'#155e8a',get:r=>r.reference_to_baseline},{name:'参考→候选',color:'#0f766e',get:r=>r.reference_to_candidate}];
     note='蓝线为参考→基线，青线为参考→候选。'+(layerOrder.size?'边界按参考侧已记录执行序排列，未登记顺序的边界附后。':'横轴沿用封存记录顺序。');
   }else{
     rows=items.slice().sort((a,b)=>(a.r.order??a.i)-(b.r.order??b.i));
     series=[{name:fmt(state.metric,'metric'),color:'#155e8a',get:r=>scalar(val(r,state.metric))}];
     note='按已记录的边界执行序展示。聚合指标的最差样本见点位详情。';
   }
   const first=(data['oracle.first_divergence']||[])[0],jump=(data['oracle.propagation']||[]).reduce((a,b)=>Number.isFinite(b.ratio)&&(!a||b.ratio>a.ratio)?b:a,null);
   graphic=trend(rows,series,{name:r=>String(r.scope).replace('<root>.layers.','L')+' '+r.boundary,unit:state.metric,
     log:state.axis==='对数',marker:r=>first&&first.scope===r.scope&&r.boundary===first.boundary,
     jump:r=>jump&&String(jump.after).startsWith(r.scope+'#')});
 }else if(id==='communication.aligned'){
   controls=picker(state,'domain',[...new Set(items.map(o=>o.r.group??'未记录通信域'))],'通信域');
   let rows=items.filter(o=>(o.r.group??'未记录通信域')===state.domain&&Number.isFinite(o.r.group_seq)).sort((a,b)=>a.r.group_seq-b.r.group_seq);
   if(rows.length>500){
     const reduced=[];for(let i=0;i<rows.length;i+=Math.ceil(rows.length/240)){
       const bucket=rows.slice(i,i+Math.ceil(rows.length/240));
       const peak=bucket.reduce((a,b)=>b.r.cross_rank_difference_ns>a.r.cross_rank_difference_ns?b:a);
       reduced.push(...new Set([bucket[0],peak,bucket[bucket.length-1]]));
     }rows=reduced;
   }
   graphic=trend(rows,[{name:'占用差',color:'#155e8a',get:r=>r.cross_rank_difference_ns}],{unit:'duration_ns',x:r=>r.group_seq,name:r=>'序号 '+r.group_seq});
   const top=rows.reduce((a,b)=>!a||b.r.cross_rank_difference_ns>a.r.cross_rank_difference_ns?b:a,null);
   if(top&&Array.isArray(top.r.rank_device_union_ns))graphic+=rankBars(top);
   note='各 rank 设备活动并集的最大值减最小值。点位保留来源通信序号和各 rank 占用。';
 }else if(id==='performance.baseline'){
   graphic=baselineDistribution(items);
   const steps=items.filter(o=>Number.isInteger(o.r.generation_step));
   if(steps.length){
     const cohort=r=>[r.source,r.instrumentation,r.rank,r.scope,fmt(r.shape,'shape')].join(' · ');
     controls+=picker(state,'cohort',[...new Set(steps.map(o=>cohort(o.r)))],'生成条件');
     graphic+=trend(steps.filter(o=>cohort(o.r)===state.cohort).sort((a,b)=>a.r.generation_step-b.r.generation_step),
       [{name:'中位数',color:'#155e8a',get:r=>r.median_ns},{name:'p95',color:'#0f766e',get:r=>r.p95_ns}],
       {unit:'duration_ns',x:r=>r.generation_step,name:r=>'step '+r.generation_step});
   }
   note='细线从最小值延伸到最大值，粗线从中位数延伸到 p95。选择一行可查看完整输入形状和时延。';
 }else if(id==='perf.gap'){
   for(const {r,i} of items.slice(0,8)){
     if(!(r.window_ns>0))continue;
     let body='',n=0;
     for(const [key,title,color] of [['all_activity_union_ns','活动并集 / 空白','#155e8a'],...Object.keys(palette).map(k=>[k+'_union_ns',fmt(k),palette[k]])]){
       if(!Number.isFinite(r[key]))continue;
       const y=18+23*n++,share=r[key]/r.window_ns;
       body+=mark(i,title+' '+fmt(r[key],key),label(4,y,title)+line(160,y-5,690,y-5,'#e2e8f0',10)+line(160,y-5,160+530*share,y-5,color,10)+label(700,y,fmt(share,'time_share')));
     }
     graphic+=svgFrame(body,Math.max(25,n*23),fmt(r.kind)+' · '+fmt(r.window_ns,'window_ns'));
   }
   note='活动并集与空白覆盖同一完整窗口。四类活动各自以窗口为分母，区间可重叠。';
 }else if(id==='perf.device'||id==='perf.op'){
   const totals=new Map(),key=items.some(o=>Number.isFinite(o.r.device_work_ns))?'device_work_ns':'host_ns';
   for(const {r} of items)totals.set(r.category,(totals.get(r.category)||0)+(r[key]||0));
   const rows=items.filter(o=>Number.isFinite(o.r[key])).sort((a,b)=>b.r[key]-a.r[key]).slice(0,12);
   const max=rows[0]?.r[key]||1;
   const body=rows.map(({r,i})=>{
     const reading=fmt(r[key],key)+' · '+(totals.get(r.category)?fmt(r[key]/totals.get(r.category),'time_share'):'分母为零');
     return metricChartRow(i,r.name,line(0,9,(100*r[key]/max)+'%',9,palette[r.category]||'#155e8a',10),reading,reading);
   }).join('');
   if(body)graphic=metricChart('热点与类别内占比',body);
   note='按当前筛选内同类别的工作量求占比。设备工作量包含多流重叠。';
 }else if(id==='route.coverage'){
   const totals=new Map();
   for(const {r} of items)if(Number.isFinite(r.calls))totals.set(r.route_decision,(totals.get(r.route_decision)||0)+r.calls);
   let x=0,body='',sum=[...totals.values()].reduce((a,b)=>a+b,0);
   for(const [key,v] of totals){
     const next=x+800*v/sum;if(sum)body+=line(x,20,next,20,key==='hit'?'#18794e':key==='fallback'?'#d97706':'#94a3b8',18);
     x=next;
   }
   graphic=svgFrame(body,40,'路由调用分布');note=[...totals].map(([k,v])=>fmt(k)+' '+v+' 次').join(' · ');
 }else if(id==='diagnosis'){
   const events=[];
   for(const {r,i} of items){
     const when=r.occurrence_task_relative_ns??r.task_relative_ns;
     if(Number.isFinite(when))events.push({t:when,i,name:fmt(r.rule)});
     if(Number.isFinite(r.first_detected_relative_ns))events.push({t:r.first_detected_relative_ns,i,name:'首次检出'});
   }
   if(events.length){
     events.push({t:0,i:events[0].i,name:'任务启动'});events.sort((a,b)=>a.t-b.t);
     const end=Math.max(1,events[events.length-1].t);let body=line(75,25,755,25,'#94a3b8',2);
     events.forEach(({i,t,name},j)=>{
       const x=75+680*t/end,y=52+22*j;
       body+=mark(i,name+' · '+fmt(t,'duration_ns'),line(x,25,x,y-8,'#cbd5e1')+'<circle cx="'+x+'" cy="25" r="4" fill="#b83227"/>'+label(x,y,name+' '+fmt(t,'duration_ns'),'middle'));
     });graphic=svgFrame(body,66+events.length*22,'任务内事件时间线');
   }
   note='按可映射的任务内时刻排列，原始事件时钟和检出记录见详情。';
 }else if(id==='generation.steps'){
   controls=picker(state,'request',[...new Set(items.map(o=>o.r.request_id))],'请求');
   const rows=items.filter(o=>o.r.request_id===state.request&&Number.isInteger(o.r.generation_step)).sort((a,b)=>a.r.generation_step-b.r.generation_step);
   graphic=trend(rows,[{name:'采集中的主机推进耗时',color:'#155e8a',get:r=>r.host_duration_ns}],{unit:'duration_ns',x:r=>r.generation_step,name:r=>'step '+r.generation_step});
   note='使用当前请求各生成步的主机推进区间。';
 }else if(id==='group.ranks'){
   graphic=items.map(({r,i})=>'<button data-point="'+i+'">rank '+esc(r.rank)+' · '+esc(fmt(r.status))+'</button>').join(' ');
 }else{bars(panel,section,indices);return;}
 plot.innerHTML=controls+graphic+(graphic&&note?'<p class="description">'+esc(note)+'</p>':'');
 plot.querySelectorAll('[data-chart-choice]').forEach(select=>select.onchange=()=>{const key=select.dataset.chartChoice;state[key]=typeof state[key]==='number'?Number(select.value):select.value;relationship(section,indices);});
}
function rankBars({r,i}){
 const values=r.rank_device_union_ns,max=values.reduce((m,v)=>Math.max(m,v||0),1);
 const body=values.map((v,rank)=>Number.isFinite(v)?mark(i,'序号 '+r.group_seq,label(4,18+24*rank,'rank '+rank)+line(100,12+24*rank,100+540*v/max,12+24*rank,'#155e8a',9)+label(655,18+24*rank,fmt(v,'duration_ns'))):'').join('');
 return svgFrame(body,values.length*24+5,'最大占用差调用的各 rank 占用');
}

function initChartEvents(){
 const tooltip=document.createElement('div');tooltip.className='tooltip';tooltip.hidden=true;document.body.append(tooltip);
 document.addEventListener('pointermove',event=>{
   const point=event.target.closest('[data-tip]');
   tooltip.hidden=!point;if(!point)return;
   tooltip.textContent=point.dataset.tip;
   tooltip.style.left=Math.max(2,Math.min(event.clientX+12,window.innerWidth-tooltip.offsetWidth-4))+'px';
   tooltip.style.top=Math.max(2,Math.min(event.clientY+12,window.innerHeight-tooltip.offsetHeight-4))+'px';
 });
 const open=event=>{
   const point=event.target.closest('[data-point]');if(!point)return;
   const panel=point.closest('.panel');
   if(panel){
     tooltip.hidden=true;
     if(point.closest('.layer-ranking'))focusRow(panel.id,Number(point.dataset.point),false);
     else detail(panel.id,Number(point.dataset.point));
   }
 };
 document.addEventListener('click',open);
 document.addEventListener('keydown',e=>{if(e.key==='Enter'&&e.target.matches('g[data-point]'))open(e);});
}
