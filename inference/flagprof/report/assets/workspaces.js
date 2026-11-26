// Section ordinals are stable across filtering, sorting, tabs and deep links.
const dimensions = {
 dev:'设备',diff_status:'差异状态', rank:'rank', layer:'层 / 模块', phase:'阶段', shape:'输入规模', sid:'签名', step:'生成步',
 sample:'样本', level:'证据等级', metric:'指标', category:'事件类别', route_decision:'路由',
 entity:'设备实体', group:'通信域', comparison_status:'可比性', rule:'异常类型', state:'事件状态',
 introduced:'同输入引入', equal:'条件一致', instrumentation:'采集条件', scope:'计时范围'
};
dimensions.task_label='所属任务';
const workspaceTabs = new Map();
function valuesFor(id,row,key) {
 let values;
 switch(key) {
 case 'layer':
   values=[...(Array.isArray(row.scopes)?row.scopes:[])];
   if (/^perf\.layer(?:$|\.)/.test(id)) values.push(row.name);
   if (/^(oracle\.|perf\.(op|device))/.test(id) && id!=='oracle.identity_diff') values.push(row.scope);
   break;
 case 'scope': values=/^(performance\.|comparison\.metrics)/.test(id)?[row.scope]:[];break;
 case 'diff_status':
   values=id.startsWith('oracle.')?[row.introduced===true?'同输入引入':row.digest_equal===true?'相等':row.digest_equal===false?'不等':null]:[];break;
 case 'level':values=Array.isArray(row.levels)?row.levels:[row.level];break;
 case 'sid': values=[row.sid,row.sid_ref,row.sid_cmp];break;
 case 'phase':
   values=[row.phase,id==='perf.gap'?row.kind:null];
   if(id==='resource'||id==='resource.stages')values.push(row.name);
   if(id==='resource.series')for(const band of cfg.meta.phase_bands||[])if(row.source_time_ns>=band.start_ns&&row.source_time_ns<=band.end_ns)values.push(band.name);
   break;
 case 'step': values=[row.generation_step??row.step];break;
 case 'shape': values=[row.input_shape??row.shape];break;
 default: values=[row[key]];
 }
 return values.filter(v=>v!=null && v!=='').map(v=>typeof v==='object'?fmt(v,key):String(v));
}
function initFilters(){
 for(const [key,label] of Object.entries(dimensions)){
   const options=new Set();
   for(const section of cfg.sections){
     if(!section.filter_scope.includes(key))continue;
     const rows=data[section.id]||[];
     let supported=false;
     for(const row of rows){
       const values=valuesFor(section.id,row,key);
       if(values.length){supported=true;for(const v of values)options.add(v);}
     }
     if(!supported)section.filter_scope=section.filter_scope.filter(k=>k!==key);
   }
   if(options.size<2)continue;
   const wrapper=document.createElement('label'),select=document.createElement('select');
   wrapper.append(label,select);select.id='filter-'+key;select.add(new Option('全部',''));
   for(const v of Array.from(options).sort((a,b)=>a.localeCompare(b,undefined,{numeric:true})))select.add(new Option(fmt(v,key),v));
   select.onchange=apply;$('dimensions').append(wrapper);
 }
}
function selectedValue(key){return $('filter-'+key)?.value||'';}
function accepts(section,row){
 const query=$('search').value.trim().toLowerCase();
 if(query&&!JSON.stringify(row).toLowerCase().includes(query))return false;
 return (section.filter_scope||[]).every(key=>!selectedValue(key)||valuesFor(section.id,row,key).includes(selectedValue(key)));
}
function setupWorkspaces(){
 for(const section of cfg.sections){
   const panel=$(section.id);
   let tabs=workspaceTabs.get(section.group);
   if(!tabs){
     const box=document.createElement('section'),title=document.createElement('h2'),bar=document.createElement('div');
     title.textContent=section.group_title;box.className='workspace';bar.className='tabs';
     box.append(title,bar);panel.before(box);tabs={box,bar,items:[]};workspaceTabs.set(section.group,tabs);
   }
   tabs.box.append(panel);tabs.items.push(section);
   const button=document.createElement('button');button.textContent=section.tab_label;button.dataset.tab=section.id;
   button.onclick=()=>activate(section.id);tabs.bar.append(button);
 }
 for(const {items,bar} of workspaceTabs.values()){
   bar.hidden=items.length===1;
   if(items.length===1&&items[0].title===items[0].group_title)$(items[0].id).querySelector('h2')?.setAttribute('hidden','');
   const first=items.find(s=>['diagnosis','oracle.op_sensitivity.introduced','perf.layer','route.coverage.missing_ops'].includes(s.id));
   activate((first||items[0]).id);
 }
 if('IntersectionObserver' in window){
   const observer=new IntersectionObserver(entries=>{for(const entry of entries)if(entry.isIntersecting){
     document.querySelectorAll('.sidebar a').forEach(a=>a.classList.toggle('current',entry.target.contains($(a.hash.slice(1)))));
   }},{rootMargin:'-10% 0px -65% 0px'});
   for(const {box} of workspaceTabs.values())observer.observe(box);
 }
}
function activate(id){
 const section=cfg.sections.find(s=>s.id===id);
 if(!section)return;
 const tabs=workspaceTabs.get(section.group);if(!tabs)return;
 for(const s of tabs.items)$(s.id).hidden=s.id!==id;
 for(const button of tabs.bar.children)button.setAttribute('aria-pressed',String(button.dataset.tab===id));
 if($(id).tagName==='DETAILS')$(id).open=true;
}
function revealHash(){
 const id=decodeURIComponent(location.hash.slice(1)),panel=$(id);
 if(!panel)return;
 activate(id);
 if(id==='diagnosis'||id==='diagnosis.coverage'){
   const filters=document.querySelector('.filters');
   panel.style.scrollMarginTop=filters&&getComputedStyle(filters).position==='sticky'?(filters.offsetHeight+12)+'px':'';
 }
 panel.scrollIntoView();
}
function columnTools(section,state){
 const panel=$(section.id),rows=data[section.id]||[],context=[],empty=[];
 const constantKeys=new Set(['dev','entity','rank','unit','association_label','instrumentation','missing_durations','level','status','backend','source','phase','generation_step']);
 state.omit=new Set();
 section.columns.forEach((key,i)=>{
   const head=panel.querySelectorAll('thead th')[i],sort=panel.querySelector('.sort');
   const b=document.createElement('button');b.textContent=head.textContent;head.replaceChildren(b);
   b.onclick=()=>{state.desc=state.sort===key?!state.desc:true;state.sort=key;sort.value=key;state.page=0;panel.querySelector('.direction').textContent=state.desc?'降序':'升序';draw(section);};
   const value=val(rows[0]||{},key);
   if(rows.every(r=>val(r,key)==null||val(r,key)==='')){
     if(!cfg.missing_text[key]&&!rows.some(r=>r._field_notes?.[key]))empty.push(key);
   }
   else if(rows.length>1 && constantKeys.has(key) && rows.every(r=>val(r,key)===value)){
     context.push(b.textContent+' '+fmt(value,key));state.omit.add(key);
   }
 });
 empty.forEach(k=>state.omit.add(k));
 const tools=panel.querySelector('.table-tools');
 if(context.length){const note=document.createElement('span');note.className='column-context';note.textContent=context.join(' · ');tools.before(note);}
 if(state.omit.size){
   const button=document.createElement('button');button.textContent='展开 '+state.omit.size+' 个收起列';tools.append(button);
   button.onclick=()=>{state.expanded=!state.expanded;button.textContent=state.expanded?'收起辅助列':'展开 '+state.omit.size+' 个收起列';draw(section);};
 }
 for(const {field:key,value:v,label} of section.quick_filters||[]){
   if(!selectedElement(key,v)||!rows.some(r=>valuesFor(section.id,r,key).includes(v)))continue;
   const b=document.createElement('button');b.textContent=label;b.onclick=()=>{$('filter-'+key).value=v;apply();};tools.append(b);
 }
}
function selectedElement(key,value){const select=$('filter-'+key);return select&&Array.from(select.options).some(o=>o.value===value);}

function tableCell(section,state,row,key,index){
 const value=val(row,key),scalarValue=scalar(value),numeric=typeof scalarValue==='number';
 const field=row.unit==='ns'&&['baseline','candidate','delta'].includes(key)?'duration_ns':key;
 const shown=fieldText(row,key,field);
 let content=esc(shown.length<=140?shown:shown.slice(0,137)+'…'),style='',tone='';
 if(key==='name'&&typeof value==='string'&&value.length>64&&/[<(]/.test(value)){
   const stem=value.split(/[<(]/)[0].split(/\s+/).pop();content=esc(stem.slice(-72)+'…');
 }else if(key==='ulp_histogram'&&value&&typeof value==='object'){
   const entries=Object.entries(value).filter(([,n])=>Number.isFinite(n));
   const max=entries.reduce((v,e)=>Math.max(v,e[1]),1);
   content='<span class="histogram">'+entries.map(([b,n])=>'<i title="'+esc(b+' ULP · '+n)+'" style="height:'+Math.max(1,22*n/max)+'px"></i>').join('')+'</span>';
 }else if(key==='absolute_error_quantiles'&&value&&typeof value==='object'){
   content=Object.entries(value).filter(([k])=>['p50','p90','p99'].includes(k)).map(([k,v])=>esc(k+' '+fmt(v))).join(' · ')||content;
 }else if(key==='report'&&typeof value==='string'&&!value.startsWith('/')&&!value.includes(':')&&!value.split('/').includes('..')){
   content='<a href="'+esc(value)+'">打开 rank 结果</a>';
 }
 if(numeric&&state.barKey===key&&state.barMax>0){
   style=' style="--share:'+Math.min(100,100*Math.abs(scalarValue)/state.barMax)+'%"';
   tone+=' spark';
 }
 if(['delta','relative_change'].includes(key)&&Number.isFinite(value)&&value!==0){
   const duration=/_ns$/.test(row.metric),rate=/per_second|throughput/.test(row.metric);
   if(duration||rate)tone+=(duration?value<0:value>0)?' better':' worse';
 }
 return '<td data-panel="'+esc(section.id)+'" data-row="'+index+'" data-field="'+esc(key)+'" data-unit="'+(/_ns$/.test(key)?'ns':'')+'" class="'+(numeric?'numeric':'')+tone+'"'+style+' title="'+esc(shown)+'">'+content+'</td>';
}
