// All cards share an absolute source window. A missing bucket breaks its path.
function monitorGrid(section) {
 const panel=$(section.id),state=states[section.id],rows=data[section.id]||[];
 if(!state.series)state.series=seriesFor(section);
 const window=cfg.meta.monitor_window;
 let lower=window?.[0],upper=window?.[1];
 if(lower==null||upper==null){
   const points=Object.values(state.series).filter(s=>s.points.length);
   lower=points.reduce((v,s)=>Math.min(v,s.points[0][0]),Infinity);
   upper=points.reduce((v,s)=>Math.max(v,s.points.at(-1)[0]),-Infinity);
 }
 if(!(upper>lower))return;
 const low=panel.querySelector('.time-low'),high=panel.querySelector('.time-high');
 if(!high.value){low.value=0;high.value=(upper-lower)/1e9;}
 const start=lower+1e9*Number(low.value),end=Math.max(start+1,lower+1e9*Number(high.value));
 const select=panel.querySelector('.series-select');
 select.parentElement.firstChild.textContent='指标类别';
 if(!select.options.length){
   select.add(new Option('全部',''));
   const families=new Set(rows.map(r=>r.metric.split('.')[0]));
   for(const family of families)select.add(new Option(family,family));
 }
 const grid=panel.querySelector('.timeline');grid.classList.add('monitor-grid');grid.replaceChildren();
 const cards=[],groups=new Map();
 for(const i of state.selected){
   const r=rows[i],s=state.series[r.entity+' / '+r.metric];
   if(!s||(select.value&&r.metric.split('.')[0]!==select.value))continue;
   if(!groups.has(r.entity)){
     const title=document.createElement('h3'),group=document.createElement('div');
     title.textContent=r.entity;grid.append(title,group);group.className='metric-cards';groups.set(r.entity,group);
   }
   const points=s.points.filter(([t])=>t>=start&&t<=end),valid=points.filter(p=>Number.isFinite(p[1]));
   const max=valid.reduce((m,p)=>Math.max(m,p[1]),-Infinity),min=valid.reduce((m,p)=>Math.min(m,p[1]),Infinity);
   let path='',active=false;
   for(const [t,value] of points){
     if(!Number.isFinite(value)){active=false;continue;}
     const x=100*(t-start)/(end-start),y=39-34*(value-min)/(max-min||1);
     path+=(active?'L':'M')+x+' '+y+' ';active=true;
   }
   const card=document.createElement('article');card.className='metric-card';
   card.innerHTML='<b>'+esc(fmt(r.metric,'metric'))+'</b><span class="reading"></span><svg viewBox="0 0 100 44" preserveAspectRatio="none" data-point="'+i+'" role="img" aria-label="'+esc(r.metric)+'">'+
     '<path d="M0 40H100M0 22H100M0 5H100" stroke="#e2e8f0" stroke-width=".4" fill="none"/>'+
     '<path d="'+path+'" stroke="#155e8a" stroke-width="1.4" vector-effect="non-scaling-stroke" fill="none"/><line class="crosshair" y1="0" y2="44" stroke="#b45309" stroke-width=".6" visibility="hidden"/></svg>'+
     '<small>'+esc(valid.length?axisValue(min,s.unit)+' – '+axisValue(max,s.unit):'窗口内缺少有效观测')+'</small>';
   groups.get(r.entity).append(card);cards.push({card,points,unit:s.unit});
 }
 const duration=(upper-lower)/1e9;
 if(!panel.querySelector('.range-presets')){
   const presets=document.createElement('span');presets.className='range-presets';
   for(const [title,seconds] of [['全时段',duration],['最近 60 秒',60],['最近 5 分钟',300]]){
     const button=document.createElement('button');button.textContent=title;button.onclick=()=>{low.value=Math.max(0,duration-seconds);high.value=duration;monitorGrid(section);};presets.append(button);
   }
   panel.querySelector('.chart-controls').append(presets);
 }
 grid.onpointermove=e=>{
   const svg=e.target.closest('.metric-card svg');if(!svg)return;
   const rect=svg.getBoundingClientRect(),ratio=Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width)),time=start+ratio*(end-start);
   for(const {card,points,unit} of cards){
     const cursor=card.querySelector('.crosshair');cursor.setAttribute('x1',100*ratio);cursor.setAttribute('x2',100*ratio);cursor.setAttribute('visibility','visible');
     const nearest=points.reduce((a,b)=>!a||Math.abs(b[0]-time)<Math.abs(a[0]-time)?b:a,null);
     card.querySelector('.reading').textContent=nearest?num((nearest[0]-lower)/1e9)+' s · '+(nearest[1]==null?'缺失':axisValue(nearest[1],unit)):'缺失';
   }
 };
 grid.onpointerleave=()=>cards.forEach(({card})=>{card.querySelector('.crosshair').setAttribute('visibility','hidden');card.querySelector('.reading').textContent='';});
 state.peak=()=>{
   const first=cards[0];if(!first)return;
   const peak=first.points.reduce((a,b)=>Number.isFinite(b[1])&&(!a||b[1]>a[1])?b:a,null);
   if(!peak)return;
   const t=(peak[0]-lower)/1e9,span=Math.max(duration*.05,1e-6);
   low.value=Math.max(0,t-span);high.value=Math.min(duration,t+span);monitorGrid(section);
 };
}
