// Older reports inserted a null at the start of every unoccupied display bucket.
// An occupied bucket with a null records an actual read failure or a break around an extremum.
function monitorSeries(row, sourceWindow) {
 const [begin,end]=sourceWindow||[0,256e9],width=(end-begin)/256;
 const buckets=row.points||{},bucketTime=slot=>begin+Math.floor(slot*width);
 const bucketTimes=!Array.isArray(row.samples_points);
 const source=bucketTimes?Object.entries(buckets).map(([slot,value])=>[bucketTime(slot),value]):row.samples_points;
 const padding=new Set();
 if(!bucketTimes&&row.points&&width>0&&source.length>=256){
   const nullTimes=new Set(source.filter(p=>p[1]===null).map(p=>Number(p[0])));
   for(let slot=0;slot<256;slot++){
     if(Object.prototype.hasOwnProperty.call(buckets,slot))continue;
     const time=bucketTime(slot);
     if(!nullTimes.has(time)){padding.clear();break;}
     padding.add(time);
   }
 }
 const points=[];
 for(const [timestamp,value] of source){
   const t=Number(timestamp);
   if(value===null&&padding.delete(t))continue;
   points.push([t,value]);
 }
 points.sort((a,b)=>a[0]-b[0]);
 const times=[...new Set(points.map(p=>p[0]))],intervals=[];
 for(let i=1;i<times.length;i++)intervals.push(times[i]-times[i-1]);
 intervals.sort((a,b)=>a-b);
 // Without a recorded schedule, use the series' observed cadence only if it repeats.
 const period=Number(row.sample_period_ns)||(intervals.length>=2?intervals[Math.floor((intervals.length-1)/2)]:0);
 const reduced=row.valid>points.filter(p=>Number.isFinite(p[1])).length;
 const gapLimit=Math.max(2.5*period,!row.sample_period_ns&&(reduced||bucketTimes)?width:0);
 return {points,unit:row.unit||'',period,gapLimit,bucketTimes,reduced,
         slot:t=>Math.min(255,Math.floor((t-begin)/width)),interrupted:new Set(row.interrupted_slots||[])};
}

function monitorBreak(left,right,series,interruptions) {
 if(!Number.isFinite(left[1])||!Number.isFinite(right[1]))return '此段有采样缺失';
 const gap=interruptions.find(({window:[a,b]})=>left[0]<b&&right[0]>a);
 if(gap)return gap.label;
 if(!series.gapLimit&&right[0]!==left[0])return '采样周期未知';
 if(right[0]-left[0]>series.gapLimit){
   // Within a reduced bucket, distant extrema can represent densely sampled data.
   const slot=series.slot(left[0]);
   if(!series.reduced||slot!==series.slot(right[0]))return '此段未采样';
   if(series.interrupted.has(slot))return '此段采样不连续';
 }
 return '';
}

function monitorReading(chart,time,snap) {
 const {points,series,interruptions}=chart;
 // Look across the complete series even when the zoomed window lies inside a gap.
 let low=0,high=series.points.length;
 while(low<high){
   const mid=Math.floor((low+high)/2);
   if(series.points[mid][0]<time)low=mid+1;else high=mid;
 }
 const left=series.points[low-1],right=series.points[low];
 const nearby=[left,right].filter(p=>p&&p[0]>=chart.start&&p[0]<=chart.end);
 nearby.sort((a,b)=>Math.abs(time-a[0])-Math.abs(time-b[0]));
 const nearest=nearby[0];
 if(nearest&&Math.abs(nearest[0]-time)<=snap){
   // Partially missing buckets can store break markers at a valid extremum's timestamp.
   const real=Number.isFinite(nearest[1])?nearest:points.find(p=>p[0]===nearest[0]&&Number.isFinite(p[1]));
   return real||'采样缺失';
 }
 const gap=interruptions.find(({window:[a,b]})=>a<=time&&time<=b);
 if(gap)return gap.label;
 if(left&&right){
   const reason=monitorBreak(left,right,series,interruptions);
   if(reason)return reason;
 }
 if(nearest&&Number.isFinite(nearest[1])&&(left&&right||Math.abs(nearest[0]-time)<=series.period/2))return nearest;
 return '此时未采样';
}

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
 if(!Number.isFinite(lower)||!Number.isFinite(upper))return;
 if(upper===lower){lower-=5e8;upper+=5e8;}
 const interruptions=cfg.meta.monitor_gaps||[];
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
   const position=([t,value])=>[100*(t-start)/(end-start),max===min?22:39-34*(value-min)/(max-min)];
   const discrete=['code','bitmask','bool'].includes(s.unit);
   let path='',markers='',previous=null;
   for(const point of points){
     if(Number.isFinite(point[1])){
       const xy=position(point).join(' ');
       if(!discrete)path+=(previous&&!monitorBreak(previous,point,s,interruptions)?'L':'M')+xy+' ';
       markers+='M'+xy+'h0 ';
     }
     previous=point;
   }
   const card=document.createElement('article');card.className='metric-card';
   let summary=points.length?'窗口内只有缺失读数':'窗口内无采样点';
   if(valid.length){
     const range=max===min?axisValue(min,s.unit):axisValue(min,s.unit)+' – '+axisValue(max,s.unit);
     summary=(s.bucketTimes?'有读数的时间格 ':'图中采样点 ')+valid.length+' · '+range;
   }
   card.innerHTML='<b>'+esc(fmt(r.metric,'metric'))+'</b><span class="reading"></span><svg viewBox="0 0 100 44" preserveAspectRatio="none" data-point="'+i+'" role="img" aria-label="'+esc(r.metric)+'">'+
     '<path d="M0 40H100M0 22H100M0 5H100" stroke="#e2e8f0" stroke-width=".4" fill="none"/>'+
     '<path class="trend-line" d="'+path+'" stroke="#40748b" stroke-width="1.3" vector-effect="non-scaling-stroke" fill="none"/>'+
     '<path class="sample-dots" d="'+markers+'" stroke="#085678" stroke-width="5" stroke-linecap="round" vector-effect="non-scaling-stroke" fill="none"/>'+
     '<line class="crosshair" y1="0" y2="44" stroke="#b45309" stroke-width="1" vector-effect="non-scaling-stroke" visibility="hidden"/>'+
     '<path class="active-sample" stroke="#b45309" stroke-width="8" stroke-linecap="round" vector-effect="non-scaling-stroke" fill="none"/></svg>'+
     '<small>'+esc(summary)+'</small>';
   groups.get(r.entity).append(card);cards.push({card,points,series:s,position,interruptions,start,end});
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
   for(const chart of cards){
     const {card,series,position}=chart;
     const cursor=card.querySelector('.crosshair');cursor.setAttribute('x1',100*ratio);cursor.setAttribute('x2',100*ratio);cursor.setAttribute('visibility','visible');
     const reading=monitorReading(chart,time,Math.min((end-start)*3/rect.width,series.period?series.period/4:Infinity));
     const sample=Array.isArray(reading);
     card.querySelector('.active-sample').setAttribute('d',sample?'M'+position(reading).join(' ')+'h0':'');
     card.querySelector('.reading').textContent=sample?(series.bucketTimes?'时间格约 ':'采样 ')+num((reading[0]-lower)/1e9)+' s · '+axisValue(reading[1],series.unit):reading;
   }
 };
 grid.onpointerleave=()=>cards.forEach(({card})=>{card.querySelector('.crosshair').setAttribute('visibility','hidden');card.querySelector('.active-sample').removeAttribute('d');card.querySelector('.reading').textContent='';});
 state.peak=()=>{
   const first=cards[0];if(!first)return;
   const peak=first.points.reduce((a,b)=>Number.isFinite(b[1])&&(!a||b[1]>a[1])?b:a,null);
   if(!peak)return;
   const t=(peak[0]-lower)/1e9,span=Math.max(duration*.05,1e-6);
   low.value=Math.max(0,t-span);high.value=Math.min(duration,t+span);monitorGrid(section);
 };
}
