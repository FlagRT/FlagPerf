const resourceNames = {
 'memory.used':'设备显存占用', device_used_bytes:'设备显存占用', 'memory.total':'设备显存容量',
 'process.device_memory':'进程设备显存', allocated_bytes:'分配器已分配显存', active_bytes:'分配器活跃显存',
 reserved_bytes:'分配器预留显存', allocation_bytes:'单次分配大小', release_source_time_ns:'释放时刻',
 'process.rss':'进程驻留内存', 'host.memory.available':'主机可用内存', 'process.cpu':'进程 CPU 使用率',
 'power.draw':'板卡功率', 'temperature.chip':'芯片温度', 'temperature.memory':'显存温度',
 'clock.compute':'计算频率', 'clock.compute.max':'最高计算频率', 'clock.memory':'显存频率',
 'process.count':'设备进程数', 'ascend.health':'芯片健康状态', 'ascend.error.count':'设备错误数',
 'ascend.clock.reasons':'频率限制原因'
};
function resourceName(metric) {
 const activity = /^ascend\.(\w+)\.activity$/.exec(metric);
 return Object.hasOwn(resourceNames,metric) ? resourceNames[metric] : (activity ? activity[1].toUpperCase()+' 活动率' : metric);
}
function resourceEntity(entity) {
 const pid = /^proc:\d+:(\d+):\d+$/.exec(entity);
 return pid ? 'PID '+pid[1] : String(entity ?? '对象未记录').replace(/^host:(.{8}).*$/, '主机 $1');
}
function resourceOptions(section, select, series) {
 const preferred = section.chart?.default_series?.join(' / ');
 for (const [key, item] of Object.entries(series)) {
   const field = item.metric, metric = field.split('/').pop();
   const scope = [resourceEntity(item.device), field.slice(0, Math.max(0, field.lastIndexOf('/'))),
                  item.component === '同主机资源观测' ? '' : item.component].filter(Boolean);
   const label = /^(cpu|host)/i.test(item.device) ? resourceName(metric).replace('显存','内存') : resourceName(metric);
   const option = new Option([label, ...scope].join(' · '), key);
   option.title = key;
   if (key === preferred) select.prepend(option);
   else select.add(option);
   // Some saved display envelopes predate the process metric units.
   if (!item.unit) {
     if (['process.rss','process.device_memory'].includes(metric)) item.unit='B';
     else if (metric==='process.cpu'||/^ascend\.\w+\.activity$/.test(metric)) item.unit='%';
     else if (metric==='release_source_time_ns') item.unit='ns';
   }
 }
 if (preferred && series[preferred]) select.value=preferred;
 const panel=$(section.id), description=panel.querySelector('.description');
 panel.classList.add('resource-trend');
 const metric=series[preferred]?.metric.split('/').pop();
 const notes=[];
 if (['memory.used','device_used_bytes'].includes(metric)) notes.push('设备显存读数覆盖整个设备。');
 else if (section.id==='memory.series') notes.push('各来源分别记录分配、活跃或预留的内存。');
 else if (['process.rss','host.memory.available'].includes(metric)) {
   notes.push('本次未采到设备显存占用时间序列。进程驻留内存表示进程占用的主机物理内存。');
 }
 if (section.id==='resource.series') {
   if (cfg.meta.phase_bands?.length) notes.push('浅色背景标出已记录的任务阶段。');
   if (Number.isFinite(cfg.meta.resource_fault_ns)) notes.push('红虚线标出窗口内的告警时刻。');
 }
 if (description && notes.length) {
   description.textContent=notes.join('')+(section.id==='memory.series'?description.textContent:'');
 }
}
function resourceReading(valid, unit) {
 if (!valid.length) return '当前时间范围内没有有效观测。';
 let low=Infinity, high=-Infinity;
 for (const point of valid) { low=Math.min(low,point[1]); high=Math.max(high,point[1]); }
 const range=low===high ? axisValue(low,unit) : axisValue(low,unit)+' – '+axisValue(high,unit);
 return '窗口内观测值 '+range+' · '+valid.length+' 个有效显示点 · 横轴为距首个采样点的秒数';
}

const resourceMeasures = [['mean','均值','#176a83'],['maximum','最大值','#a04b15'],['p95','P95','#684a92']];
function statisticNumber(value,unit) {
 if (!Number.isFinite(value)) return '—';
 if (unit==='Hz' && Math.abs(value)>=1e6) return num(value/1e6)+' MHz';
 if (['bitmask','bool','code','count'].includes(unit)) return num(value);
 return axisValue(value,unit==='C'?'°C':unit).trim();
}

function resourceStatistics(section, items) {
 if (!items.length) return '<p>当前筛选下没有资源统计记录。</p>';
 // Older stage aggregates omit the unit. Recover it only from an unambiguous entity/metric pair.
 const units=new Map();
 for (const r of data.resource||[]) {
   if (!r.unit) continue;
   const key=JSON.stringify([r.entity,r.metric]);
   units.set(key,units.has(key)&&units.get(key)!==r.unit ? '' : r.unit);
 }
 const groups=new Map();
 for (const item of items) {
   const r=item.r, unit=r.unit||units.get(JSON.stringify([r.entity,r.metric]))||'';
   const key=JSON.stringify([r.metric,unit]);
   if (!groups.has(key)) groups.set(key,{metric:r.metric, unit, rows:[]});
   groups.get(key).rows.push(item);
 }
 const stage=section.id==='resource.stages';
 const intro=stage ? '每行汇总同名阶段的各次运行。' : '每行对应一个实际观测窗口。';
 let html='<p>'+intro+'同一指标和单位共用刻度。三条色带从上到下依次表示均值、最大值和 P95。'+
   '均值按已覆盖时长加权，区间型指标的最大值取各采样窗口均值中的最大值。横杠表示该值未提供。</p><div class="resource-stat-grid">';
 for (const {metric,unit,rows} of groups.values()) {
   const numeric=unit && !['code','bitmask','bool','unknown'].includes(unit);
   let low=Infinity, high=-Infinity;
   for (const {r} of rows) for (const [field] of resourceMeasures) {
     if (Number.isFinite(r[field])) { low=Math.min(low,r[field]); high=Math.max(high,r[field]); }
   }
   const observed=Number.isFinite(low);
   low=Math.min(0,low); high=Math.max(0,high);
   const x=value=>(high===low?0:100*(value-low)/(high-low))+'%';
   const scale=!observed ? '缺少有效统计' : numeric ? '刻度 '+statisticNumber(low,unit)+' – '+statisticNumber(high,unit) :
               ({code:'状态码',bitmask:'位掩码',bool:'布尔状态'}[unit] || '单位未记录');
   html+='<section class="resource-stat-group"><h3>'+esc(resourceName(metric)||'指标未记录')+'</h3>'+
         '<small>'+esc(metric||'')+' · '+esc(scale)+'</small>';
   for (const {r,i} of rows) {
     const window=Array.isArray(r.window)?r.window.join(' – ')+' ns':'';
     const meta=[resourceEntity(r.entity), r.name ?? r.phase ?? '窗口未命名'];
     if (Number.isFinite(r.instances)) meta.push(r.instances+' 次阶段实例');
     const coverage=r.coverage ?? r.coverage_ratio;
     const quality=['时间覆盖 '+(Number.isFinite(coverage)?num(100*coverage)+'%':'—'),
                    '样本 '+(r.samples??'—')];
     if (r.status && r.status!=='available') quality.push(fmt(r.status));
     let marks='', readings='';
     resourceMeasures.forEach(([field,title,color],index)=>{
       const v=r[field], y=4+index*7;
       const info=field==='maximum'&&r.maximum_kind==='window_mean_maximum'?'采样窗口均值中的最大值':title;
       readings+='<span data-stat="'+field+'" title="'+esc(r._field_notes?.[field]||info)+'">'+title+' '+esc(statisticNumber(v,unit))+'</span>';
       if (numeric && Number.isFinite(v)) {
         marks+=line(x(0),y,x(v),y,color,3)+'<circle cx="'+x(v)+'" cy="'+y+'" r="2" fill="'+color+'"/>';
       }
     });
     html+='<button class="resource-stat-row" type="button" data-point="'+i+'" title="'+esc([...meta,window].filter(Boolean).join(' · '))+'">'+
           '<strong>'+esc(meta.join(' · '))+'</strong><span class="resource-stat-values">'+readings+'</span>'+
           (marks?'<svg aria-hidden="true">'+marks+'</svg>':'')+'<small>'+esc(quality.join(' · '))+'</small></button>';
   }
   html+='</section>';
 }
 return html+'</div>';
}
