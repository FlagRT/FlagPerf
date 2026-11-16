# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Display envelopes retain extrema and gaps; statistics use original observations."""
import json


def unit(metric):
    # Units from the versioned telemetry inventory are preferred by callers.
    if metric.endswith('.bytes') or 'memory.' in metric or metric.endswith('_bytes'):return 'B'
    if metric.startswith('power.'):return 'W'
    if metric.startswith('energy.'):return 'J'
    if metric.startswith(('temperature.','temp.')):return '°C'
    if metric.startswith(('clock.','frequency.')):return 'Hz'
    if metric.endswith('.percent') or metric.startswith(('utilization.','util.')):return '%'
    return ''


def memory_envelope(model,points,ident='memory.series'):
    """External grouping keeps arbitrarily many raw points out of the HTML."""
    db=model.db
    db.execute('create table if not exists curve_points(series text,ts integer,value real,payload text)')
    db.execute('delete from curve_points')
    count=0
    for row in points:
        if row.get('source_time_ns') is None:continue
        for metric,value in row.get('values',{}).items():
            key=json.dumps([row.get('dev'),row.get('component'),metric],ensure_ascii=False)
            db.execute('insert into curve_points values(?,?,?,?)',(key,row['source_time_ns'],value,json.dumps(row)))
            count+=1
        if count%256==0:
            db.commit()
            if model.path.stat().st_size>model.limits.scratch_bytes:raise ValueError('curve work exceeds scratch budget')
    db.commit()
    db.execute('create index if not exists curve_series on curve_points(series,ts)')
    for key,low,high,n in db.execute('select series,min(ts),max(ts),count(*) from curve_points group by series'):
        dev,component,metric=json.loads(key);bucket=[];previous=None
        def emit():
            if not bucket:return
            valid=[r for r in bucket if r[1] is not None]
            selected=[bucket[0],bucket[-1]]
            if valid:selected.extend([min(valid,key=lambda p:p[1]),max(valid,key=lambda p:p[1])])
            missing=next((p for p in bucket if p[1] is None),None)
            if missing:
                # A partially missing bucket is a break, not an interpolated measurement.
                selected=[(bucket[0][0],None),(bucket[-1][0],None)]+([max(valid,key=lambda p:p[1]),min(valid,key=lambda p:p[1])] if valid else [])
            values=sorted(set(selected),key=lambda p:(p[0],p[1] is not None))
            if missing:values=[v for ts,value in values for v in ((ts,None),(ts,value),(ts,None))]
            for ts,value in values:
                model.add(ident,dict(dev=dev,component=component,source_time_ns=ts,values={metric:value},
                    curve_source_points=n,curve_rule='time_bucket_first_last_extrema_gap',unit=unit(metric)))
        # At most 512 buckets per series; each bucket accumulator retains only extrema/endpoints.
        for ts,value in db.execute('select ts,value from curve_points where series=? order by ts',(key,)):
            slot=min(511,int((ts-low)*512/max(1,high-low)))
            if previous is not None and slot!=previous:emit();bucket=[]
            bucket.append((ts,value));previous=slot
            if len(bucket)>6:
                first,last=bucket[0],bucket[-1];valid=[v for v in bucket if v[1] is not None]
                missing=next((v for v in bucket if v[1] is None),None)
                bucket=[first,last]+([min(valid,key=lambda p:p[1]),max(valid,key=lambda p:p[1])] if valid else [])+([missing] if missing else [])
        emit()
    db.execute('delete from curve_points')
    return count
