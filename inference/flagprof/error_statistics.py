# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Exact finite-error quantiles with bounded memory and temporary disk usage."""
import math
from pathlib import Path
import struct
import tempfile
from flagprof.store.analysis_read import ResourceLimit


class ErrorDistribution:
    def __init__(self,work=None,limits=None):
        from flagprof.analyzers.workspace import Limits
        self.limits=limits or Limits()
        self.temp=tempfile.TemporaryDirectory(prefix='error-distribution-',dir=work)
        self.available=self.limits.scratch_bytes-(sum(p.stat().st_size for p in Path(work).rglob('*') if p.is_file()) if work else 0)
        self.buffer=[];self.files=[];self.count=0;self.near_zero=0;self.overflow=0
        self.chunk=max(1,min(65536,self.limits.cache_bytes//64))

    def add(self,error,reference):
        self.near_zero+=abs(reference)<=1e-12
        if not math.isfinite(error):self.overflow+=1;return
        self.buffer.append(error);self.count+=1
        if self.count*56>self.available:raise ResourceLimit('error quantiles exceed scratch_bytes')
        if len(self.buffer)>=self.chunk:self.flush()

    def flush(self):
        if not self.buffer:return
        path=Path(self.temp.name)/str(len(self.files))
        with path.open('wb') as stream:
            for value in sorted(self.buffer):stream.write(struct.pack('<d',value))
        self.files.append(path);self.buffer=[]

    def finish(self,n,reference_square,error_square):
        quantiles={str(p):None for p in (50,90,99)}
        if self.count and not self.overflow:
            positions={p:(self.count-1)*p/100 for p in (50,90,99)}
            needed={i for pos in positions.values() for i in (math.floor(pos),math.ceil(pos))};selected={}
            if not self.files:
                ordered=sorted(self.buffer)
                selected={i:ordered[i] for i in needed}
            else:
                self.flush()
                # SQLite external sorting avoids keeping one open descriptor per chunk.
                import sqlite3
                db=sqlite3.connect(str(Path(self.temp.name)/'ordered.sqlite'))
                try:
                    db.execute('pragma temp_store=FILE');db.execute('pragma cache_size=-1024')
                    db.execute('create table errors(value real)')
                    for path in self.files:
                        with path.open('rb') as stream:
                            while True:
                                raw=stream.read(65536)
                                if not raw:break
                                db.executemany('insert into errors values(?)',struct.iter_unpack('<d',raw))
                        path.unlink()
                        if sum(p.stat().st_size for p in Path(self.temp.name).iterdir())>self.limits.scratch_bytes:
                            raise ResourceLimit('error quantile index exceeds scratch_bytes')
                    db.commit()
                    for index,(value,) in enumerate(db.execute('select value from errors order by value')):
                        if index in needed:selected[index]=value
                        if index>=max(needed):break
                finally:db.close()
            quantiles={str(p):selected[math.floor(pos)]*(1-(pos%1))+selected[math.ceil(pos)]*(pos%1) for p,pos in positions.items()}
        rms=math.sqrt(reference_square/n) if n else None
        rmse=math.sqrt(error_square/n) if n else None
        finite=lambda v:v if v is None or math.isfinite(v) else None
        return dict(absolute_error_quantiles=quantiles,error_quantile_rule='linear_(n-1)*p',
            error_quantile_elements=self.count,error_quantile_overflow=self.overflow,
            reference_rms=finite(rms),rmse=finite(rmse),normalized_rmse=finite(rmse/rms) if rms and rmse is not None else None,
            normalized_rmse_reason='zero_reference_scale' if n and rms==0 else 'no_valid_elements' if not n else None,
            near_zero_reference_elements=self.near_zero,near_zero_threshold=1e-12,
            near_zero_reference_fraction=self.near_zero/n if n else None)

    def close(self):self.temp.cleanup()
