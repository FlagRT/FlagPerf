"""Immutable stage publication for the optional offline importer. No old Store run is resumed."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sqlite3
from .disk import ArrayFile,JSONFile,Sink,streaming,integer_order_key
from .limits import read_json


DEFAULT_SOURCE_BYTES = 1 << 30
MAX_SOURCE_BYTES = 8 << 30


def source_budget(value=DEFAULT_SOURCE_BYTES):
    if type(value) is not int or not 0 < value <= MAX_SOURCE_BYTES:
        raise ValueError('source byte budget must be between 1 and 8 GiB')
    return value


def syncdir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def identity(path):
    st=Path(path).stat();return [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]

def fileinfo(path):
    before=identity(path);sha=streaming.sha_file(path)
    if before!=identity(path):raise ValueError('changed while hashing')
    return dict(bytes=before[2],sha256=sha)

def publish(path,value):
    path=Path(path);temp=path.with_name(path.name+'.partial')
    with temp.open('xb') as f:
        f.write(json.dumps(value,sort_keys=True,separators=(',',':')).encode()+b'\n');f.flush();os.fsync(f.fileno())
    # link gives no-replace publication, unlike replacing another stage manifest.
    os.link(temp,path);syncdir(path.parent);temp.unlink();syncdir(path.parent)

def binding(source,occurrence,options=None):
    here=Path(__file__).resolve().parent;root=here.parents[1]
    return dict(source=fileinfo(source),occurrence=occurrence,options=options or {},
        parser={str(p.relative_to(root)):streaming.sha_file(p) for p in sorted((root/'flagprof').rglob('*.py'))},
        prototype={p.name:streaming.sha_file(p) for p in sorted(here.glob('*.py'))})

def manifest(directory,kind,bind,files,extra=None):
    rows={}
    for rel in files:
        p=Path(directory)/rel
        if p.is_symlink() or not p.is_file():raise ValueError('stage file type')
        with p.open('rb') as f:os.fsync(f.fileno())
        rows[str(rel)]=fileinfo(p)
    syncdir(directory)
    value=dict(kind=kind,binding=bind,files=rows,extra=extra or {})
    publish(Path(directory)/'committed.json',value);return value

def validate(directory,bind,kind,*,source_max_bytes=None):
    directory=Path(directory);p=directory/'committed.json'
    if p.stat().st_size>4<<20:raise ValueError('checkpoint manifest budget')
    m=read_json(p)
    if m['binding']!=bind or m['kind']!=kind:raise ValueError('stage binding mismatch')
    if len(m['files'])>4096:raise ValueError('stage file budget')
    if kind=='sealed-source':
        limit=source_budget(m.get('extra',{}).get('source_max_bytes',DEFAULT_SOURCE_BYTES))
        if source_max_bytes is not None and limit!=source_budget(source_max_bytes):
            raise ValueError('sealed source budget binding mismatch')
        if sum(v['bytes'] for v in m['files'].values())>limit:raise ValueError('sealed source byte budget')
    for name,expected in m['files'].items():
        rel=Path(name);target=directory/rel
        if rel.is_absolute() or '..' in rel.parts or target.is_symlink() or not target.resolve().is_relative_to(directory.resolve()):raise ValueError('unsafe stage path')
        if fileinfo(target)!=expected:raise ValueError('stage content mismatch')
    return m

def save_scan(db,directory,bind,crash=None):
    directory=Path(directory);directory.mkdir()
    target=directory/'scan.sqlite';dest=sqlite3.connect(target)
    dest.create_collation('BIGINT',lambda a,b:(int(a)>int(b))-(int(a)<int(b)))
    dest.create_function('integer_order_key',1,integer_order_key,deterministic=True)
    db.c.backup(dest);assert dest.execute('PRAGMA quick_check').fetchone()==('ok',);dest.close()
    if crash=='scan-before-publish':os._exit(71)
    manifest(directory,'scan',bind,['scan.sqlite'])
    if crash=='scan-after-publish':os._exit(72)


def tagged(v, base=None):
    if isinstance(v,(JSONFile,ArrayFile)):return {'$file':str(v.path.resolve().relative_to(base.resolve())) if base else str(v.path),'$type':type(v).__name__}
    if isinstance(v,dict):return {k:tagged(x,base) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [tagged(x,base) for x in v]
    return v

def untag(v, base=None):
    if isinstance(v,dict) and set(v)=={'$file','$type'}:return {'JSONFile':JSONFile,'ArrayFile':ArrayFile}[v['$type']]((base/checked_relative(v['$file'])) if base else v['$file'])
    if isinstance(v,dict):return {k:untag(x,base) for k,x in v.items()}
    if isinstance(v,list):return [untag(x,base) for x in v]
    return v

def checked_relative(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts:raise ValueError('stage reference escape')
    return p

def save_prepared(p,bind,crash=None):
    data={k:tagged(v,p.directory) for k,v in vars(p).items() if k!='directory'}
    data['_checkpoint_version']=2
    for meta in data['files'].values():
        if meta is not None:meta['path']=str(Path(meta['path']).resolve().relative_to(p.directory.resolve()))
    publish(p.directory/'prepared-state.json',data)
    if crash=='derived-before-publish':os._exit(73)
    files=[q.name for q in p.directory.iterdir() if q.is_file() and q.suffix in ('.json','.jsonl')]
    manifest(p.directory,'derived',bind,files)
    if crash=='derived-after-publish':os._exit(74)

def load_prepared(directory,bind):
    from .pipeline import Prepared
    committed=validate(directory,bind,'derived');p=Prepared(directory)
    state=Path(directory)/'prepared-state.json'
    if state.stat().st_size>4<<20:raise ValueError('prepared metadata budget')
    data=read_json(state);version=data.pop('_checkpoint_version',1)
    if version not in (1,2):raise ValueError('checkpoint version')
    for k,v in data.items():setattr(p,k,untag(v,Path(directory) if version==2 else None))
    if version==2:
        for meta in p.files.values():
            if meta is not None:meta['path']=str(Path(directory)/checked_relative(meta['path']))
    # A forged path in state cannot make the importer read outside the stage.
    def check(v):
        if isinstance(v,(JSONFile,ArrayFile)):
            if not v.path.resolve().is_relative_to(Path(directory).resolve()):raise ValueError('stage reference escape')
        elif isinstance(v,dict):
            for x in v.values():check(x)
        elif isinstance(v,list):
            for x in v:check(x)
    check(p.fields)
    for meta in p.files.values():
        if meta is None:continue
        path=Path(meta['path'])
        if not path.resolve().is_relative_to(Path(directory).resolve()):raise ValueError('stage artifact reference escape')
        expected=committed['files'].get(str(path.relative_to(directory)))
        if expected!={k:meta[k] for k in ('bytes','sha256')}:raise ValueError('stage artifact metadata mismatch')
    return p


def seal(source,destination,occurrence,*,max_bytes=1<<30,normal_close=True):
    max_bytes=source_budget(max_bytes)
    if not normal_close:raise ValueError('producer not normally closed')
    source=Path(source);destination=Path(destination);destination.mkdir()
    pending=destination/'data';pending.mkdir();files=[];total=0
    from .limits import walk_files
    for p in walk_files(source,max_files=4096,max_entries=8192):
        if p.is_symlink():raise ValueError('source symlink')
        if not p.is_file():continue
        if len(files)>=4096:raise ValueError('seal file budget')
        before=identity(p);rel=p.relative_to(source);target=pending/rel;target.parent.mkdir(parents=True,exist_ok=True)
        h=hashlib.sha256()
        with p.open('rb') as src,target.open('xb') as dst:
            while block:=src.read(65536):
                total+=len(block)
                if total>max_bytes:raise ValueError('seal disk budget')
                dst.write(block);h.update(block)
            dst.flush();os.fsync(dst.fileno())
        if identity(p)!=before or streaming.sha_file(p)!=h.hexdigest() or streaming.sha_file(target)!=h.hexdigest():raise ValueError('seal source/copy changed')
        files.append(str(Path('data')/rel))
    for directory,_dirs,_filenames in os.walk(pending,topdown=False):syncdir(directory)
    syncdir(pending)
    return manifest(destination,'sealed-source',dict(occurrence=occurrence,normal_close=True),files,
                    extra=dict(source_max_bytes=max_bytes))
