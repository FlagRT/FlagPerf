"""Host and registered process counters. No target-side polling thread."""
import os
from pathlib import Path
from .contract import Call, Metric, Observation, ReadError, raw_clock


def process_stat(pid, root='/proc'):
    text = Path(root, str(pid), 'stat').read_text()
    fields = text[text.rindex(')')+2:].split()
    return dict(start_ticks=int(fields[19]), ticks=int(fields[11])+int(fields[12]),
                rss=int(fields[21])*os.sysconf('SC_PAGE_SIZE'))


def host_pid(key, root='/proc', known_pid=None):
    """Resolve namespace PID plus start ticks; PID reuse never joins."""
    if key.get('boot_id') != Path(root, 'sys/kernel/random/boot_id').read_text().strip():
        return None
    # Resolve once, then verify the same kernel identity without rescanning every
    # process for every registered target on each slow sampling round.
    candidates=[Path(root,str(known_pid))] if known_pid is not None else [Path(root,str(key['pid']))]
    def matches(path):
        if str(os.stat(path/'ns/pid').st_ino)!=key['pid_ns']:return False
        status=(path/'status').read_text()
        ns=next(line.split()[1:] for line in status.splitlines() if line.startswith('NSpid:'))
        return int(ns[-1])==key['pid'] and process_stat(path.name,root)['start_ticks']==key['proc_start_ticks']
    for path in candidates:
        try:
            if matches(path):return int(path.name)
        except (OSError,ValueError,StopIteration):pass
    if known_pid is not None:return None
    for path in Path(root).iterdir():
        if not path.name.isdecimal():
            continue
        try:
            if matches(path):return int(path.name)
        except (OSError, ValueError, StopIteration):
            continue
    return None


class Provider:
    container_safe = True

    def __init__(self, root='/proc'):
        self.root = Path(root)
        self.boot = (self.root/'sys/kernel/random/boot_id').read_text().strip()
        self.previous = {}
        self.registered = set()
        self.metrics = [Metric('host.memory.available', 'B', 'host', '/proc/meminfo', group='slow'),
                        Metric('process.cpu', '%', 'process', '/proc/PID/stat', group='slow'),
                        Metric('process.rss', 'B', 'process', '/proc/PID/stat', group='slow')]

    def calls(self):
        return [Call('host.memory', 'host', 'host:'+self.boot, ('host.memory.available',), self.memory, 'slow')]

    def memory(self):
        start = raw_clock()
        available = next(int(line.split()[1])*1024 for line in (self.root/'meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        return Observation('host:'+self.boot, {'host.memory.available': available}, start, raw_clock())

    def process(self, key, pid):
        entity = 'proc:{pid_ns}:{pid}:{proc_start_ticks}'.format(**key)
        self.registered.add(entity)
        def read():
            start = raw_clock()
            stat = process_stat(pid, self.root)
            if stat['start_ticks'] != key['proc_start_ticks']:
                raise ReadError('read_error', 'process identity ended')
            old = self.previous.get(entity)
            if entity in self.registered:
                self.previous[entity] = (start, stat['ticks'])
            cpu = (100*(stat['ticks']-old[1])/os.sysconf('SC_CLK_TCK')/((start-old[0])/1e9)) if old else None
            return Observation(entity, {'process.cpu': cpu, 'process.rss': stat['rss']}, start, raw_clock(),
                               {'process.cpu': 'read_error'} if cpu is None else {})
        return Call(entity, entity, entity, ('process.cpu', 'process.rss'), read, 'slow')

    def forget(self, entity):
        self.registered.discard(entity)
        self.previous.pop(entity, None)
