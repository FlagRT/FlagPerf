"""Static point queries over exact host-thread intervals.

Each interval is stored in one centered node. A query visits one tree path
and only matching intervals plus one stopping entry per node. In particular,
a long ancestor cannot force a scan of all preceding, expired siblings.
"""
from collections import defaultdict
from runtime.trace_utils import ns


class _Node:
    __slots__ = ('center', 'starts', 'ends', 'left', 'right')

    def __init__(self, intervals):
        self.center = intervals[len(intervals) // 2][0]
        left, right, overlap = [], [], []
        for start, end, label in intervals:
            target = left if end <= self.center else right if start > self.center else overlap
            target.append((start, end, label))
        self.starts = overlap
        self.ends = sorted(overlap, key=lambda row: row[1], reverse=True)
        self.left = _Node(left) if left else None
        self.right = _Node(right) if right else None

    def at(self, stamp, stats=None):
        result, node = [], self
        while node is not None:
            if stats is not None:
                stats['nodes'] = stats.get('nodes', 0) + 1
            before = stamp < node.center
            for start, end, label in node.starts if before else node.ends:
                if stats is not None:
                    stats['intervals'] = stats.get('intervals', 0) + 1
                if (before and start > stamp) or (not before and end <= stamp):
                    break
                result.append(label)
            node = node.left if before else node.right
        return result


class ThreadScopeIndex:
    def __init__(self, scopes):
        groups = defaultdict(list)
        for label, scope in scopes:
            start = ns(scope['ts'])
            end = start + ns(scope.get('dur', 0))
            # Empty and backwards windows never contain a timestamp.
            rows = groups[(scope.get('pid'), scope.get('tid'))]
            if end > start:
                rows.append((start, end, label))
        self.groups = {key: _Node(sorted(rows, key=lambda row: row[0])) if rows else None
                       for key, rows in groups.items()}

    def at(self, pid, tid, stamp, stats=None):
        node = self.groups.get((pid, tid))
        return node.at(stamp, stats) if node else []
