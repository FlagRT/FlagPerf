"""Vendor-neutral interval arithmetic."""
from decimal import Decimal
from bisect import bisect_right
from collections import defaultdict

def ns(value): return int(Decimal(str(value))*1000)


class ScopeIndex:
    """Exact host-thread windows, including overlapping ancestors and exclusive ends."""
    def __init__(self, scopes):
        grouped = defaultdict(list)
        for label, scope in scopes:
            start = ns(scope['ts'])
            grouped[(scope.get('pid'), scope.get('tid'))].append((start, start+ns(scope['dur']), label))
        self.groups = {}
        for key, frames in grouped.items():
            frames.sort(key=lambda row: row[0])
            maximum, ends = float('-inf'), []
            for _, end, _ in frames:
                maximum = max(maximum, end)
                ends.append(maximum)
            self.groups[key] = frames, [row[0] for row in frames], ends

    def at(self, pid, tid, stamp):
        frames, starts, ends = self.groups.get((pid, tid), ([], [], []))
        index, output = bisect_right(starts, stamp)-1, []
        while index >= 0 and ends[index] > stamp:
            start, end, label = frames[index]
            if start <= stamp < end:
                output.append(label)
            index -= 1
        return output


def merged_intervals(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if end < start: raise ValueError('negative interval')
        if merged and start <= merged[-1][1]: merged[-1][1]=max(end,merged[-1][1])
        else: merged.append([start,end])
    return merged


def union_ns(intervals):
    return sum(b-a for a,b in merged_intervals(intervals))


def overlap_ns(left,right):
    left, right = merged_intervals(left), merged_intervals(right)
    i, j, total = 0, 0, 0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        total += max(0, min(b,d)-max(a,c))
        if b < d:
            i += 1
        else:
            j += 1
    return total
