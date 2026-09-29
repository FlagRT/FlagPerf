# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Preview wall-time estimates, separate from execution qualification."""
import math
import time
from runtime.common import digest

COST_MARGIN = 1.5


def p75(values):
    ordered = sorted(values[-8:])
    return ordered[max(0, math.ceil(len(ordered) * .75) - 1)]


class Budget:
    def __init__(self, seconds, timeout, profiles, mode='fixed', clock=None):
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.seconds, self.timeout, self.profiles = seconds, timeout, profiles
        self.mode = mode or 'fixed'
        self.cleanup = 0.0

    def remaining(self):
        return max(0.0, self.seconds - (self.clock() - self.started - self.cleanup))

    def baseline(self, key):
        good = [r['seconds'] for r in self.profiles[key]['timings']
                if r['phase'] == 'baseline' and r['completed'] and not r['timed_out']]
        return max(1., p75(good) if good else self.seconds / 4 / max(1, len(self.profiles)))

    def verification(self, key, include=None):
        selection = digest(include if include is not None else self.profiles[key].get('accepted', []))
        good = [r['seconds'] for r in self.profiles[key]['timings']
                if r['phase'] in ('final', 'common', 'control') and r['completed']
                and r.get('usable', True) and not r['timed_out'] and r.get('selection_sha256') == selection]
        # Discovery and the first restored worker can compile; neither is a warm sample.
        return min(self.timeout, max(1., COST_MARGIN * p75(good) if good else 3 * self.baseline(key)))

    def estimate(self, key, phase, candidate=None, group=None, include=None):
        history = self.profiles[key]['timings']
        good = [r['seconds'] for r in history if r['phase'] == phase and r['completed'] and not r['timed_out']]
        if self.mode == 'fixed':
            if not good:
                good = [r['seconds'] for r in history if r['completed'] and not r['timed_out']]
            estimate = COST_MARGIN * max(good[-8:]) if good else self.seconds / 4 / max(1, len(self.profiles))
        elif phase in ('final', 'common', 'control'):
            estimate = self.verification(key, include)
        elif phase in ('candidate', 'group'):
            increments = [max(0., r['seconds'] - r['warm_estimate']) / len(r['group'])
                          for r in history if r['phase'] in ('candidate', 'group')
                          and r.get('usable') and not r['timed_out'] and r.get('group')
                          and 'warm_estimate' in r]
            incremental = max(1., COST_MARGIN * p75(increments)) if increments else 2 * self.baseline(key)
            estimate = self.verification(key, include) + len(group or [candidate]) * incremental
        else:
            estimate = COST_MARGIN * p75(good) if good else self.verification(key, include)
        lower = [r['allowance'] for r in history if r['timed_out'] and
                 (r.get('selection_sha256') == digest(include) if include is not None else
                  r.get('candidate') == candidate if candidate is not None else r['phase'] == phase)]
        if lower:
            estimate = max(estimate, COST_MARGIN * max(lower))
        return min(self.timeout, max(1., estimate))

    def verification_tasks(self, proposed=None):
        sets = {k:list(p.get('accepted', [])) for k,p in self.profiles.items()}
        if proposed:
            sets[proposed[0]] = list(proposed[1])
        common = set.intersection(*(set(s) for s in sets.values()))
        tasks = []
        for key, include in sets.items():
            # Empty profiles still need an opportunity to acquire and verify a selection.
            tasks.append({'environment':key, 'phase':'final', 'include':include,
                          'seconds':self.verification(key, include)})
            if common and set(include) != common:
                joint = sorted(common)
                tasks.append({'environment':key, 'phase':'common', 'include':joint,
                              'seconds':self.verification(key, joint)})
        return tasks

    def reserve(self, proposed=None):
        initialized = all(any(t['phase'] == 'baseline' and t['completed'] and not t['timed_out']
                              for t in p['timings']) for p in self.profiles.values())
        if self.mode == 'fixed' or not initialized:
            return min(self.seconds * .25, 2 * len(self.profiles) * self.timeout)
        return sum(t['seconds'] for t in self.verification_tasks(proposed))

    def room(self, proposed=None):
        return max(0., self.remaining() - self.reserve(proposed))
