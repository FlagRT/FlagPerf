"""Bounded process reclamation and monotonic heartbeat observation."""
import subprocess
import time


def reclaim(child, term_seconds=3, kill_seconds=3):
    if child.poll() is not None:
        return dict(confirmed=True, returncode=child.returncode, killed=False)
    child.terminate()
    killed = False
    try:
        child.wait(timeout=term_seconds)
    except subprocess.TimeoutExpired:
        killed = True
        child.kill()
        try:
            child.wait(timeout=kill_seconds)
        except subprocess.TimeoutExpired:
            return dict(confirmed=False, returncode=None, killed=True)
    return dict(confirmed=True, returncode=child.returncode, killed=killed)


class Heartbeat:
    def __init__(self, path, timeout):
        self.path, self.timeout = path, timeout
        self.last_seen = time.monotonic()
        self.signature = None

    def expired(self):
        now = time.monotonic()
        try:
            stat = self.path.stat()
            signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if signature != self.signature:
                self.signature, self.last_seen = signature, now
        except FileNotFoundError:
            pass
        return now-self.last_seen > self.timeout
