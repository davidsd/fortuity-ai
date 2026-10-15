"""Memory guard: current RSS via ps, abort cleanly if above a cap."""
import os, subprocess, sys, time


def rss_gb():
    out = subprocess.run(['ps', '-o', 'rss=', '-p', str(os.getpid())], capture_output=True, text=True).stdout
    try:
        return int(out.strip()) / 1e6  # ps reports KB
    except ValueError:
        return 0.0


class MemGuard:
    def __init__(self, cap_gb, log=None):
        self.cap = cap_gb
        self.log = log
        self.peak = 0.0

    def check(self, tag=""):
        r = rss_gb()
        self.peak = max(self.peak, r)
        if self.log:
            print(f"[mem] {tag} rss={r:.2f}GB peak={self.peak:.2f}GB cap={self.cap}GB", flush=True)
        if r > self.cap:
            print(f"[mem] ABORT: rss {r:.2f} GB exceeds cap {self.cap} GB at {tag}", flush=True)
            sys.exit(3)
        return r
