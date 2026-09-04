#!/usr/bin/env python3
"""Target-aware stop watcher (replaces stop_watcher.py).

A hit on the ALL-ONES digest (the headline target) stops every pod: touch
r20_GLOBAL_STOP, which every runner honours at its next poll and then deletes
its pod after fetching results.  A hit on the hedge pod's random digest does
NOT stop the all-ones pods; that pod's own runner shuts it down.
"""
import glob, os, re, time
S = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(S, "stop_watcher2.log")
ONES = "f" * 64
seen = set()
def log(m):
    with open(LOG, "a") as f: f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}\n")
log("target-aware watcher started")
while True:
    for p in glob.glob(os.path.join(S, "r20_*HIT.txt")):
        if p in seen: continue
        seen.add(p)
        txt = open(p).read()
        m = re.search(r"hash=([0-9a-f]{64})", txt)
        h = m.group(1) if m else "?"
        log(f"hit file {os.path.basename(p)} target={h[:16]}...")
        if h == ONES:
            open(os.path.join(S, "r20_GLOBAL_STOP"), "w").write(p)
            log("ALL-ONES HIT: global stop issued")
        else:
            log("hedge (random-target) hit: all-ones pods left running")
    time.sleep(30)
