#!/usr/bin/env python3
"""When any pod reports a verified 20-round hit, stop the others.

Polls the scratchpad for r20*HIT.txt every 60 s.  On the first one: touch
r20_GLOBAL_STOP (the parametrised runners see it at their next poll and delete
their pods) and SIGTERM the original H100 runner (its handler deletes its pod).
"""
import glob, os, signal, subprocess, time
S = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(S, "stop_watcher.log")
def log(m):
    with open(LOG, "a") as f: f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}\n")
log("watcher started")
while True:
    hits = glob.glob(os.path.join(S, "r20*HIT.txt"))
    if hits:
        log(f"hit file(s) seen: {hits}")
        open(os.path.join(S, "r20_GLOBAL_STOP"), "w").write(str(hits))
        r = subprocess.run(["pgrep", "-f", "python3 run_r20_gpu.py"], capture_output=True, text=True)
        for pid in r.stdout.split():
            try:
                os.kill(int(pid), signal.SIGTERM); log(f"SIGTERM -> H100 runner pid {pid}")
            except Exception as e:
                log(f"kill {pid} failed: {e}")
        log("done"); break
    time.sleep(60)
