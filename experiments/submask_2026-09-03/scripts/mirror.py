#!/usr/bin/env python3
"""Independent 60-second mirror of the three live pods, plus a runner watchdog.

Copies status.json / run.log / HIT.txt / stdout.log from every pod into the
repository's experiment directory (read-only on the pods), independently
re-verifies any hit listed in status.json with verify_r19.py --rounds 20, and
logs loudly if a runner process disappears.  Purely additive: it never
stops, starts or deletes anything.
"""
import json, os, subprocess, time
S = os.path.dirname(os.path.abspath(__file__))
D = "/home/administrator/sha/publish/experiments/submask_2026-09-03/gpu_run_snapshot/live"
KEY = "/home/administrator/sha/sha256/sha256_key"
VERIFY = "/home/administrator/sha/publish/code/verify_r19.py"
LOG = os.path.join(S, "mirror.log")
PODS = {"t1": ("216.81.151.3", 13516), "t2": ("185.216.23.244", 31196),
        "dev": ("154.54.102.45", 13524)}
RUNNER_PAT = {"t1": "run_r20_pod.py t1", "t2": "run_r20_pod.py t2", "dev": "attach_pod.py dev"}
verified = set()


def log(m):
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}\n")


def scp(ip, port, remote, local):
    try:
        r = subprocess.run(["scp", "-i", KEY, "-P", str(port), "-o", "StrictHostKeyChecking=no",
                            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                            "-o", "ConnectTimeout=15", f"root@{ip}:{remote}", local],
                           capture_output=True, text=True, timeout=90)
        return r.returncode == 0
    except Exception:
        return False


def runner_alive(pat):
    me = os.getpid()
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode()
        except Exception:
            continue
        if cmd.startswith("python3 ") and pat in cmd and "mirror" not in cmd:
            return True
    return False


log("mirror started")
dead_reported = set()
while True:
    for tag, (ip, port) in PODS.items():
        d = os.path.join(D, tag); os.makedirs(d, exist_ok=True)
        for f in ("status.json", "run.log", "HIT.txt", "stdout.log"):
            scp(ip, port, f"/workspace/r20/{f}", os.path.join(d, f))
        sp = os.path.join(d, "status.json")
        if os.path.exists(sp):
            try:
                s = json.load(open(sp))
                for w in s.get("hits", []):
                    key = (tag, w)
                    if key in verified:
                        continue
                    r = subprocess.run(["python3", VERIFY, "--rounds", "20", "--hash", s["target"],
                                        "--words", w], capture_output=True, text=True, timeout=60)
                    ok = "OK" in r.stdout.splitlines()[-1] if r.stdout else False
                    log(f"MIRROR {'VERIFIED' if ok else 'FAILED-VERIFY'} HIT on {tag} target={s['target'][:16]}... words={w}")
                    verified.add(key)
            except Exception as e:
                log(f"{tag}: status parse error {str(e)[:80]}")
        if not runner_alive(RUNNER_PAT[tag]) and tag not in dead_reported:
            lg = os.path.join(S, f"r20_{tag}_run.log")
            done = os.path.exists(lg) and "DONE" in open(lg).read()[-2000:]
            if not done:
                log(f"!!! RUNNER DIED for {tag}: pod may be orphaned -- attach a runner or delete pod")
                dead_reported.add(tag)
    time.sleep(60)
