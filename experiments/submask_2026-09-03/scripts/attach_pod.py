#!/usr/bin/env python3
"""Attach a runner to a pod whose solver was started by hand.

Polls every 5 minutes; on HIT.txt fetches it (so stop_watcher.py fires), on
GLOBAL_STOP / EXITED / wall cap copies logs back and DELETES the pod.
    python3 attach_pod.py <tag> <pod_id> <ip> <port> [hours]
"""
import json, os, subprocess, sys, time, tomllib, urllib.request, urllib.error

S = os.path.dirname(os.path.abspath(__file__))
TAG, POD, IP, PORT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
HOURS = float(sys.argv[5]) if len(sys.argv) > 5 else 61.0
LOG = os.path.join(S, f"r20_{TAG}_run.log")
GLOBAL_STOP = os.path.join(S, "r20_GLOBAL_STOP")
KEY = "/home/administrator/sha/sha256/sha256_key"
API = "https://rest.runpod.io/v1"
api_key = tomllib.load(open(os.path.expanduser("~/.runpod/config.toml"), "rb"))["default"]["api_key"]


def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def api(method, path):
    req = urllib.request.Request(API + path, method=method,
                                 headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def ssh(cmd, timeout=120):
    try:
        return subprocess.run(["ssh", "-i", KEY, "-p", str(PORT), "-o", "StrictHostKeyChecking=no",
                               "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                               "-o", "ConnectTimeout=20", f"root@{IP}", cmd],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", "timeout")


def fetch(remote, local, timeout=120):
    try:
        r = subprocess.run(["scp", "-i", KEY, "-P", str(PORT), "-o", "StrictHostKeyChecking=no",
                            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                            f"root@{IP}:{remote}", local], capture_output=True, text=True,
                           timeout=timeout)
        return r.returncode == 0
    except Exception as e:
        log(f"fetch {remote} failed: {str(e)[:80]}")
        return False


def main():
    log(f"attached to pod {POD} at {IP}:{PORT}, cap {HOURS} h")
    t0 = time.time()
    found = False
    try:
        while True:
            time.sleep(300)
            el = time.time() - t0
            r = ssh("cd /workspace/r20 && tail -n 1 run.log; ls HIT.txt 2>/dev/null; "
                    "cat HIT.txt 2>/dev/null; "
                    "kill -0 $(cat solver.pid) 2>/dev/null && echo RUNNING || echo EXITED")
            lines = r.stdout.strip().splitlines() if r.returncode == 0 else ["<ssh failed>"]
            fetch("/workspace/r20/status.json", os.path.join(S, f"r20_{TAG}_status.json"))
            log(f"t={el/3600:.2f}h | " + " | ".join(l.strip()[:200] for l in lines[-3:]))
            if any("HIT.txt" in l for l in lines):
                found = True
                fetch("/workspace/r20/HIT.txt", os.path.join(S, f"r20_{TAG}_HIT.txt"))
            if os.path.exists(GLOBAL_STOP):
                log("global stop seen; shutting down"); break
            if found or any(l.strip() == "EXITED" for l in lines) or el > HOURS * 3600:
                break
        for f in ("run.log", "status.json", "HIT.txt", "stdout.log"):
            fetch(f"/workspace/r20/{f}", os.path.join(S, f"r20_{TAG}_{f}"))
        log(f"finished: found={found}")
    finally:
        safe = True
        if found:
            local = os.path.join(S, f"r20_{TAG}_HIT.txt")
            for _ in range(10):
                if os.path.exists(local) and os.path.getsize(local) > 0:
                    break
                fetch("/workspace/r20/HIT.txt", local)
                r = ssh("cat /workspace/r20/HIT.txt")
                if r.returncode == 0 and "words=" in r.stdout:
                    log("HIT (via ssh): " + r.stdout.strip()[:400])
                    open(local, "w").write(r.stdout)
                time.sleep(30)
            safe = os.path.exists(local) and os.path.getsize(local) > 0
        if safe:
            st = 0
            for _ in range(5):
                st = api("DELETE", f"/pods/{POD}")
                if st in (200, 204): break
                time.sleep(20)
            log(f"pod {POD} delete -> HTTP {st}")
        else:
            log(f"!!! HIT NOT FETCHED -- pod {POD} LEFT ALIVE for manual retrieval")
        log("DONE")


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(SystemExit("term")))
    main()
