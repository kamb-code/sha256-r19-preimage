#!/usr/bin/env python3
"""Run gpu_submask.py at R=20 on a rented RunPod H100 with a hard wall cap.

Creates the pod, uploads the two scripts, starts the sweep detached with a PID
file, polls status.json every few minutes into a local log, stops on HIT.txt or
the cap, copies everything back and deletes the pod in a finally block.
Never prints the API key.
"""
import json, os, subprocess, sys, time, tomllib, urllib.request, urllib.error

S = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(S, "r20_gpu_run.log")
STATUS_LOCAL = os.path.join(S, "r20_status.json")
KEY = "/home/administrator/sha/sha256/sha256_key"
PUB = open(os.path.join(S, "sha256_key.pub")).read().strip()
CODE = "/home/administrator/sha/publish/code"
HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
WALL_CAP = (HOURS + 1.0) * 3600
POLL = 300
API = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
GPU_CANDIDATES = ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe", "NVIDIA H100 NVL"]
api_key = tomllib.load(open(os.path.expanduser("~/.runpod/config.toml"), "rb"))["default"]["api_key"]


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def api(method, path, body=None):
    req = urllib.request.Request(API + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {api_key}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            txt = r.read().decode()
            return r.status, (json.loads(txt) if txt.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:400]}
    except Exception as e:
        return 0, {"error": str(e)[:200]}


def sshb(ip, port):
    return ["ssh", "-i", KEY, "-p", str(port), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=20", f"root@{ip}"]


def ssh(ip, port, cmd, timeout=120):
    try:
        return subprocess.run(sshb(ip, port) + [cmd], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", "ssh timeout")


def scp(ip, port, src, dst, to_pod=True, timeout=600):
    base = ["scp", "-i", KEY, "-P", str(port), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]
    args = base + ([src, f"root@{ip}:{dst}"] if to_pod else [f"root@{ip}:{src}", dst])
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, "", "scp timeout")


def create_pod():
    for gpu in GPU_CANDIDATES:
        body = {"name": "sha20-submask", "imageName": IMAGE, "gpuTypeIds": [gpu],
                "gpuCount": 1, "containerDiskInGb": 40, "volumeInGb": 0,
                "ports": ["22/tcp"], "env": {"PUBLIC_KEY": PUB}, "cloudType": "SECURE",
                "supportPublicIp": True, "computeType": "GPU"}
        st, resp = api("POST", "/pods", body)
        if st in (200, 201) and resp.get("id"):
            log(f"pod created: id={resp['id']} gpu={gpu}")
            return resp["id"]
        log(f"create with {gpu!r} -> HTTP {st}: {str(resp)[:300]}")
    raise SystemExit("could not create a pod")


def wait_ssh(pod_id):
    t0 = time.time()
    while time.time() - t0 < 900:
        st, p = api("GET", f"/pods/{pod_id}")
        ip = p.get("publicIp"); pm = p.get("portMappings") or {}; port = pm.get("22")
        if ip and port:
            r = subprocess.run(sshb(ip, port) + ["echo ok"], capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and "ok" in r.stdout:
                log(f"ssh up at {ip}:{port} cost/h={p.get('costPerHr')}")
                return ip, port
        time.sleep(15)
    raise SystemExit("pod never became reachable")


def main():
    log(f"R=20 submask-family run, cap {HOURS} h on the sweep, {WALL_CAP/3600:.1f} h on the pod")
    pod_id = create_pod()
    t_create = time.time()
    try:
        ip, port = wait_ssh(pod_id)
        ssh(ip, port, "mkdir -p /workspace/r20")
        for f in ("submask_family.py", "gpu_submask.py"):
            r = scp(ip, port, os.path.join(CODE, f), f"/workspace/r20/{f}")
            if r.returncode != 0:
                log(f"scp {f} failed: {r.stderr[:200]}"); raise SystemExit(1)
        r = ssh(ip, port, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; "
                          "python3 -c 'import torch;print(torch.__version__)'")
        log("pod: " + " | ".join(r.stdout.split()))
        cmd = (f"cd /workspace/r20 && (setsid nohup python3 gpu_submask.py --rounds 20 --roots 3 "
               f"--stop-on-hit --hours {HOURS} --out /workspace/r20 > stdout.log 2>&1 < /dev/null "
               f"& echo $! > solver.pid) ; sleep 3; kill -0 $(cat solver.pid) 2>/dev/null "
               f"&& echo started || echo NOT-STARTED")
        r = ssh(ip, port, cmd)
        log("solver: " + (r.stdout.strip() or r.stderr.strip()[:200]))
        if "started" not in r.stdout:
            raise SystemExit("solver did not start")
        found = False
        while True:
            time.sleep(POLL)
            elapsed = time.time() - t_create
            r = ssh(ip, port, "cd /workspace/r20 && tail -n 1 run.log; ls HIT.txt 2>/dev/null; "
                              "kill -0 $(cat solver.pid) 2>/dev/null && echo RUNNING || echo EXITED")
            lines = r.stdout.strip().splitlines() if r.returncode == 0 else ["<ssh failed>"]
            scp(ip, port, "/workspace/r20/status.json", STATUS_LOCAL, to_pod=False, timeout=120)
            log(f"t={elapsed/3600:.2f}h | " + " | ".join(l.strip()[:200] for l in lines[-3:]))
            if any("HIT.txt" in l for l in lines):
                found = True
            if found or any(l.strip() == "EXITED" for l in lines) or elapsed > WALL_CAP:
                break
        for f in ("run.log", "status.json", "HIT.txt", "stdout.log"):
            scp(ip, port, f"/workspace/r20/{f}", os.path.join(S, "r20_" + f), to_pod=False, timeout=300)
        log(f"finished: found={found} elapsed={(time.time()-t_create)/3600:.2f} h")
    finally:
        st, _ = api("DELETE", f"/pods/{pod_id}")
        log(f"pod {pod_id} delete -> HTTP {st}")
        st, p = api("GET", f"/pods/{pod_id}")
        log(f"post-delete status: HTTP {st} {str(p)[:100]}")
        log("DONE")


if __name__ == "__main__":
    import signal
    def _term(signum, frame):
        raise SystemExit(f"signal {signum}")
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGHUP, _term)
    main()
