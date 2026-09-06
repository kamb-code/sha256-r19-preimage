#!/usr/bin/env python3
"""Run a list of jobs with a pool of single-thread workers under a wall cap.

usage: driver.py JOBFILE OUTDIR [--par 8] [--cap 1800]
JOBFILE: one job per line: R KIND TSEED SOLVER SSEED [extra tokens...]
Each job runs `timeout CAP python run_job.py ...`; on timeout a JSON with
status=timeout and t_solve=CAP is written.  Skips jobs whose JSON exists.
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

SCR = "/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad"
PY = f"{SCR}/venv/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__))


def run(job, outdir, cap):
    toks = job.split()
    name = "_".join(toks)
    out = os.path.join(outdir, name + ".json")
    if os.path.exists(out):
        return name, "cached"
    cmd = ["timeout", "-s", "KILL", str(int(cap) + 30), PY, f"{HERE}/run_job.py", *toks[:5], out, *toks[5:],
           "--timeout", str(cap)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if not os.path.exists(out):
        rec = {"R": int(toks[0]), "kind": toks[1], "tseed": int(toks[2]), "solver": toks[3], "sseed": int(toks[4]),
               "extra": (toks[6] if len(toks) > 6 else None), "status": "timeout", "t_solve": cap,
               "wall": dt, "stderr": p.stderr[-300:]}
        with open(out, "w") as f:
            json.dump(rec, f)
        return name, f"timeout after {dt:.0f}s"
    return name, p.stdout.strip()


def main():
    jobfile, outdir = sys.argv[1], sys.argv[2]
    par = 8; cap = 1800
    a = sys.argv[3:]
    while a:
        if a[0] == "--par":
            par = int(a[1]); a = a[2:]
        elif a[0] == "--cap":
            cap = float(a[1]); a = a[2:]
        else:
            raise SystemExit(a)
    os.makedirs(outdir, exist_ok=True)
    jobs = [l.strip() for l in open(jobfile) if l.strip() and not l.startswith("#")]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=par) as ex:
        for name, res in ex.map(lambda j: run(j, outdir, cap), jobs):
            print(f"[{time.time() - t0:7.0f}s] {name}: {res}", flush=True)


if __name__ == "__main__":
    main()
