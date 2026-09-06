#!/usr/bin/env python3
"""Run a job list through a pool of NPAR single-threaded solver processes.

Each job line: R structure seed solver timeout [extra args...]
Results are appended (one JSON per line) to results.jsonl by solve_one.py; the
driver logs start/finish and wall-clock to driver.log.
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/venv/bin/python"
NPAR = int(os.environ.get("NPAR", "8"))
jobs_file, out = sys.argv[1], sys.argv[2]
jobs = [l.split() for l in open(jobs_file) if l.strip() and not l.startswith("#")]
log = open(os.path.join(HERE, os.path.basename(jobs_file) + ".log"), "a")
T0 = time.time()


def run(job):
    R, S, seed, solver, to = job[:5]
    extra = job[5:]
    t0 = time.time()
    cmd = [PY, os.path.join(HERE, "solve_one.py"), R, S, seed, solver, to, "--out", out] + extra
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=float(to) + 300)
        status = p.stdout.strip()[-200:] if p.returncode == 0 else "ERR " + p.stderr.strip()[-300:]
    except subprocess.TimeoutExpired:
        status = "HARD-TIMEOUT"
    log.write(f"[{time.time() - T0:8.1f}s] {' '.join(job)} -> {time.time() - t0:8.1f}s {status}\n")
    log.flush()


with ThreadPoolExecutor(NPAR) as ex:
    list(ex.map(run, jobs))
log.write(f"ALL DONE in {time.time() - T0:.1f}s\n")
log.close()
