#!/usr/bin/env python3
"""Aggregate run_job JSON results: median/min/max solve time per (R, kind, extra, solver)."""
import glob
import json
import statistics
import sys
from collections import defaultdict

dirs = sys.argv[1:]
rows = []
for d in dirs:
    for f in glob.glob(f"{d}/*.json"):
        rows.append(json.load(open(f)))

groups = defaultdict(list)
for r in rows:
    groups[(r["R"], r["kind"], r.get("extra") or "-", r["solver"])].append(r)

print(f"{'R':>3} {'kind':5} {'extra':8} {'solver':10} {'n':>2} {'solved':>6} {'verified':>8} {'median_s':>9} {'min_s':>8} {'max_s':>8}  times")
for key in sorted(groups):
    g = groups[key]
    times = [x["t_solve"] for x in g]
    solved = sum(x["status"] == "sat" for x in g)
    ver = sum(bool(x.get("verified")) for x in g)
    unsat = sum(x["status"] == "unsat" for x in g)
    print(f"{key[0]:>3} {key[1]:5} {key[2]:8} {key[3]:10} {len(g):>2} {solved:>6} {ver:>8} "
          f"{statistics.median(times):>9.2f} {min(times):>8.2f} {max(times):>8.2f}  "
          + " ".join(f"{t:.1f}{'' if x['status']=='sat' else ('U' if x['status']=='unsat' else 'T')}" for t, x in zip(times, g))
          + (f"  [unsat={unsat}]" if unsat else ""))
