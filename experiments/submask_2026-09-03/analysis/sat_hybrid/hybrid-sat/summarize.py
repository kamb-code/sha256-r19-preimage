#!/usr/bin/env python3
import json
import sys
from collections import defaultdict

fn = sys.argv[1] if len(sys.argv) > 1 else "results.jsonl"
rows = [json.loads(l) for l in open(fn)]
g = defaultdict(list)
for d in rows:
    key = (d["R"], d["structure"] + ("+a0" if d.get("fix_a0") else "") + (f"+tight{d['tight']}" if d.get("tight") else "")
           + ("+ones" if d.get("ones") else ""), d["solver"])
    g[key].append(d)
print(f"{'R':>3} {'structure':<18} {'solver':<11} {'n':>2} {'solved':>6} {'times (s; T=timeout, U=unsat)':<60} {'median':>8} verified")
for key in sorted(g):
    ds = sorted(g[key], key=lambda d: d["seed"])
    ts = []
    for d in ds:
        if d["status"] == "SATISFIABLE":
            ts.append(f"{d['solve_s']:.1f}")
        elif d["status"] == "UNSATISFIABLE":
            ts.append(f"U{d['solve_s']:.0f}")
        else:
            ts.append(f"T{d['timeout']:.0f}")
    solved = [d["solve_s"] for d in ds if d["status"] == "SATISFIABLE"]
    ver = all(d.get("verified", False)
              and (d["structure"] == "none" or d.get("family_ok", False))
              and (d["structure"] not in ("collapse", "ctxcoll") or d.get("collapse_ok", False))
              and (d["structure"] not in ("context", "ctxcoll") or d.get("context_ok", False))
              for d in ds if d["status"] == "SATISFIABLE")
    if len(solved) == len(ds):
        s = sorted(solved)
        med = s[len(s) // 2]
        medstr = f"{med:8.1f}"
    else:
        medstr = f">{ds[0]['timeout']:.0f}" if len(solved) <= len(ds) // 2 else f"{sorted(solved)[len(ds)//2]:8.1f}"
    print(f"{key[0]:>3} {key[1]:<18} {key[2]:<11} {len(ds):>2} {len(solved):>6} {' '.join(ts):<60} {medstr:>8} {ver}")
