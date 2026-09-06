#!/usr/bin/env python3
"""Summarise results*.jsonl: per class and overall min/mean/max of a5->C1, a5->C0,
a4->C0 (control), split strict (context independent of a5) / non-strict, and list
the lowest a5->C1 conditions.  Then refute the lowest ones with fresh seeds and
N=1000 states."""
import json, sys, os, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tie_scan as T

M = 0xFFFFFFFF
files = sys.argv[1:] or ['results2.jsonl']
rows = []
for f in files:
    for line in open(f):
        rows.append(json.loads(line))
print(f"{len(rows)} measured conditions")


def stats(v):
    v = np.array(v)
    return f"min {v.min():5.2f}  mean {v.mean():5.2f}  max {v.max():5.2f}  (n={len(v)})"


def defs_of(r):
    return {int(k): tuple(v) for k, v in r['defs'].items()}


for strict in (True, False):
    sub = [r for r in rows if r['strict'] == strict]
    if not sub:
        continue
    print(f"\n===== {'STRICT (context independent of a5)' if strict else 'NON-STRICT (e9 saturated through a9: a9 depends on a5)'}: {len(sub)} conditions")
    print("  a5->C1 :", stats([r['a5'][1] for r in sub]))
    print("  a5->C0 :", stats([r['a5'][0] for r in sub]))
    print("  a5->C2 :", stats([r['a5'][2] for r in sub]))
    print("  a5->C3 :", stats([r['a5'][3] for r in sub]))
    c = [r['a4C0'] for r in sub if r['a4C0'] is not None]
    print("  a4->C0 (control, a4 free):", stats(c) if c else "n/a")
    byc = collections.defaultdict(list)
    for r in sub:
        byc[r['cls']].append(r)
    print(f"  {'class':45s} {'n':>6s}  a5->C1 min/mean/max      a5->C0 min/mean/max    a4->C0 min/mean/max")
    for cls in sorted(byc):
        rr = byc[cls]
        c1 = np.array([r['a5'][1] for r in rr]); c0 = np.array([r['a5'][0] for r in rr])
        c4 = np.array([r['a4C0'] for r in rr if r['a4C0'] is not None])
        s4 = f"{c4.min():5.2f}/{c4.mean():5.2f}/{c4.max():5.2f}" if c4.size else "  n/a"
        print(f"  {cls:45s} {len(rr):6d}  {c1.min():5.2f}/{c1.mean():5.2f}/{c1.max():5.2f}        {c0.min():5.2f}/{c0.mean():5.2f}/{c0.max():5.2f}      {s4}")

print("\n===== overall")
print("  a5->C1 :", stats([r['a5'][1] for r in rows]))
below4 = [r for r in rows if r['a5'][1] < 4]
print(f"  conditions with a5->C1 < 4 bits: {len(below4)}")
rows.sort(key=lambda r: r['a5'][1])
print("\n  lowest 15 a5->C1:")
for r in rows[:15]:
    print(f"   C1={r['a5'][1]:5.2f} C0={r['a5'][0]:5.2f} C2={r['a5'][2]:5.2f} C3={r['a5'][3]:5.2f} strict={r['strict']} a4C0={r['a4C0']}  {r['cls']}  {defs_of(r)}")
print("\n  lowest 10 STRICT a5->C1:")
for r in [x for x in rows if x['strict']][:10]:
    print(f"   C1={r['a5'][1]:5.2f} C0={r['a5'][0]:5.2f} C2={r['a5'][2]:5.2f} C3={r['a5'][3]:5.2f} a4C0={r['a4C0']}  {r['cls']}  {defs_of(r)}")

if '--refute' in os.environ.get('ANALYZE_FLAGS', ''):
    print("\n===== refuter: lowest 8 overall and lowest 4 strict, seeds 101..103, N=1000")
    todo = rows[:8] + [x for x in rows if x['strict']][:4]
    for r in todo:
        d = defs_of(r)
        ws = []
        for s in (101, 102, 103):
            res = T.measure(d, N=1000, seed=s, words=(5,))
            ws.append(res[5]['mean'][1])
        print(f"   scan {r['a5'][1]:5.2f} -> refuter {np.mean(ws):5.2f} (+-{np.std(ws):.2f})  {r['cls']}  {d}")
