#!/usr/bin/env python3
"""Summarise a structural sweep: per saturation pattern (which of a8..a11 are
free / sat_r / sat_h / sat_s0) and per tie family, min/mean/max of the a5 -> C1
and a5 -> C0 weights, and the global best configurations."""
import sys, json, collections
import numpy as np

fn = sys.argv[1]
rows = [json.loads(l) for l in open(fn)]
print(f"{fn}: {len(rows)} legal configurations")
C1 = np.array([r['mean'][1] for r in rows]); C0 = np.array([r['mean'][0] for r in rows])
C2 = np.array([r['mean'][2] for r in rows]); C3 = np.array([r['mean'][3] for r in rows])
print(f"a5->C1: min {C1.min():.3f} mean {C1.mean():.3f} max {C1.max():.3f}")
print(f"a5->C0: min {C0.min():.3f} mean {C0.mean():.3f} max {C0.max():.3f}")
print(f"a5->C2: min {C2.min():.3f} mean {C2.mean():.3f} max {C2.max():.3f}")
print(f"a5->C3: min {C3.min():.3f} mean {C3.mean():.3f} max {C3.max():.3f}")


def pattern(r):
    d = r['defs']
    return ','.join({'free': 'F', 'sat_r': 'R', 'sat_h': 'H', 'sat_s0': 'S'}[d[str(i)][0]] for i in (8, 9, 10, 11))


by = collections.defaultdict(list)
for r in rows:
    by[pattern(r)].append(r)
print("\nper saturation pattern of (a8,a9,a10,a11)  [F free, R e_i via a_i, H e_{i+4} via a_i, S e_{i+1} via S0(a_i)]:")
print(f"{'pattern':10} {'n':>6} {'C1 min':>7} {'C1 mean':>8} {'C1 max':>7} | {'C0 min':>7} {'C0 mean':>8} | {'min C1 config'}")
for pat, rs in sorted(by.items(), key=lambda kv: min(r['mean'][1] for r in kv[1])):
    c1 = np.array([r['mean'][1] for r in rs]); c0 = np.array([r['mean'][0] for r in rs])
    b = rs[int(np.argmin(c1))]
    print(f"{pat:10} {len(rs):>6} {c1.min():7.3f} {c1.mean():8.3f} {c1.max():7.3f} | {c0.min():7.3f} {c0.mean():8.3f} | {b['name']}  C0={b['mean'][0]:.2f}")

print("\n20 lowest a5->C1 configurations:")
for r in sorted(rows, key=lambda r: r['mean'][1])[:20]:
    print(f"  C1 {r['mean'][1]:.3f} (zero {r['zero'][1]:.2f}, min-state {r['min_state'][1]:.2f})  C0 {r['mean'][0]:.2f}  C2 {r['mean'][2]:.2f}  C3 {r['mean'][3]:.2f}   {r['name']}")

print("\nconfigurations with C1 below 2.5 AND C0 below 2.5 (both edges mild):")
both = [r for r in rows if r['mean'][1] < 2.5 and r['mean'][0] < 2.5]
print(f"  {len(both)}")
for r in sorted(both, key=lambda r: r['mean'][1])[:10]:
    print(f"  C1 {r['mean'][1]:.3f} C0 {r['mean'][0]:.2f} C2 {r['mean'][2]:.2f} C3 {r['mean'][3]:.2f}  {r['name']}")

print("\nlowest C1 with C0 mild (< 2.0):")
for r in sorted([r for r in rows if r['mean'][0] < 2.0], key=lambda r: r['mean'][1])[:10]:
    print(f"  C1 {r['mean'][1]:.3f} C0 {r['mean'][0]:.2f} C2 {r['mean'][2]:.2f} C3 {r['mean'][3]:.2f}  {r['name']}")
