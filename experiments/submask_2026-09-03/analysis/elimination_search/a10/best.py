#!/usr/bin/env python3
"""Inspect the enumeration result for w = a_wi: distribution of occurrence
classes, the configurations that come closest to freeing C0,C1,C2 of w, the
symbolic path report for the best few, and a numeric perturbation check of them."""
import sys, pickle, collections
import symeng as E
import numcheck as N

wi = int(sys.argv[1].lstrip('a'))
nshow = int(sys.argv[2]) if len(sys.argv) > 2 else 4
d = pickle.load(open(f"enum_a{wi}.pkl", "rb"))
w = f'a{wi}'
agg = d['agg']
print(f"w = {w}: {d['n_legal']:,} legal configurations of {d['total']:,}")
# marginal: per constraint, how many configurations have w absent / linear / heavy
for j in range(4):
    c = collections.Counter()
    for (cls, tri), n in agg.items(): c[cls[j]] += n
    print(f"  C{j}: " + ", ".join(f"{k}={v:,}" for k, v in sorted(c.items())))
free012 = sum(n for (cls, tri), n in agg.items() if all(c == 'none' for c in cls[:3]))
print(f"  w absent from all of C0,C1,C2: {free012:,}")
# 'linear absorber' patterns: w linear in C_j and absent from C_i, i<j
for j in range(3):
    n = sum(n for (cls, tri), n in agg.items() if cls[j] == 'lin' and all(c == 'none' for c in cls[:j]))
    print(f"  w linear in C{j} and absent from earlier lookups: {n:,}")
best = sorted(d['best'], key=lambda b: (b[0], not b[2]))
print(f"\nclosest configurations ({len(best)} stored; score = number of C0,C1,C2 containing w):")
seen = 0
for score, cls, tri, defs in best:
    if seen >= nshow: break
    seen += 1
    print(f"\n--- score {score}, classes {cls}, triangular={tri}\n    defs = {defs}")
    S = E.build(defs)
    print(E.report_paths(S, w, js=(0, 1, 2)))
    trials = 6 if any(v[0] == 'sat_s0' for v in defs.values()) else 30
    N.run(defs, wi, trials=trials, seed=7)
