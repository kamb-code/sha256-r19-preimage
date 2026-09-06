#!/usr/bin/env python3
"""Reproduce the a5 -> C1 (and C0, C2, C3, and control a4 -> C0) edge weight for
one context condition, with both implementations.

    python3 reproduce.py "{'a6': 'eq4', 'a7': 'eq6', 'e8': 0xFFFFFFFF, 'c9': 0, 'e10': 0, 'e11': 0xFFFFFFFF}"

Condition keys (see edge_bulk.py): v, a6, a7 (int constant or 'eq4'/'neq4', 'eq6'/'neq6'),
e8 (needs a7 = 'eq6'), e8via ('a8' | 'a4'), c9 (= e9 - a5, through a9), e10, e11.
CPU only; no table needed (edge weights are properties of the recovered-W formulas).
"""
import sys, os, ast
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_bulk import measure
from refute import to_defs
import numcheck as N

cond = ast.literal_eval(sys.argv[1]) if len(sys.argv) > 1 else \
    {'a6': 'eq4', 'a7': 'eq6', 'e8': 0xFFFFFFFF, 'c9': 0, 'e10': 0, 'e11': 0xFFFFFFFF}
print("condition:", cond)
for seed, ns in ((1, 200), (2, 200), (3, 2000)):
    r = measure(cond, seed, ns=ns)
    print(f"bulk engine   seed {seed} states {ns:5d} x 32 flips: legal={r['legal']}  a5->C0 {r['C0_mean']:.2f}  "
          f"a5->C1 {r['C1_mean']:.2f} (min {r['C1_min']}, max {r['C1_max']}, moved {r['C1_moved']:.3f})  "
          f"a5->C2 {r['C2_mean']:.2f}  a5->C3 {r['C3_mean']:.2f}  | control a4->C0 {r['a4C0_mean']:.2f}")
defs = to_defs(cond)
print("numcheck defs:", defs)
for seed in (17, 18):
    _, _, ham = N.run(defs, 5, trials=40, seed=seed, verbose=False, allbits=True)
    print(f"numcheck      seed {seed} 40 planted digests x 32 flips: a5->C0..C3 = "
          + " ".join(f"{h:.2f}" for h in ham))
