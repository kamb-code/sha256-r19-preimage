#!/usr/bin/env python3
"""Sweep the constant c9 = e9 - a5 (fixed through a9 = c9 + T2(a8,a7,a6), legal)
under the context that kills every Maj/Ch atom of a5 in C1 that can be killed
(a6 = a4, a7 = a6, e8 = -1, e11 = 0), and measure the a5 -> C1 edge.
Candidates: all 1-bit and 2-bit constants (and their complements), plus 2000
random constants.  Reports the lightest.
"""
import itertools, json, sys, time
import numpy as np
from multiprocessing import Pool
import a5edge

M = 0xFFFFFFFF
BASE = {6: ('eq', 4), 7: ('eq', 6), 8: ('sat_r', M), 11: ('sat_r', 0)}


def one(c9):
    np.seterr(over='ignore')
    w, _ = a5edge.edge_weights({**BASE, 9: ('sat_off', c9)}, n=200, seed=3)
    return c9, [w[j][0] for j in range(4)]


if __name__ == "__main__":
    cands = {0, M}
    for i in range(32):
        cands.add(1 << i); cands.add((1 << i) ^ M)
    for i, j in itertools.combinations(range(32), 2):
        v = (1 << i) | (1 << j); cands.add(v); cands.add(v ^ M)
    rng = np.random.default_rng(11)
    cands |= {int(x) for x in rng.integers(0, 1 << 32, 2000, dtype=np.uint64)}
    cands = sorted(cands)
    print(f"{len(cands)} values of c9", flush=True)
    t0 = time.time()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 24) as p:
        res = list(p.imap_unordered(one, cands, chunksize=16))
    res.sort(key=lambda r: r[1][1])
    with open("c9_sweep.json", "w") as f:
        json.dump(res, f)
    c1 = np.array([r[1][1] for r in res])
    print(f"a5 -> C1 over c9: min {c1.min():.2f} mean {c1.mean():.2f} max {c1.max():.2f}; {time.time()-t0:.0f}s")
    print("lightest 12:")
    for c9, w in res[:12]:
        print(f"  c9 = 0x{c9:08x}  C0 {w[0]:.2f}  C1 {w[1]:.2f}  C2 {w[2]:.2f}  C3 {w[3]:.2f}")
    print("heaviest 3:")
    for c9, w in res[-3:]:
        print(f"  c9 = 0x{c9:08x}  C1 {w[1]:.2f}")
