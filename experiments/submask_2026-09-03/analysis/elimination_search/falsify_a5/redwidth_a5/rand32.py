#!/usr/bin/env python3
"""Full-width (w=32) random search over contexts (a4,a6,a7,c8,c9) for the a5->C1
edge, via the verified reduction T_1 = c' - G(a5); 64 states x 32 bit flips per
context.  Also the structured sub-families: c9 small, a4=a6=a7, a7=a6 & a4=~a6."""
import numpy as np, time, sys
from multiprocessing import Pool
from wmodel import Model
m = Model(32); M = m.M
S = 64
def G(a5, a4, a6, a7, c8, c9, a3):
    c7 = (a3 + a7 - m.S0(a6)) & M
    y = (a5 + c9) & M
    X = (c8 - m.Maj(a7, a6, a5)) & M
    Y = (c7 - m.Maj(a6, a5, a4)) & M
    return (m.S0(a5) + m.Maj(a5, a4, a3) - m.S1(y) - m.Ch(y, X, Y)) & M
def worker(args):
    seed, C, fam = args
    rng = np.random.default_rng(seed)
    r = lambda: rng.integers(0, 1 << 32, size=(C, 1), dtype=np.int64)
    a4, a6, a7, c8, c9 = r(), r(), r(), r(), r()
    if fam == 'famA': a6 = a4; a7 = a4
    if fam == 'famB': a7 = a6; a4 = (~a6) & M; c8 = (a6 - 1) & M      # a5 absent from C0, e8 = -1
    if fam == 'smallc9': c9 = rng.integers(-8, 9, size=(C, 1), dtype=np.int64) & M
    if fam == 'famBsmall': a7 = a6; a4 = (~a6) & M; c8 = (a6 - 1) & M; c9 = rng.integers(-8, 9, size=(C, 1), dtype=np.int64) & M
    a5 = rng.integers(0, 1 << 32, size=(C, S), dtype=np.int64)
    a3 = rng.integers(0, 1 << 32, size=(C, S), dtype=np.int64)
    cp = rng.integers(0, 1 << 32, size=(C, S), dtype=np.int64)
    T0 = (cp - G(a5, a4, a6, a7, c8, c9, a3)) & M
    tot = np.zeros(C, dtype=np.int64)
    for b in range(32):
        Tb = (cp - G(a5 ^ (1 << b), a4, a6, a7, c8, c9, a3)) & M
        tot += m.popcount(Tb ^ T0).sum(axis=1)
    mean = tot / (S * 32)
    i = int(np.argmin(mean))
    return fam, mean.min(), mean.mean(), mean.max(), (mean < 4).sum(), C, [int(x[i, 0]) for x in (a4, a6, a7, c8, c9)]
if __name__ == '__main__':
    t0 = time.time()
    jobs = [(1000 + j, 4096, 'rand') for j in range(64)] + [(2000 + j, 4096, 'famA') for j in range(16)] \
         + [(3000 + j, 4096, 'famB') for j in range(16)] + [(4000 + j, 4096, 'smallc9') for j in range(16)] \
         + [(5000 + j, 4096, 'famBsmall') for j in range(16)]
    agg = {}
    with Pool(10) as p:
        for fam, mn, me, mx, nb, C, prm in p.imap_unordered(worker, jobs):
            a = agg.setdefault(fam, [9e9, 0.0, 0, 0, 0, None])
            if mn < a[0]: a[0] = mn; a[5] = prm
            a[1] += me * C; a[2] = max(a[2], mx); a[3] += nb; a[4] += C
    for fam, (mn, s, mx, nb, n, prm) in agg.items():
        print(f"w=32 {fam:10s}: {n:,} contexts x {S} states x 32 flips: a5->C1 min {mn:.3f} mean {s/n:.3f} max {mx:.3f}; below 4 bits: {nb}; argmin ctx {[hex(x) for x in prm]}")
    print(f"{time.time()-t0:.0f}s")
