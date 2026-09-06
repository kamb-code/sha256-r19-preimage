#!/usr/bin/env python3
"""Independent re-measurement of explicit contexts with the FULL round function
(no reduction), at any width, plus the exact fibre structure of a5 -> C1 target.

A context is (a4, a6, a7, c8, c9) with a8 = c8 - a4 + S0(a7),
a9 = c9 + S0(a8) + Maj(a8,a7,a6); a10, a11 random per state (they only shift
the target by a constant).  For each of N random states: all 2^w values of a5
(w <= 12) or N x w single-bit flips (w = 32):
  * mean Hamming weight of dT_j per flipped bit of a5, j = 0..3
  * per-bit sensitivity: fraction of (state, a5) for which flipping bit b of a5
    changes T_1 at all
  * exact fibre: number of a5 with T_1(a5) == T_1(a5_true), and the image size
    of a5 -> T_1 (w <= 12)
Usage: python3 lift.py W a4 a6 a7 c8 c9 [N]   (hex or int; 'r' = random per state)
"""
import sys
import numpy as np
from wmodel import Model


def measure(w, a4, a6, a7, c8, c9, N=64, seed=1, verbose=True, rng=None):
    m = Model(w); M = m.M
    rng = rng or np.random.default_rng(seed)
    NW = 1 << w
    full = w <= 12
    R_ = NW if full else 1
    tot = np.zeros(4); cnt = 0
    sens = np.zeros(w)
    fibres = []; images = []
    for t in range(N):
        rnd = lambda: int(rng.integers(0, NW))
        A4 = rnd() if a4 is None else a4
        A6 = rnd() if a6 is None else a6
        A7 = rnd() if a7 is None else a7
        C8 = rnd() if c8 is None else c8
        C9 = rnd() if c9 is None else c9
        A8 = (C8 - A4 + m.S0(A7)) & M
        A9 = (C9 + m.S0(A8) + m.Maj(A8, A7, A6)) & M
        a = {-1: m.IV[0], -2: m.IV[1], -3: m.IV[2], -4: m.IV[3]}
        n = NW if full else 1
        for i in range(4):
            a[i] = np.full(n, rnd(), dtype=np.int64)
        for i in range(12, 20):
            a[i] = np.full(n, rnd(), dtype=np.int64)
        for i, v in ((4, A4), (6, A6), (7, A7), (8, A8), (9, A9), (10, rnd()), (11, rnd())):
            a[i] = np.full(n, v, dtype=np.int64)
        a[5] = np.arange(NW, dtype=np.int64) if full else np.full(1, rnd(), dtype=np.int64)
        _, _, T0 = m.targets(a)
        if full:
            vals, counts = np.unique(T0[1], return_counts=True)
            images.append(vals.size)
            fibres.append(counts[np.searchsorted(vals, T0[1][rnd()])])
        for b in range(w):
            a1 = dict(a); a1[5] = a[5] ^ (1 << b)
            _, _, T1 = m.targets(a1)
            for j in range(4):
                tot[j] += m.popcount(T1[j] ^ T0[j]).sum()
            sens[b] += ((T1[1] ^ T0[1]) != 0).sum()
            cnt += n
    mean = tot / cnt
    sens = sens / (cnt / w)
    out = dict(mean=mean, sens=sens, image_mean=float(np.mean(images)) if images else None,
               image_min=int(np.min(images)) if images else None,
               fibre_mean=float(np.mean(fibres)) if fibres else None,
               fibre_max=int(np.max(fibres)) if fibres else None)
    if verbose:
        fmt = lambda x: 'r' if x is None else f'{x:#x}'
        print(f"w={w} ctx a4={fmt(a4)} a6={fmt(a6)} a7={fmt(a7)} c8={fmt(c8)} c9={fmt(c9)}: "
              f"a5->C0..C3 = {np.round(mean, 3).tolist()}  "
              + (f"image {out['image_mean']:.1f}/{NW} (min {out['image_min']}), fibre mean {out['fibre_mean']:.2f} max {out['fibre_max']}"
                 if full else "") + f"\n   per-bit sensitivity of C1: {np.round(sens, 2).tolist()}")
    return out


if __name__ == "__main__":
    w = int(sys.argv[1])
    p = [None if x == 'r' else int(x, 0) for x in sys.argv[2:7]]
    N = int(sys.argv[7]) if len(sys.argv) > 7 else 64
    measure(w, *p, N=N)
