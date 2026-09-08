#!/usr/bin/env python3
"""E8: rotational-XOR (RX).  Constants are handled for free in RX cryptanalysis,
so "the round constants kill it" is only true of PLAIN rotational cryptanalysis.
The honest question is whether RX beats it.  It cannot: the cost of an RX trail
is bounded below by the modular additions, and the best RX-difference through a
modular addition is the ZERO one, whose probability is the plain rotational
probability.  Verified exhaustively over ALL RX-difference triples at w = 6,7,8.
"""
import math
import numpy as np

for w in (6, 7, 8):
    mw = (1 << w) - 1
    for g in (1, 2):
        n = 1 << w
        x = np.arange(n, dtype=np.uint32)
        X, Y = np.meshgrid(x, x, indexing="ij")
        X = X.ravel(); Y = Y.ravel()

        def rl(v, k):
            k %= w
            return ((v << np.uint32(k)) | (v >> np.uint32(w - k))) & np.uint32(mw)

        S = (X + Y) & np.uint32(mw)
        rS = rl(S, g)
        rX = rl(X, g)
        rY = rl(Y, g)
        best = (0.0, None)
        # for each (d1,d2) the required d3 is determined per (x,y); the best d3 is
        # the modal value of the required-d3 histogram
        for d1 in range(n):
            A = rX ^ np.uint32(d1)
            for d2 in range(n):
                B = rY ^ np.uint32(d2)
                need = (((A + B) & np.uint32(mw)) ^ rS).astype(np.int64)
                cnt = np.bincount(need, minlength=n)
                m = int(cnt.max())
                p = m / need.size
                if p > best[0]:
                    best = (p, (d1, d2, int(cnt.argmax())))
        pred = 0.25 * (1 + 2.0 ** (g - w) + 2.0 ** (-g) + 2.0 ** (-w))
        print(f"w={w} g={g}: best RX probability over all {n**3:,} difference triples "
              f"= {best[0]:.8f} = 2^{math.log2(best[0]):.4f} at (d1,d2,d3)={best[1]}; "
              f"zero-difference (plain rotational) value = {pred:.8f}  "
              f"{'ZERO IS OPTIMAL' if abs(best[0]-pred) < 1e-12 else 'RX BEATS PLAIN'}")
