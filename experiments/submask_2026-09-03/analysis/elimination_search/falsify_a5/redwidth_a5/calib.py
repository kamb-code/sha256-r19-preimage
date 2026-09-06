#!/usr/bin/env python3
"""Calibration of edge-weight scales per width: what a heavy edge, an S0-only
edge, a linear edge and a Maj/Ch-only (mild) edge measure, so reduced-width
numbers can be read.  Also the a4 -> C0 control and the prior best a5 config."""
import numpy as np
from wmodel import Model


def synth(m, rng, f, N=4000):
    x = rng.integers(0, 1 << m.w, size=N, dtype=np.int64)
    c = rng.integers(0, 1 << m.w, size=N, dtype=np.int64)
    r1 = rng.integers(0, 1 << m.w, size=N, dtype=np.int64)
    r2 = rng.integers(0, 1 << m.w, size=N, dtype=np.int64)
    base = (f(m, x, r1, r2) + c) & m.M
    tot = 0
    for b in range(m.w):
        y = (f(m, x ^ (1 << b), r1, r2) + c) & m.M
        tot += m.popcount(y ^ base).sum()
    return tot / (N * m.w)


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    print(f"{'w':>3} {'a4->C0 ctl':>11} {'a5->C1 prior':>13} {'a5->C0 prior':>13} {'a3->C0 mild':>12} | "
          f"{'S0(x)+c':>8} {'S0-S1(x+k)':>11} {'x+c lin':>8} {'Maj+c':>6} {'Ch+c':>6}")
    for w in (5, 6, 8, 12, 16, 32):
        m = Model(w)
        ctl = m.edge_weights({i: ('free',) for i in (5, 6, 7, 8, 9, 10, 11)}, 4, rng, N=400)['mean']
        pri = m.edge_weights({4: ('free',), 6: ('free',), 7: ('eq', 6), 8: ('sat_r', m.M), 9: ('free',),
                              10: ('free',), 11: ('free',)}, 5, rng, N=400)['mean']
        mild = m.edge_weights({i: ('free',) for i in range(4, 12)}, 3, rng, N=400)['mean']
        s0 = synth(m, rng, lambda m, x, r1, r2: m.S0(x))
        s01 = synth(m, rng, lambda m, x, r1, r2: (m.S0(x) - m.S1((x + r1) & m.M)) & m.M)
        lin = synth(m, rng, lambda m, x, r1, r2: x)
        maj = synth(m, rng, lambda m, x, r1, r2: m.Maj(x, r1, r2))
        ch = synth(m, rng, lambda m, x, r1, r2: m.Ch(r1, x, r2))
        print(f"{w:>3} {ctl[0]:>11.2f} {pri[1]:>13.2f} {pri[0]:>13.2f} {mild[0]:>12.2f} | "
              f"{s0:>8.2f} {s01:>11.2f} {lin:>8.2f} {maj:>6.2f} {ch:>6.2f}")
