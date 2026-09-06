#!/usr/bin/env python3
"""Probes beyond the tie menu, for the a5 -> C1 edge.

1. The kernel.  In every configuration with context independent of a5 (strict),
   the C1 target T_1 = W17 - s1(W15) - W10 - W1 contains a5 through W10 only:
       -W10 ∋ -S0(a5) + S1(e9) + Ch(e9, e8, e7) + Maj(a5, a4, a3)   (e9 = a5 + c9),
   so the one-input part is  k(a5) = S1(a5 + c9) - S0(a5)  (e9 unsaturated), or
   k(a5) = S0(c' - a5) - S0(a5) (e9 = t saturated through a9 = c' - a5).  We measure
   the weight of k alone per flipped bit, over c9 / c', and the weight of the
   RESIDUAL T_1 - k(a5), which must be mild if the decomposition is complete.
2. Arbitrary fixed values of e8, e10, e11 (random t, not just 0 / -1).
3. a9 = ~a5 explicitly (the one complement identity S0(~x) = ~S0(x)), with the
   collapses that keep a5 out of s1(W15).
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tie_scan as T
from tie_scan import S0, S1, s1, Ch, Maj, T2, popcount, u, U, M, rotr

rng = np.random.default_rng(7)
N = 200 * 32


def flips(x):
    bits = np.repeat(np.arange(32, dtype=np.uint32), x.size // 32)
    return x ^ (U(1) << bits).astype(U)


def kernel_weights():
    x = rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(U)
    xf = flips(x)
    print("kernel k(x) = S1(x + c9) - S0(x)   [e9 = x + c9 unsaturated]")
    for c9 in [0, M, 1, 0x80000000, 0x7FFFFFFF] + [int(v) for v in rng.integers(0, 1 << 32, 5, dtype=np.uint64)]:
        k = lambda z: S1(z + u(c9)) - S0(z)
        w = popcount(k(x) ^ k(xf)).mean()
        print(f"   c9=0x{c9:08x}: {w:5.2f} bits/flip")
    print("kernel k(x) = S0(c' - x) - S0(x)    [e9 saturated through a9 = c' - x]")
    for cp in [0, M, 1, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFE] + [int(v) for v in rng.integers(0, 1 << 32, 5, dtype=np.uint64)]:
        k = lambda z: S0(u(cp) - z) - S0(z)
        w = popcount(k(x) ^ k(xf)).mean()
        print(f"   c'=0x{cp:08x}: {w:5.2f} bits/flip" + ("   (S0(~x) = ~S0(x): k = -1 - 2 S0(x))" if cp == M else ""))
    print("for reference: S0(x) alone", f"{popcount(S0(x) ^ S0(xf)).mean():5.2f}",
          "  2*S0(x)", f"{popcount((U(2)*S0(x)) ^ (U(2)*S0(xf))).mean():5.2f}",
          "  x alone", f"{popcount(x ^ xf).mean():5.2f}")


def residual(defs, label):
    """Weight of T_1 and of T_1 - k(a5) under a5 flips."""
    order = T.order_of(defs)
    a, free = T.base_state(np.random.default_rng(1), 200)
    a = T.realise(defs, order, dict(a), free)
    e, W, T0 = T.targets(a)
    aa, ff = T.expand(a, free, 32)
    aa[5] = flips(aa[5])
    aa = T.realise(defs, order, aa, ff)
    e1, W1, T1 = T.targets(aa)
    sat9 = defs.get(9, ('free',))[0] in ('sat_r', 'neq', 'eq')
    if sat9:
        k0 = S0(a[9]) - S0(a[5]); k1 = S0(aa[9]) - S0(aa[5])
    else:
        k0 = S1(e[9]) - S0(a[5]); k1 = S1(e1[9]) - S0(aa[5])
    wT = popcount(T1[1] ^ np.tile(T0[1], 32)).mean()
    wR = popcount((T1[1] - k1) ^ np.tile(T0[1] - k0, 32)).mean()
    wK = popcount(k1 ^ np.tile(k0, 32)).mean()
    ctxdep = any(bool(np.any(aa[i] != np.tile(a[i], 32))) for i in range(4, 12) if i != 5)
    print(f"{label:60s} a5->C1 {wT:5.2f} = kernel {wK:5.2f} + residual {wR:5.2f}" + ("  [ctx depends on a5]" if ctxdep else ""))


if __name__ == '__main__':
    kernel_weights()
    print("\n--- decomposition T_1 = k(a5) + mild, per condition ---")
    residual({7: ('eq', 6), 8: ('sat_r', M)}, "a7=a6, e8=-1")
    residual({7: ('eq', 6), 8: ('sat_r', M), 11: ('sat_r', 0)}, "a7=a6, e8=-1, e11=0")
    residual({}, "random context")
    residual({7: ('eq', 6), 8: ('sat_r', M), 9: ('sat_r', M)}, "a7=a6, e8=e9=-1 (a9 = -1 - a5 + T2)")
    residual({7: ('eq', 6), 8: ('sat_r', M), 9: ('sat_r', 0)}, "a7=a6, e8=-1, e9=0")
    residual({7: ('rotr', 6, 11), 8: ('sat_r', M)}, "a7=ROTR^11(a6), e8=-1")
    residual({6: ('S0add', 4, 0), 7: ('eq', 6), 8: ('sat_r', M)}, "a6=S0(a4), a7=a6, e8=-1")
    print("\n--- arbitrary fixed values of e8, e10, e11 (random t), with a7 = a6 ---")
    for r in (8, 10, 11):
        ws = []
        for t in [int(v) for v in rng.integers(0, 1 << 32, 12, dtype=np.uint64)] + [0, M, 1, 0x80000000]:
            d = {7: ('eq', 6), r: ('sat_r', t)}
            res = T.measure(d, N=200, seed=1, words=(5,))
            ws.append(res[5]['mean'][1])
        print(f"   e{r} = t via a{r}: a5->C1 min {min(ws):5.2f} mean {np.mean(ws):5.2f} max {max(ws):5.2f} over 16 values of t")
    for t8 in [0, M] + [int(v) for v in rng.integers(0, 1 << 32, 3, dtype=np.uint64)]:
        for t9 in [int(v) for v in rng.integers(0, 1 << 32, 6, dtype=np.uint64)] + [0, M]:
            d = {7: ('eq', 6), 8: ('sat_r', t8), 9: ('sat_r', t9)}
            res = T.measure(d, N=200, seed=1, words=(5,))
            print(f"   e8=0x{t8:08x} e9=0x{t9:08x} via a8,a9: a5->C1 {res[5]['mean'][1]:5.2f}  C0 {res[5]['mean'][0]:5.2f} C2 {res[5]['mean'][2]:5.2f} C3 {res[5]['mean'][3]:5.2f}")
    print("\n--- a9 = ~a5 explicitly (S0(a9) = ~S0(a5)), the only exact one-input identity ---")
    for d, lab in [({9: ('neq', 5)}, "a9=~a5"),
                   ({7: ('eq', 6), 8: ('sat_r', M), 9: ('neq', 5)}, "a7=a6, e8=-1, a9=~a5"),
                   ({7: ('eq', 6), 8: ('sat_r', M), 9: ('neq', 5), 10: ('eq', 8)}, "... + a10=a8 (Maj(a10,a9,a8) collapses)"),
                   ({7: ('eq', 6), 8: ('sat_r', M), 9: ('neq', 5), 10: ('sat_r', 0)}, "... + e10=0 via a10"),
                   ({7: ('eq', 6), 8: ('sat_r', M), 9: ('neq', 5), 10: ('sat_r', M), 11: ('sat_r', 0)}, "... + e10=-1, e11=0"),
                   ({7: ('eq', 6), 8: ('sat_r', M), 9: ('eq', 5)}, "a7=a6, e8=-1, a9=a5 (illegal: equality with the unknown)")]:
        residual(d, lab)
