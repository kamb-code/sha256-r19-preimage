#!/usr/bin/env python3
"""The exact a5-dependence of the C1 lookup target when a8..a11 are free context
words (every tie among a4,a6,a7 and every value of a8..a11 is a point of this
space), and its verification against the full round function.

  T_1 = W17 - s1(W15) - W10 - W1,  and only W10 contains a5:
  W10 = a10 - T2(a9,a8,a7) - e6 - S1(e9) - Ch(e9,e8,e7) - K10
  e6 = a2 + a6 - S0(a5) - Maj(a5,a4,a3)
  e9 = a5 + c9,            c9 = a9 - S0(a8) - Maj(a8,a7,a6)
  e8 = c8 - Maj(a7,a6,a5), c8 = a4 + a8 - S0(a7)
  e7 = c7 - Maj(a6,a5,a4), c7 = a3 + a7 - S0(a6)
  =>  T_1 = c'(state) - G(a5),
  G(a5) = S0(a5) + Maj(a5,a4,a3) - S1(a5+c9) - Ch(a5+c9, c8-Maj(a7,a6,a5), c7-Maj(a6,a5,a4))

Context-controlled parameters of G: (a4, a6, a7, c8, c9); the map (a8,a9) ->
(c8,c9) is a bijection for fixed (a4,a6,a7), and a10, a11 only enter c'.
State: a3 (in Maj and c7), c' (uniform, through carries), a5.
"""
import numpy as np
from wmodel import Model


def G_of(m, a5, a4, a6, a7, c8, c9, a3):
    M = m.M
    c7 = (a3 + a7 - m.S0(a6)) & M
    y = (a5 + c9) & M
    X = (c8 - m.Maj(a7, a6, a5)) & M
    Y = (c7 - m.Maj(a6, a5, a4)) & M
    return (m.S0(a5) + m.Maj(a5, a4, a3) - m.S1(y) - m.Ch(y, X, Y)) & M


def verify(w, N=2000, seed=3):
    m = Model(w); rng = np.random.default_rng(seed)
    defs = {i: ('free',) for i in (4, 6, 7, 8, 9, 10, 11)}
    a = m.base_state(rng, N, 5, [4, 6, 7, 8, 9, 10, 11])
    m.realise(defs, a)
    _, _, T = m.targets(a)
    c8 = (a[4] + a[8] - m.S0(a[7])) & m.M
    c9 = (a[9] - m.S0(a[8]) - m.Maj(a[8], a[7], a[6])) & m.M
    cprime = (T[1] + G_of(m, a[5], a[4], a[6], a[7], c8, c9, a[3])) & m.M
    # change a5 (all bits, and random values): c' must not move
    bad = 0
    for b in range(w):
        a1 = dict(a); a1[5] = a[5] ^ (1 << b)
        _, _, T1 = m.targets(a1)
        cp1 = (T1[1] + G_of(m, a1[5], a[4], a[6], a[7], c8, c9, a[3])) & m.M
        bad += int((cp1 != cprime).sum())
    a2 = dict(a); a2[5] = rng.integers(0, 1 << w, size=N, dtype=np.int64)
    _, _, T2 = m.targets(a2)
    cp2 = (T2[1] + G_of(m, a2[5], a[4], a[6], a[7], c8, c9, a[3])) & m.M
    bad += int((cp2 != cprime).sum())
    print(f"w={w}: T_1 + G(a5) independent of a5 in {N*(w+1)-bad}/{N*(w+1)} checks -> {'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


if __name__ == "__main__":
    ok = all(verify(w) for w in (5, 6, 8, 12, 32))
    print("reduction", "VERIFIED" if ok else "BROKEN")
