#!/usr/bin/env python3
"""R21 joint-tail audit: does the tail-lift frame remove the two exact residuals?

Frame 1 (current):  C0,C1,C2 determine a1,a2,a3;  C3 and C4 are residual checks.
Frame 2 (tail-lift): sigma0 is a GF(2) bijection, so for fixed preceding data
      W4 = s0^{-1}(K3p - W3),   W5 = s0^{-1}(K4p - W4)
   recover W4, W5 and hence a4, a5.  The claim to test is that this converts two
   2^-w residual checks into an exact triangular recovery.

The audit note's decision rule: if the reformulation only MOVES the unresolved
condition, the consistency residual (tail-derived a4,a5 against the a4,a5 the
context assumed) is a fresh full-width condition and nothing is gained.

Positive control: on a true solution every identity must hold exactly.
"""
from __future__ import annotations
import numpy as np, sys
from model import Model


def instance(m, rng):
    """A genuine instance: random message, its true state, its digest."""
    W = [int(x) for x in rng.integers(0, 1 << m.w, 16, dtype=np.int64)]
    a, e, Wf = m.forward(W)
    return W, a, e, Wf


def Kp(m, a, e, j):
    """K_{j}' = W_{16+j} - s1(W_{14+j}) - W_{9+j}, the a_{j+1}-free part of C_j."""
    W = lambda r: m.recoverW(a, e, r)
    return (W(16 + j) - m.s1(W(14 + j)) - W(9 + j)) & m.M


def control(w, trials=200, seed=1):
    """On true solutions, check the tail-lift identities exactly."""
    m = Model(w, R=21)
    rng = np.random.default_rng(seed)
    okC, okW, okA = 0, 0, 0
    for _ in range(trials):
        W, a, e, Wf = instance(m, rng)
        # C_j identity itself: W_{16+j} = s1(W_{14+j}) + W_{9+j} + s0(W_{1+j}) + W_j
        good = all((Wf[16+j] == (m.s1(Wf[14+j]) + Wf[9+j] + m.s0(Wf[1+j]) + Wf[j]) & m.M)
                   for j in range(5))
        okC += good
        # tail-lift: recover W4 from C3 and W5 from C4
        K3p, K4p = Kp(m, a, e, 3), Kp(m, a, e, 4)
        W4 = m.s0_inv((K3p - Wf[3]) & m.M)
        W5 = m.s0_inv((K4p - W4) & m.M)
        okW += (int(W4) == Wf[4]) and (int(W5) == Wf[5])
        # and W4 -> a4, W5 -> a5 given the preceding state
        a4 = (int(W4) + m.T2(a[3], a[2], a[1]) + e[0] + m.S1(e[3])
              + m.Ch(e[3], e[2], e[1]) + m.K[4]) & m.M
        a5 = (int(W5) + m.T2(a4, a[3], a[2]) + e[1] + m.S1(e[4])
              + m.Ch(e[4], e[3], e[2]) + m.K[5]) & m.M
        okA += (a4 == a[4]) and (a5 == a[5])
    return okC, okW, okA, trials


def dependency(w, trials=400, seed=2):
    """Does K3p / K4p depend on a4 and a5?  That is where the cycle lives."""
    m = Model(w, R=21)
    rng = np.random.default_rng(seed)
    moved = {'K3p_on_a4': 0, 'K3p_on_a5': 0, 'K4p_on_a4': 0, 'K4p_on_a5': 0}
    for _ in range(trials):
        W, a, e, Wf = instance(m, rng)
        base = (Kp(m, a, e, 3), Kp(m, a, e, 4))
        for word, tag in ((4, 'a4'), (5, 'a5')):
            a2 = dict(a); e2 = dict(e)
            a2[word] = (a2[word] ^ 1) & m.M              # flip one bit
            # recompute e and the later state consistently from a
            for r in range(word, 21):
                e2[r] = (a2[r-4] + a2[r] - m.T2(a2[r-1], a2[r-2], a2[r-3])) & m.M
            k3, k4 = Kp(m, a2, e2, 3), Kp(m, a2, e2, 4)
            moved[f'K3p_on_{tag}'] += (k3 != base[0])
            moved[f'K4p_on_{tag}'] += (k4 != base[1])
    return moved, trials


def consistency_residual(w, trials=200000, seed=3):
    """Frame 2's new condition: the tail-derived (a4,a5) must equal the assumed
    ones.  Measure P(match) against the 2^-2w a random condition would give."""
    m = Model(w, R=21)
    rng = np.random.default_rng(seed)
    hit4 = hit5 = hit_both = 0
    for _ in range(trials):
        W, a, e, Wf = instance(m, rng)
        # perturb the unknowns a0..a3 away from the true solution, keep the
        # context and digest: this is the situation the sweep is actually in
        a2 = dict(a)
        for i in range(4):
            a2[i] = int(rng.integers(0, 1 << m.w))
        e2 = dict(e)
        for r in range(0, 21):
            e2[r] = (a2[r-4] + a2[r] - m.T2(a2[r-1], a2[r-2], a2[r-3])) & m.M
        Wr = lambda r: m.recoverW(a2, e2, r)
        K3p = (Wr(19) - m.s1(Wr(17)) - Wr(12)) & m.M
        K4p = (Wr(20) - m.s1(Wr(18)) - Wr(13)) & m.M
        W4 = m.s0_inv((K3p - Wr(3)) & m.M)
        W5 = m.s0_inv((K4p - W4) & m.M)
        a4t = (int(W4) + m.T2(a2[3], a2[2], a2[1]) + e2[0] + m.S1(e2[3])
               + m.Ch(e2[3], e2[2], e2[1]) + m.K[4]) & m.M
        a5t = (int(W5) + m.T2(a4t, a2[3], a2[2]) + e2[1] + m.S1(e2[4])
               + m.Ch(e2[4], e2[3], e2[2]) + m.K[5]) & m.M
        h4 = (a4t == a2[4]); h5 = (a5t == a2[5])
        hit4 += h4; hit5 += h5; hit_both += (h4 and h5)
    return hit4, hit5, hit_both, trials


if __name__ == "__main__":
    print("=== positive control: tail-lift identities on TRUE solutions ===")
    for w in (6, 8, 10, 12, 16):
        okC, okW, okA, n = control(w)
        print(f"  w={w:2d}: C_j identities {okC}/{n}   W4,W5 recovered {okW}/{n}"
              f"   a4,a5 recovered {okA}/{n}")

    print("\n=== does the cycle exist? (K3p/K4p dependence on a4, a5) ===")
    for w in (8, 12):
        mv, n = dependency(w)
        print(f"  w={w:2d}: " + "  ".join(f"{k} {v}/{n}" for k, v in mv.items()))

    print("\n=== frame 2's new condition: P(tail-derived a4,a5 == assumed) ===")
    print(f"  {'w':>3} {'trials':>8} {'a4 hit':>8} {'a5 hit':>8} {'both':>6}"
          f" {'-log2(P both)':>14} {'2w (no gain)':>13}")
    for w, n in ((5, 400000), (6, 400000), (7, 400000), (8, 400000)):
        h4, h5, hb, t = consistency_residual(w, trials=n)
        import math
        lg = (-math.log2(hb / t)) if hb else float('inf')
        print(f"  {w:>3} {t:>8} {h4:>8} {h5:>8} {hb:>6} {lg:>14.2f} {2*w:>13}")
