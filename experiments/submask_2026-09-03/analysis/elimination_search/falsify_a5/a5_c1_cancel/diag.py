#!/usr/bin/env python3
"""Diagnostics for the a5 -> C1 edge.

 A. Attribution.  The C1 target's a5-dependence (symext.py) is exactly
        D(a5) = -Sigma0(a5) + Sigma1(a5 + c9) - Maj(a5,a4,a3) + Ch(a5 + c9, e8, e7).
    Measure the single-bit weight of each atom alone and of the pairs, directly on
    the closed form (no round function needed), and confirm on the full state by
    freezing e9 (ILLEGAL, diagnostic only: it is what a9 = f(a5) would look like if
    the rest of the state did not notice).
 B. The two illegal ties, evaluated in the illegal frame (context re-solved after
    the flip, i.e. the context tracks the unknown): a9 solved for e9 = -1, and
    a4 = a5.  Both stay heavy; the Sigma0 atom migrates, it does not cancel.
"""
import numpy as np
import a5edge
from a5edge import U, popcount, rand_words
import sys
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import S0, S1, Ch, Maj, T2

np.seterr(over='ignore')
M = 0xFFFFFFFF
N = 2000
rng = np.random.default_rng(7)


def wt(f, n=N):
    x = rand_words(rng, n)
    acc = 0.0
    for i in range(32):
        acc += popcount(f(x) ^ f(x ^ U(1 << i))).mean()
    return acc / 32


import sys as _s
PART_A = len(_s.argv) < 2
if PART_A: print("A. closed-form atoms of the C1 a5-dependence (mean bits changed per single-bit flip, 2000 states x 32 bits)")
c = rand_words(rng, N); a4 = rand_words(rng, N); a3 = rand_words(rng, N); e8 = rand_words(rng, N); e7 = rand_words(rng, N)
print(f"  Sigma0(x)                              : {wt(lambda x: S0(x)):.2f}   (exactly 3 in XOR; it is a 3-bit XOR change, no carries)")
print(f"  Sigma1(x + c), random c                : {wt(lambda x: S1(x + c)):.2f}")
print(f"  Sigma1(x)  (c9 = 0)                    : {wt(lambda x: S1(x)):.2f}")
print(f"  Sigma1(x + c) - Sigma0(x), random c    : {wt(lambda x: S1(x + c) - S0(x)):.2f}")
print(f"  Sigma1(x) - Sigma0(x)      (c9 = 0)    : {wt(lambda x: S1(x) - S0(x)):.2f}")
print(f"  Sigma1(x) ^ Sigma0(x)      (XOR, ref)  : {wt(lambda x: S1(x) ^ S0(x)):.2f}   (GF(2)-linear, rank 31: exactly 6 or fewer)")
print(f"  -Sigma0(x) - Maj(x,a4,a3)              : {wt(lambda x: -S0(x) - Maj(x, a4, a3)):.2f}")
print(f"  full D, random c9, random e8,e7        : {wt(lambda x: -S0(x) + S1(x + c) - Maj(x, a4, a3) + Ch(x + c, e8, e7)):.2f}")
print(f"  full D, c9 = 0                         : {wt(lambda x: -S0(x) + S1(x) - Maj(x, a4, a3) + Ch(x, e8, e7)):.2f}")
print(f"  full D, c9 = 0, e8 = -1, a4 = a3 (Maj->a4, Ch->x|e7): {wt(lambda x: -S0(x) + S1(x) + (x | e7)):.2f}")
print(f"  full D, c9 = 0, e8 = 0, Maj killed (Ch-> ~x&e7)      : {wt(lambda x: -S0(x) + S1(x) + (~x & e7)):.2f}")
print(f"  full D, c9 = 0, e8 = e7 = const (Ch const), Maj killed: {wt(lambda x: -S0(x) + S1(x)):.2f}")

print("\n   full-state confirmation: freeze e9 (illegal diagnostic) in the closest-miss context")
base = {6: ('eq', 4), 7: ('eq', 6), 8: ('sat_r', M), 11: ('sat_r', 0)}
w, _ = a5edge.edge_weights(base)
print("   context fixed, nothing frozen        a5 ->", a5edge.fmt(w))
w, _ = a5edge.edge_weights(base, freeze_e={9: M})
print("   e9 frozen at -1 (Sigma0 atom alone)  a5 ->", a5edge.fmt(w))
w, _ = a5edge.edge_weights(base, freeze_e={9: 0})
print("   e9 frozen at 0                       a5 ->", a5edge.fmt(w))
w, _ = a5edge.edge_weights({**base, 9: ('sat_off', 0)})
print("   e9 - a5 = 0 through a9 (LEGAL)       a5 ->", a5edge.fmt(w))

print("\nB. illegal ties, illegal frame (context re-solved after every flip of a5)")


def illegal_frame(defs, n=200, seed=1):
    rng = np.random.default_rng(seed)
    free = {i: rand_words(rng, n) for i in range(4, 12)}
    dig = {i: rand_words(rng, n) for i in range(12, 20)}
    unk = {i: rand_words(rng, n) for i in (0, 1, 2, 3)}
    # a5 is treated as a context word by realise (unknown_words excludes 5), so
    # every definition that references it is re-solved after each flip.
    per = np.zeros((4, 32))
    base5 = free[5].copy()
    a = a5edge.realise(dict(defs), {**free, 5: base5}, dig, unk, unknown_words=(0, 1, 2, 3))
    T0 = a5edge.targets(a)[2]
    for i in range(32):
        f2 = dict(free); f2[5] = base5 ^ U(1 << i)
        a = a5edge.realise(dict(defs), f2, dig, unk, unknown_words=(0, 1, 2, 3))
        T1 = a5edge.targets(a)[2]
        for j in range(4):
            per[j, i] = popcount(T0[j] ^ T1[j]).mean()
    return {j: (per[j].mean(), per[j].min(), per[j].max()) for j in range(4)}


for name, defs in [("a9 tracks a5 so that e9 = -1 (plus a7=a6, e8=-1, a6=a4, e11=0)",
                    {6: ('eq', 4), 7: ('eq', 6), 8: ('sat_r', M), 9: ('sat_r', M), 11: ('sat_r', 0)}),
                   ("a9 tracks a5 so that e9 = 0", {6: ('eq', 4), 7: ('eq', 6), 8: ('sat_r', M), 9: ('sat_r', 0), 11: ('sat_r', 0)}),
                   ("a4 = a5 (the submask family with a5 as the unknown), e8 = e9 = -1",
                    {4: ('eq', 5), 8: ('sat_r', M), 9: ('sat_r', M)}),
                   ("a4 = ~a5, e8 = e9 = -1", {4: ('neq', 5), 8: ('sat_r', M), 9: ('sat_r', M)}),
                   ("a4 = rotr(a5, 13)", {4: ('rot', 5, 13), 7: ('eq', 6), 8: ('sat_r', M)}),
                   ]:
    w = illegal_frame(defs)
    print(f"   {name}\n      a5 ->", a5edge.fmt(w))
