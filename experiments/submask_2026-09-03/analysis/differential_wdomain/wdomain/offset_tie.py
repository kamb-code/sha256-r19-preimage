#!/usr/bin/env python3
"""Offset alignment, the one idea that falls out of the W-domain analysis.

W_i (equivalently a_i) reaches the next rounds through TWO one-input functions
of itself with DIFFERENT additive offsets:  Sigma0(a_i) inside T2, and
Sigma1(e_i) = Sigma1(a_i + c) inside T1 four rounds on.  A single 2^32 table
can invert one-input functions, so the obstruction is not "one-input" as such;
it is that the two copies carry different offsets, forcing the two-axis table
F_c(x) = Sigma0(x) + Sigma1(x + c).

There is a context condition that ALIGNS them, and it is outside the menu the
elimination search enumerated ({equality, complement among words; e_r in
{0,-1}}):

    e_{k+4} = a_k     <=>   T1^{(k+4)} = 0   <=>   a_{k+4} = T2(a_{k+3}, a_{k+2}, a_{k+1})

For k = 4 at R = 20 that reads a_8 = T2(a_7, a_6, a_5), which is FREE: it fixes
a context word from three other context words, exactly as the submask family
fixes a_8 to force e_8 = -1.  (The two uses of a_8 are mutually exclusive.)

Measured here: the a_4 -> C_0 feedback edge weight under the tie, against the
random-context baseline of 12.2 bits and the 2-bit absorption threshold.
"""
import sys, random
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, IV, S0, S1, s0, s1, Ch, Maj, T2, digest,
                            recover_W, backward_chain)

R = 20
rng = random.Random(20260908)
RWd = lambda: rng.getrandbits(32)
pc = lambda x: bin(x & M).count("1")


def C(free, ab):
    a = dict(ab); a.update(free)
    a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(R):
        e[r] = (a[r-4] + a[r] - T2(a[r-1], a[r-2], a[r-3])) & M
    W = {r: recover_W(a, e, r) for r in range(R)}
    return [(W[16+j] - s1(W[14+j]) - W[9+j] - s0(W[1+j]) - W[j]) & M
            for j in range(R - 16)], e


MODES = {
    "random context (baseline)":      lambda c: c,
    "submask family: e8 = e9 = -1":   None,        # filled below
    "offset tie: a8 = T2(a7,a6,a5)  => e8 = a4": None,
    "offset tie k=5: a9 = T2(a8,a7,a6) => e9 = a5": None,
    "both offset ties (e8=a4, e9=a5)": None,
}


def ctx_random():
    return {i: RWd() for i in range(4, 12)}


def ctx_family():
    c = ctx_random()
    c[8] = (M - c[4] + S0(c[7]) + Maj(c[7], c[6], c[5])) & M
    c[9] = (M - c[5] + S0(c[8]) + Maj(c[8], c[7], c[6])) & M
    c[5] = c[4]
    c[8] = (M - c[4] + S0(c[7]) + Maj(c[7], c[6], c[5])) & M
    c[9] = (M - c[5] + S0(c[8]) + Maj(c[8], c[7], c[6])) & M
    return c


def ctx_tie8():
    c = ctx_random()
    c[8] = T2(c[7], c[6], c[5])          # e8 = a4
    return c


def ctx_tie9():
    c = ctx_random()
    c[9] = T2(c[8], c[7], c[6])          # e9 = a5
    return c


def ctx_tie89():
    c = ctx_random()
    c[8] = T2(c[7], c[6], c[5])
    c[9] = T2(c[8], c[7], c[6])
    return c


builders = [("random context (baseline)", ctx_random),
            ("submask family e8=e9=-1  ", ctx_family),
            ("offset tie e8 = a4       ", ctx_tie8),
            ("offset tie e9 = a5       ", ctx_tie9),
            ("both ties e8=a4, e9=a5   ", ctx_tie89)]

NT = 300
print(f"R={R}: bits moved in C_j per flipped bit of a_4 (absorption needs <= 2)")
print("  context                      e8            C0    C1    C2    C3")
h = digest([RWd() for _ in range(16)], R)
ab, _ = backward_chain(h, R)
for name, mk in builders:
    tot = [0] * 4
    e8s = set()
    for _ in range(NT):
        c = mk()
        free = {0: RWd(), 1: RWd(), 2: RWd(), 3: RWd()}
        free.update(c)
        c0, e = C(free, ab)
        e8s.add("=-1" if e[8] == M else ("=a4" if e[8] == free[4] else "free"))
        f2 = dict(free); f2[4] = free[4] ^ (1 << rng.randrange(32))
        # keep the tie/family condition consistent with the new a_4
        if mk is ctx_family:
            f2[8] = (M - f2[4] + S0(f2[7]) + Maj(f2[7], f2[6], f2[5])) & M
            f2[9] = (M - f2[5] + S0(f2[8]) + Maj(f2[8], f2[7], f2[6])) & M
        c1, _ = C(f2, ab)
        for j in range(4):
            tot[j] += pc(c0[j] ^ c1[j])
    print(f"  {name}  {'/'.join(sorted(e8s)):>6}   "
          + "  ".join(f"{t/NT:5.2f}" for t in tot))

print()
print("The tie e8 = a4 aligns the two offsets, so C_0's a_4-dependence becomes")
print("Sigma0(a_4) + Sigma1(a_4) + Maj/Ch terms -- a SINGLE-axis one-input")
print("function.  Whether that is a cut depends on the Hamming weight above:")
print("absorption of a_4 by C_3 needs a_4 MILD in C_0 (<= 2 bits), not merely")
print("one-input, because C_0 is already spent absorbing a_1.")
