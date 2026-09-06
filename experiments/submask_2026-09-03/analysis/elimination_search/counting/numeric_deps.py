#!/usr/bin/env python3
"""Numeric confirmation on random states (no table needed).

(a) Frame B avalanche matrix at R=20 in the family-on-context variant
    (a5 = v, a8/a9 built so that e8 = e9 = -1 holds AT a4 = v), flipping one bit
    of each unknown a1..a4 and recording the Hamming weight of dC_j.
(b) Decomposition of the a4 -> C0 edge into its four carriers
    S0(a4), Maj(a4,a3,a2), S1(e8), Ch(e8,e7,e6).
(c) Exhaustive a4-freeze residual: for fixed (v,a2,a3,context), count the a4 in
    [0,2^32) for which the C0 residual (true minus provisional-at-v) vanishes.
    Prediction: exactly a4 = v plus Poisson(1) chance coincidences, i.e. the
    freeze passes with probability ~2^-32 and nothing a context word can do
    changes that, because S0 has one input.
(d) Per-context constants (KC0,KC1,KC2,kappa3) vs single-bit flips of each free
    context word v,a6,a7,a10,a11 (a8,a9 rebuilt): Hamming weights.  The
    multi-context amortisation needs a word with weight 0 on KC0..KC2 and >0 on
    kappa3.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, recover_W, U32, MISS, u32)

R = 20
rng = np.random.default_rng(20260905)
rnd = lambda: int(rng.integers(0, 1 << 32, dtype=np.uint64))
hw = lambda x: bin(x).count("1")


def family_context(v, a6, a7, a10, a11):
    a8 = (M - v + S0(a7) + Maj(a7, a6, v)) & M
    a9 = (M - v + S0(a8) + Maj(a8, a7, a6)) & M
    return {4: v, 5: v, 6: a6, 7: a7, 8: a8, 9: a9, 10: a10, 11: a11}


def state(ctx, unk, chain):
    a = dict(ctx); a.update(unk); a.update(chain)
    a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(R):
        e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M
    return a, e


def constraints(a, e):
    W = {r: recover_W(a, e, r) for r in range(R)}
    return [(s0(W[1 + t]) + W[t] + W[9 + t] + s1(W[14 + t]) - W[16 + t]) & M for t in range(4)]


def constants(ctx, chain):
    """KC0, KC1, KC2, kappa3 exactly as attack_context computes them."""
    a = dict(chain); a.update(ctx)
    a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(8, R):
        e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M
    a4, a5, a6, a7, a8, a9, a10 = (a[i] for i in range(4, 11))
    e8, e9, e10 = e[8], e[9], e[10]
    assert e8 == M and e9 == M
    T1_7 = (a7 - T2(a6, a5, a4)) & M
    c6 = (a6 - S0(a5)) & M
    W9base = ((a9 - T2(a8, a7, a6)) - K[9]) & M
    W10base = ((a10 - T2(a9, a8, a7)) - S1(e9) - K[10]) & M
    W11base = ((a[11] - T2(a10, a9, a8)) - S1(e10) - K[11]) & M
    Wr = {r: recover_W(a, e, r) for r in range(12, R)}
    K0p = (Wr[16] - s1(Wr[14])) & M
    K1p = (Wr[17] - s1(Wr[15])) & M
    K2p = (Wr[18] - s1(Wr[16])) & M
    K3p = (Wr[19] - s1(Wr[17]) - Wr[12]) & M
    W9hat = (W9base - (a5 - S0(a4) - Maj(a4, 0, 0)) - S1(e8)
             - Ch(e8, T1_7, (c6 - Maj(a5, a4, 0)) & M)) & M
    D = ((a6 - S0(a5) - Maj(a5, a4, 0)) + Ch(e9, e8, T1_7)) & M
    KC0 = (K0p - W9hat) & M
    KC1 = (K1p - W10base + D) & M
    KC2 = (K2p - W11base + T1_7 + Ch(e10, e9, e8)) & M
    return KC0, KC1, KC2, K3p


# ---------------------------------------------------------------- (a)
print("(a) frame B avalanche at R=20, family-on-context (a5=v, e8=e9=-1 at a4=v)")
print("    mean Hamming weight of dC_j for one flipped bit of a_k, 300 trials/cell")
T = 300
for base_at_v in (True, False):
    acc = np.zeros((4, 4))
    for _ in range(T):
        v = rnd()
        ctx = family_context(v, rnd(), rnd(), rnd(), rnd())
        chain = {i: rnd() for i in range(12, R)}
        unk = {0: rnd(), 1: rnd(), 2: rnd(), 3: rnd(), 4: v if base_at_v else rnd()}
        ctx_u = {k: ctx[k] for k in range(5, 12)}
        c0 = constraints(*state(ctx_u, unk, chain))
        for k in range(1, 5):
            u2 = dict(unk); u2[k] ^= 1 << int(rng.integers(0, 32))
            c1 = constraints(*state(ctx_u, u2, chain))
            for j in range(4):
                acc[j, k - 1] += hw(c0[j] ^ c1[j])
    acc /= T
    print(f"    base point a4 {'= v (e8 = -1 exactly)' if base_at_v else 'random (e8 random)'}:")
    print("           " + "".join(f"{'a'+str(k):>7}" for k in range(1, 5)))
    for j in range(4):
        print(f"      C{j}: " + "".join(f"{acc[j, k]:7.1f}" for k in range(4)))

# ---------------------------------------------------------------- (b)
print("\n(b) decomposition of the a4 -> C0 edge (a4 enters C0 only through W9):")
print("    W9 = a9 - T2(a8,a7,a6) - e5 - S1(e8) - Ch(e8,e7,e6) - K9,")
print("    e5 = a1 + a5 - S0(a4) - Maj(a4,a3,a2),  e8 = a4 + a8 - T2(a7,a6,a5),")
print("    e6 = a2 + a6 - S0(a5) - Maj(a5,a4,a3),  e7 = a3 + a7 - S0(a6) - Maj(a6,a5,a4)")
acc = {}
names = ["S0(a4) in e5", "Maj(a4,a3,a2) in e5", "S1(e8)", "Ch(e8,e7,e6) incl. e6,e7", "all four (= dW9)"]
for n in names: acc[n] = 0.0
for _ in range(T):
    v = rnd(); ctx = family_context(v, rnd(), rnd(), rnd(), rnd())
    a5, a6, a7, a8 = ctx[5], ctx[6], ctx[7], ctx[8]
    a1, a2, a3 = rnd(), rnd(), rnd()
    a4 = v; a4p = a4 ^ (1 << int(rng.integers(0, 32)))
    def parts(x):     # the a4-dependent pieces of W9 as functions of a4 = x
        e8 = (x + a8 - T2(a7, a6, a5)) & M
        e7 = (a3 + a7 - S0(a6) - Maj(a6, a5, x)) & M
        e6 = (a2 + a6 - S0(a5) - Maj(a5, x, a3)) & M
        return (S0(x), Maj(x, a3, a2), S1(e8), Ch(e8, e7, e6))
    p0, p1 = parts(a4), parts(a4p)
    acc[names[0]] += hw(p0[0] ^ p1[0]); acc[names[1]] += hw(p0[1] ^ p1[1])
    acc[names[2]] += hw(p0[2] ^ p1[2]); acc[names[3]] += hw(p0[3] ^ p1[3])
    w0 = (p0[0] + p0[1] - p0[2] - p0[3]) & M; w1 = (p1[0] + p1[1] - p1[2] - p1[3]) & M
    acc[names[4]] += hw(w0 ^ w1)
for n in names:
    print(f"    {n:<28} {acc[n]/T:5.2f} bits per flipped bit of a4")

# ---------------------------------------------------------------- (c)
print("\n(c) exhaustive a4-freeze residual in C0 (frame B: freeze a4 at v inside C0, solve, then check)")
print("    count of a4 in [0,2^32) with  [S0(a4)+Maj(a4,a3,a2)-S1(e8(a4))-Ch(e8(a4),e7(a4),e6(a4))]")
print("      == same at a4 = v      (all other words fixed).  Expect: {a4=v} + Poisson(1) coincidences.")
CH = 1 << 26
for trial in range(4):
    v = rnd(); ctx = family_context(v, rnd(), rnd(), rnd(), rnd())
    a5, a6, a7, a8 = (u32(ctx[i]) for i in (5, 6, 7, 8))
    a2, a3 = u32(rnd()), u32(rnd())
    t0 = time.time()
    def phi_of(A4):
        e8 = (A4 + a8 - T2(a7, a6, a5)) & MISS
        e7 = (a3 + a7 - S0(a6) - Maj(a6, a5, A4)) & MISS
        e6 = (a2 + a6 - S0(a5) - Maj(a5, A4, a3)) & MISS
        return (S0(A4) + Maj(A4, a3, a2) - S1(e8) - Ch(e8, e7, e6)) & MISS
    target = int(phi_of(np.array([v], dtype=U32))[0])
    s0v = int(S0(u32(v)))
    hits = []; hits_s0 = 0
    for lo in range(0, 1 << 32, CH):
        A4 = np.arange(lo, lo + CH, dtype=np.uint64).astype(U32)
        idx = np.nonzero(phi_of(A4) == U32(target))[0]
        hits += [lo + int(i) for i in idx]
        # the S0-only part: even with e8 saturated (a8 tracking a4) and the Maj/Ch
        # terms gone, the residual vanishes only at a4 = v (S0 is a bijection)
        hits_s0 += int((S0(A4) == U32(s0v)).sum())
    print(f"    trial {trial}: v=0x{v:08x}  full residual zero at a4 in {[hex(h) for h in hits]}"
          f"  ({len(hits)} values);  S0(a4)=S0(v) at {hits_s0} value(s)   [{time.time()-t0:.0f}s]")

# ---------------------------------------------------------------- (d)
print("\n(d) per-context constants vs free context words (family; a8,a9 rebuilt after each flip)")
print("    mean Hamming weight of d(KC0,KC1,KC2,kappa3) for one flipped bit, 300 trials/cell")
words = ["v", "a6", "a7", "a10", "a11"]
acc = np.zeros((5, 4)); zero_kc = np.zeros(5, dtype=int)
for _ in range(T):
    base = [rnd() for _ in range(5)]
    chain = {i: rnd() for i in range(12, R)}
    k0 = constants(family_context(*base), chain)
    for wi in range(5):
        b2 = list(base); b2[wi] ^= 1 << int(rng.integers(0, 32))
        k1 = constants(family_context(*b2), chain)
        for j in range(4):
            acc[wi, j] += hw(k0[j] ^ k1[j])
        if k0[0] == k1[0] and k0[1] == k1[1] and k0[2] == k1[2]:
            zero_kc[wi] += 1
acc /= T
print("             KC0    KC1    KC2  kappa3   #flips leaving (KC0,KC1,KC2) unchanged")
for wi, w in enumerate(words):
    print(f"      {w:<4}" + "".join(f"{acc[wi, j]:7.1f}" for j in range(4)) + f"        {zero_kc[wi]}/{T}")
print("    (a candidate set is a function of (KC0,KC1,KC2,v) alone: attack_context uses nothing else"
      " before the c3 test)")
