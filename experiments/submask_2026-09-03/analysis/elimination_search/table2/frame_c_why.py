#!/usr/bin/env python3
"""Why the Frame C table (Sigma0(a4) - Sigma1(a4 + c8)) could not work, in the
dependency terms of this note.  Full width, no big tables.

C0 written with a4 unknown (a0 swept, a1 the sigma0-home):
   C0 = [sigma0(W1) - W1]  +  P(a4)  +  Q(a4; a2, a3)  + const
   P(a4)         = -Sigma0(a4) + Sigma1(a4 + c8)          (one-input, per-context)
   Q(a4; a2,a3)  = Ch(a4 + c8, e7(a3,a4), e6(a2,a3,a4)) - Maj(a4, a3, a2)
so (1) the a4-dependence of C0 is NOT a one-input map: Q moves with a2, a3;
   (2) C0 carries two non-deferrable unknowns, a1 (inside sigma0(W1) - W1) and
       a4 (inside P): a table can invert either given the other, never both;
   (3) a1 is absorbable by C0 only, so Frame C has to sweep a1 too, and then
       C3 is still a 2^-32 filter: the frame moves the sweep, it removes no bits.

We measure: Hamming weights of dC0, dP, dQ per flipped bit of a4; of dQ per
flipped bit of a2 and a3; and the fixed-point accounting.
"""
import sys
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, recover_W
R = 20

def state(a):
    a = dict(a); a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(0, R):
        e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M
    return a, e

def C0(a):
    a, e = state(a)
    W = {r: recover_W(a, e, r) for r in range(0, R)}
    return (W[16] - s1(W[14]) - W[9] - s0(W[1]) - W[0]) & M

def P(a):
    c8 = (a[8] - T2(a[7], a[6], a[5])) & M
    return (-S0(a[4]) + S1((a[4] + c8) & M)) & M

def Q(a):
    a, e = state(a)
    return (Ch(e[8], e[7], e[6]) - Maj(a[4], a[3], a[2])) & M

def hw(x): return bin(x).count('1')

rng = np.random.default_rng(11)
N = 400
acc = {k: [] for k in ('C0|a4', 'P|a4', 'Q|a4', 'Q|a2', 'Q|a3', 'rest|a4', 'C0-P-Q|a4')}
for _ in range(N):
    a = {i: int(rng.integers(0, 1 << 32, dtype=np.uint64)) for i in range(0, R)}
    bit = 1 << int(rng.integers(0, 32))
    for k, idx in (('a4', 4), ('a2', 2), ('a3', 3)):
        b = dict(a); b[idx] ^= bit
        if k == 'a4':
            acc['C0|a4'].append(hw(C0(a) ^ C0(b)))
            acc['P|a4'].append(hw(P(a) ^ P(b)))
            acc['Q|a4'].append(hw(Q(a) ^ Q(b)))
            # what is left of C0's a4-dependence after removing P and Q: must be exactly 0
            ra = (C0(a) - P(a) - Q(a)) & M; rb = (C0(b) - P(b) - Q(b)) & M
            acc['C0-P-Q|a4'].append(hw(ra ^ rb))
        else:
            acc[f'Q|{k}'].append(hw(Q(a) ^ Q(b)))
print("mean Hamming weight of the change per flipped bit (400 random full-width states, random context):")
for k, v in acc.items():
    if v: print(f"   {k:>10}: {np.mean(v):6.2f}   (nonzero in {np.count_nonzero(v)}/{len(v)})")
print("\n=> C0 = [sigma0(W1)-W1 part] + P(a4) + Q(a4;a2,a3) + const exactly (residual 0 above);")
print("   P is the one-input per-context map Frame C tabulated; Q moves with a2 and a3, so C0's")
print("   a4-dependence is a THREE-input function and no one-input table inverts it; and C0 still")
print("   contains a1 non-deferrably, so a table for a4 leaves a1 without a home.")

# accounting
print("\nFrame C accounting: sweeps (a0,a1) pairs, C3 remains a 2^-32 filter.")
for p_conv, label in ((5.34e-5, "R=19 loop rate (paper)"), (2.82e-6, "R=19 loop rate in the frame_c_test3 harness"),
                      (3.0 / 8388608, "95% upper bound from 0 of 8,388,608 pairs")):
    cost = 1.0 / (p_conv * 2.0 ** -32)
    print(f"   if convergence were {p_conv:.2e} per pair ({label}): cost = 2^{np.log2(cost):.1f} pairs per preimage"
          f"  [submask frame: 2^45.4 swept a0]")
