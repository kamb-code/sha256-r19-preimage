#!/usr/bin/env python3
"""Frame B (a4 unknown): the C0 residual as a function of a4, with a0..a3 and the
context fixed.  Is there ANY collapse in it -- a provisional value that the true a4
reproduces with probability much above 2^-32?

Measures, on random instances in the shifted family (a5=a6=w, e9=e10=-1):
  (1) ladder P(low k bits of r0(a4) - r0(a4') == 0) over random a4, a4' : 2^-k if none
  (2) same with a4' restricted to the 'admissible' set of the collapsed condition
      Maj(a4,a3,a2)==a3 (a4 = a3 on bits where a2 != a3, free elsewhere)
  (3) the heavy kernel psi(a4) = S0(a4) - S1(a4 + c8), and with c8 = 0 (a8 = T2(a7,a6,a5)):
      collision rate of psi on 2^24 samples vs a random map.
"""
import sys, struct
import numpy as np
sys.path.insert(0, '/home/administrator/sha/publish/code')
from submask_family import (M, U32, MISS, ZERO, K, IV, S0, S1, s0, s1, Ch, Maj, T2, u32,
                            digest, recover_W, backward_chain)
sys.path.insert(0, '.')
from edgew20 import instance, full_state, residuals
R = 20
rng = np.random.default_rng(11)
KS = list(range(0, 33, 4))

def r0_vec(a, A4):
    """C0 residual for an array of a4 values, everything else from dict a (frame B)."""
    a0, a1, a2, a3 = (u32(a[i]) for i in range(4)); w = u32(a[5]); a7 = u32(a[7]); a8 = u32(a[8])
    e = full_state(a); W = {r: recover_W(a, e, r) for r in range(R)}
    # W16, W14, W1, W0 do not depend on a4; W9 does
    K0p = u32((W[16] - s1(W[14]) - s0(W[1]) - W[0]) & M)
    W9base = u32((a[9] - T2(a[8], a[7], a[6]) - K[9]) & M)
    E5 = (a1 + w - (S0(A4) + Maj(A4, a3, a2))) & MISS
    E6 = (a2 + w - (S0(w) + Maj(w, A4, a3))) & MISS
    E7 = (a3 + a7 - (S0(w) + Maj(w, w, A4))) & MISS
    E8 = (A4 + a8 - (S0(a7) + Maj(a7, w, w))) & MISS
    W9 = (W9base - E5 - S1(E8) - Ch(E8, E7, E6)) & MISS
    return (K0p - W9) & MISS

def ladder(diff, label):
    n = diff.size
    print(f'  {label}: n = {n:,}')
    for k in KS:
        c = int(((diff & U32((1 << k) - 1)) == ZERO).sum()); pred = n / 2 ** k
        if pred < 0.2 and c == 0: break
        print(f'     k={k:2d}  {c:>9,}  predicted {pred:>11.1f}  ratio {c/pred:6.2f}')

n = 1 << 22
d_rand = []; d_adm = []
for t in range(8):
    a = instance(rng, {6: 5}, {9: M, 10: M}, {0, 1, 2, 3, 4})
    # check the residual function agrees with the full recomputation at a point
    A4 = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(U32)
    r = r0_vec(a, A4)
    b = dict(a); b[4] = int(A4[0]); assert int(r[0]) == residuals(b)[0]
    A4p = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(U32)
    d_rand.append((r - r0_vec(a, A4p)) & MISS)
    # admissible set: a4 = a3 where a2 != a3, free where a2 == a3
    free = u32(~(a[2] ^ a[3]) & M)
    A4a = ((u32(a[3]) & ~free) | (A4p & free)) & MISS
    A4b = ((u32(a[3]) & ~free) | (A4 & free)) & MISS
    assert bool((Maj(A4a, u32(a[3]), u32(a[2])) == u32(a[3])).all())
    ne = A4a != A4b
    d_adm.append(((r0_vec(a, A4a) - r0_vec(a, A4b)) & MISS)[ne])
    print(f'   instance {t}: free bits {bin(int(free)).count("1")}, identical pairs excluded: {int((~ne).sum())}')
print('Frame B, shifted family: C0 residual r0(a4) at fixed (a0..a3, context)')
ladder(np.concatenate(d_rand), 'random a4 pairs: P(r0(a4) == r0(a4\') on low k bits)')
ladder(np.concatenate(d_adm), 'a4 pairs both in the admissible set (~2^16 of them)')

# heavy kernel with c8 = 0
X = rng.integers(0, 1 << 32, 1 << 24, dtype=np.uint64).astype(U32)
psi = (S0(X) - S1(X)) & MISS
u = np.unique(psi).size
print(f'\npsi(a4) = S0(a4) - S1(a4) on 2^24 random a4: {u:,} distinct of {X.size:,} '
      f'(random map on 2^32 would give {X.size - X.size**2/2/2**32:,.0f}); '
      f'xor-form S0^S1 rank = {np.linalg.matrix_rank(np.array([[((S0(1<<i)^S1(1<<i))>>j)&1 for i in range(32)] for j in range(32)]))}')
Y = rng.integers(0, 1 << 32, 1 << 24, dtype=np.uint64).astype(U32)
ladder(((S0(X) - S1(X)) - (S0(Y) - S1(Y))) & MISS, 'psi(a4) - psi(a4\') for random pairs')
