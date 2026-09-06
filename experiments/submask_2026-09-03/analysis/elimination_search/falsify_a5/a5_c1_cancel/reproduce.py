#!/usr/bin/env python3
"""Self-contained reproduction of the a5 -> C1 edge weight for one context
condition (no dependence on the other scripts in this directory; only
code/submask_family.py for the SHA-256 primitives).

Frame: R = 20; a0..a3 and a5 unknown, a4 = v and a6..a11 context, a12..a19 digest.
Context here (the lightest legal condition found in this study):
    a6 = a4,  a7 = a6,  e8 = -1 (via a8),  e9 - a5 = C9 (via a9 = C9 + T2(a8,a7,a6)),  e11 = 0 (via a11)
Nothing references a5.  For 200 random states flip each bit of a5 and count the
bits that change in T_j = W_{16+j} - s1(W_{14+j}) - W_{9+j} - W_j, j = 0..3
(the C_j lookup index up to a5-free terms); same for a4 with the context held
fixed, as the control.

    python3 reproduce.py [C9 hex, default 0] [n states, default 200] [seed]
"""
import sys
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import K, IV, S1, s1, Ch, T2

U = np.uint32
C9 = int(sys.argv[1], 16) if len(sys.argv) > 1 else 0
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1
np.seterr(over='ignore')
rng = np.random.default_rng(SEED)
rw = lambda: rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(U)


def popcount(x):
    return np.array([bin(int(v)).count('1') for v in x])


def context(a4, a10_free, a6_free_unused=None):
    a = {4: a4}
    a[6] = a[4]                                            # a6 = a4
    a[7] = a[6]                                            # a7 = a6  (Maj(a7,a6,a5) = a6: e8 is a5-free)
    a[8] = U(0xFFFFFFFF) - a[4] + T2(a[7], a[6], U(0))     # e8 = a4 + a8 - S0(a7) - Maj(a7,a6,.) = -1
    a[9] = U(C9) + T2(a[8], a[7], a[6])                    # e9 = a5 + a9 - T2(a8,a7,a6) = a5 + C9
    a[10] = a10_free
    return a


def finish(a):
    # e11 = a7 + a11 - T2(a10,a9,a8) = 0  via a11
    a[11] = U(0) - a[7] + T2(a[10], a[9], a[8])
    return a


def targets(a):
    e = {-1: U(IV[4]), -2: U(IV[5]), -3: U(IV[6]), -4: U(IV[7])}
    for r in range(20):
        e[r] = a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])
    W = {r: (a[r] - T2(a[r - 1], a[r - 2], a[r - 3]) - e[r - 4] - S1(e[r - 1])
             - Ch(e[r - 1], e[r - 2], e[r - 3]) - U(K[r])) for r in range(20)}
    return [W[16 + j] - s1(W[14 + j]) - W[9 + j] - W[j] for j in range(4)]


a = {-1: U(IV[0]), -2: U(IV[1]), -3: U(IV[2]), -4: U(IV[3])}
for i in range(4):
    a[i] = rw()
a[5] = rw()
a.update(finish(context(rw(), rw())))
for i in range(12, 20):
    a[i] = rw()
# sanity on the conditions
e8 = a[4] + a[8] - T2(a[7], a[6], a[5]); e9 = a[5] + a[9] - T2(a[8], a[7], a[6]); e11 = a[7] + a[11] - T2(a[10], a[9], a[8])
assert (e8 == U(0xFFFFFFFF)).all() and (e9 == a[5] + U(C9)).all() and (e11 == 0).all()

T0 = targets(a)
for word in (5, 4):
    acc = np.zeros(4); per = np.zeros((4, 32))
    for i in range(32):
        b = dict(a); b[word] = a[word] ^ U(1 << i)
        T1 = targets(b)
        for j in range(4):
            per[j, i] = popcount(T0[j] ^ T1[j]).mean()
    print(f"a{word} -> " + "  ".join(f"C{j}: {per[j].mean():5.2f} (bits {per[j].min():.1f}..{per[j].max():.1f})" for j in range(4))
          + ("" if word == 5 else "   [control, context held fixed]"))
print(f"C9 = 0x{C9:08x}, {N} states, seed {SEED}.  A weight below 4 in the a5 -> C1 column would be WEAK.")
