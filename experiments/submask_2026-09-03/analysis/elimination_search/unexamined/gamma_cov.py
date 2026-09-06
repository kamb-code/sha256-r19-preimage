#!/usr/bin/env python3
"""Coverage of the a0 -> C0-target map Gamma'(a0) = a0 + C0c + G(a0) over all 2^32.

The 'structured a0 subset' idea: pick a0 so that the C0 target KC0 - Gamma'(a0)
lies in the image of sigma0(u)-u, raising the first-lookup hit rate from 0.634
to 1.  Doing so needs Gamma'^{-1}, a global relation with THIS coverage; each
a0 reached costs one Gamma'-lookup, so the hit rate of the mirror frame equals
this coverage, and the swap is a wash.  Nothing downstream (C3) changes.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import M, K, IV, S0, S1, Ch, Maj, T2, u32, U32, MISS

am1, am2, am3, am4 = (u32(x) for x in IV[:4])
em1, em2, em3, em4 = (u32(x) for x in IV[4:])
T2iv = T2(IV[0], IV[1], IV[2])
C0c = u32(-T2iv - IV[7] - S1(IV[4]) - Ch(IV[4], IV[5], IV[6]) - K[0])
Ce0 = u32(IV[3] - T2iv)
K1 = u32(K[1])

bitmap = np.zeros(1 << 32, dtype=np.uint8)
t0 = time.time()
B = 1 << 24
for start in range(0, 1 << 32, B):
    A0 = np.arange(start, start + B, dtype=np.uint64).astype(U32)
    E0 = (A0 + Ce0) & MISS
    W0 = (A0 + C0c) & MISS
    G = (-(S0(A0) + Maj(A0, am1, am2)) - em3 - S1(E0) - Ch(E0, em1, em2) - K1) & MISS
    y = (W0 + G) & MISS                     # Gamma'(a0); target = KC0 - y
    bitmap[y.astype(np.int64)] = 1
cov = int(bitmap.sum(dtype=np.int64))
print(f"Gamma' image size {cov:,} of 2^32 = {cov / 2**32:.6f}  (1-1/e = {1-np.exp(-1):.6f}; "
      f"sigma0(u)-u image fraction 0.633673)   [{time.time()-t0:.0f}s]")
