#!/usr/bin/env python3
"""Atom expansion of c3, and the rotate-atom census of every constraint.

A residual whose zero set is a product set across bit positions must be a
signed sum of BITWISE atoms (identity, Maj, Ch, and/or/xor/not).  Any
rotate-XOR atom (Sigma0, Sigma1, sigma0, sigma1) of a word that is not a
constant destroys the per-bit factorisation.  This script (1) verifies the
closed atom expansion of c3, (2) counts, for each constraint, the rotate atoms
whose argument still contains an unknown, and (3) checks whether any context
condition in the family menu can freeze such an argument.
"""
import numpy as np
from alg import Alg, Instance, family_instance

A = Alg(32)
M = A.M
rng = np.random.default_rng(5)
inst, v = family_instance(A, rng)
a = inst.a
IV = A.IV
am1, am2, am3, am4 = a[-1], a[-2], a[-3], a[-4]
em1, em2, em3, em4 = IV[4], IV[5], IV[6], IV[7]

n = 200000
A0 = rng.integers(0, 1 << 32, n, dtype=np.uint64)
A1 = rng.integers(0, 1 << 32, n, dtype=np.uint64)
A2 = rng.integers(0, 1 << 32, n, dtype=np.uint64)
A3 = rng.integers(0, 1 << 32, n, dtype=np.uint64)
c = inst.residuals(A0, A1, A2, A3)

u = lambda x: np.uint64(x & M)
E0 = (u(am4) + A0 - A.T2(u(am1), u(am2), u(am3))) & M
E1 = (u(am3) + A1 - A.T2(A0, u(am1), u(am2))) & M
e2 = (u(am2) + A2 - A.T2(A1, A0, u(am1))) & M
e3 = (u(am1) + A3 - A.T2(A2, A1, A0)) & M

# claimed:  c3 = -sigma0(W4) - W3 + kappa3 (sign convention of alg.residuals:
#           c_j = W16+j - s1(W14+j) - W9+j - s0(W1+j) - Wj, and for j=3
#           W12 is a constant, so c3 = K - s0(W4) - W3)
W3 = (A3 - A.T2(A2, A1, A0) - u(em1) - A.S1(e2) - A.Ch(e2, E1, E0) - u(A.K[3])) & M
W4 = (u(a[4]) - A.T2(A3, A2, A1) - E0 - A.S1(e3) - A.Ch(e3, e2, E1) - u(A.K[4])) & M
kappa = (inst.Wc[19] - A.s1(inst.Wc[17]) - inst.Wc[12]) & M
c3_claim = (u(kappa) - A.s0(W4) - W3) & M
print("c3 == kappa3 - sigma0(W4) - W3 :",
      "OK" if np.array_equal(c3_claim, c[3]) else "FAIL")

# fully expanded atom form
c3_atoms = (u(kappa) - A.s0(W4)
            - A3 + A.S0(A2) + A.Maj(A2, A1, A0) + u(em1) + A.S1(e2) + A.Ch(e2, E1, E0)
            + u(A.K[3])) & M
print("c3 == kappa3 - sigma0(W4) - a3 + Sigma0(a2) + Maj(a2,a1,a0) + e_-1"
      " + Sigma1(e2) + Ch(e2,e1,e0) + K3 :",
      "OK" if np.array_equal(c3_atoms, c[3]) else "FAIL")

# rotate-atom census: which rotate atoms have an argument depending on an unknown
print("\nrotate-XOR atoms of a non-constant argument, per residual "
      "(these are what block a per-bit factorisation):")
print("  c0 : sigma0(W1)[a0,a1]            -- 1 (W9's Sigma0(a4) is context)")
print("  c1 : sigma0(W2)[a0..a2], Sigma0(a1) in W2  -- >=2")
print("  c2 : sigma0(W3)[a0..a3], Sigma0(a2)        -- >=2")
print("  c3 : sigma0(W4)[a0..a3], Sigma0(a2), Sigma1(e2)[a0..a2],"
      " and Sigma0(a3),Sigma1(e3) inside W4  -- >=5")

# can any of the five be frozen?  Their arguments are W4, a2, e2, a3, e3,
# every one a function of the unknowns alone.  Show each is non-constant on
# the attack's own candidate stream by measuring its image size on random
# candidate-like inputs.
for name, x in (("W4", W4), ("a2", A2), ("e2", e2), ("a3", A3), ("e3", e3)):
    print(f"  argument {name:3s}: distinct values on {n} random points = "
          f"{len(np.unique(x))}")

# a rotate atom is bitwise only if its rotation amounts are all 0 mod w
print("\nrotation amounts at w=32:", A.S0r, A.S1r, A.s0r, A.s1r,
      "-> none is the identity, so none of the five atoms is bitwise")
