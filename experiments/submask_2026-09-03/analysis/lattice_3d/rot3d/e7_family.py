#!/usr/bin/env python3
"""E7: does any symmetry act on the SUBMASK FAMILY itself?

The family conditions are unusually symmetry-friendly:
  a4 = a5 = v            -- covariant under any bijection applied word-wise
  e8 = e9 = 0xFFFFFFFF   -- 0xFFFFFFFF is invariant under every bit rotation
  Maj(v,a3,a2) = a3      -- purely bitwise, invariant under EVERY bit permutation
So if any part of this attack were going to be rotation-covariant, this is it.
We test where it actually breaks.

Also: slide / translation along the round axis.
"""
import numpy as np

rng = np.random.default_rng(31337)
M = np.uint32(0xFFFFFFFF)
Mi = 0xFFFFFFFF
K = [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
     0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
     0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
     0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
     0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
     0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
     0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
     0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
     0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
     0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
     0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]


def rotl(x, n):
    n = int(n) % 32
    return x if n == 0 else ((x << np.uint32(n)) | (x >> np.uint32(32 - n))).astype(np.uint32)


def S0(x): return ((x >> np.uint32(2)) | (x << np.uint32(30))) ^ \
    ((x >> np.uint32(13)) | (x << np.uint32(19))) ^ ((x >> np.uint32(22)) | (x << np.uint32(10)))


def S1(x): return ((x >> np.uint32(6)) | (x << np.uint32(26))) ^ \
    ((x >> np.uint32(11)) | (x << np.uint32(21))) ^ ((x >> np.uint32(25)) | (x << np.uint32(7)))


def s0f(x): return ((x >> np.uint32(7)) | (x << np.uint32(25))) ^ \
    ((x >> np.uint32(18)) | (x << np.uint32(14))) ^ (x >> np.uint32(3))


def Maj(a, b, c): return (a & b) ^ (a & c) ^ (b & c)


N = 1 << 22
print("=== (1) is the family closed under a bit rotation of the state? ===")
print("build a family context (a4=a5=v, e8=e9=-1), rotate all its words by g,")
print("and ask whether the rotated words still satisfy the family conditions.")
v = rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32)
a6 = rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32)
a7 = rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32)
a4 = a5 = v
a8 = (M - a4 + S0(a7) + Maj(a7, a6, a5)) & M
a9 = (M - a5 + S0(a8) + Maj(a8, a7, a6)) & M
e8 = (a4 + a8 - (S0(a7) + Maj(a7, a6, a5))) & M
e9 = (a5 + a9 - (S0(a8) + Maj(a8, a7, a6))) & M
assert np.all(e8 == M) and np.all(e9 == M)
for g in (1, 2, 8, 16, 31):
    r4, r5, r6, r7, r8, r9 = (rotl(x, g) for x in (a4, a5, a6, a7, a8, a9))
    E8 = (r4 + r8 - (S0(r7) + Maj(r7, r6, r5))) & M
    E9 = (r5 + r9 - (S0(r8) + Maj(r8, r7, r6))) & M
    p8 = float((E8 == M).mean()); p9 = float((E9 == M).mean())
    pb = float(((E8 == M) & (E9 == M)).mean())
    print(f"  g={g:2d}: rotated context has e8=-1 in {p8:.3e}, e9=-1 in {p9:.3e}, "
          f"both {pb:.3e}   (a4=a5 always survives)")

print()
print("=== (2) which pieces ARE rotation-covariant ===")
a2 = rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32)
a3 = rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32)
coll = Maj(v, a3, a2) == a3
for g in (1, 5, 16):
    coll_r = Maj(rotl(v, g), rotl(a3, g), rotl(a2, g)) == rotl(a3, g)
    print(f"  g={g:2d}: collapsed condition Maj(v,a3,a2)==a3 covariant on "
          f"{float((coll == coll_r).mean()):.6f} of samples "
          f"(bitwise -> exactly 1.0 expected)")
print(f"  0xFFFFFFFF is rotation-invariant: True; so e8=e9=-1 is a symmetric state")
print(f"  a4 = a5 is an equality between words: covariant under any word map: True")

print()
print("=== (3) the fourth constraint c3 = sigma0(W4) + W3 - kappa3 ===")
W3 = rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32)
W4 = rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32)
for g in (1, 2, 16, 31):
    lhs = rotl((s0f(W4) + W3) & M, g)
    rhs = (s0f(rotl(W4, g)) + rotl(W3, g)) & M
    print(f"  g={g:2d}: P[ rot(sigma0(W4)+W3) == sigma0(rot W4)+rot W3 ] = "
          f"{float((lhs == rhs).mean()):.6e}")
print("  (kappa3 is a fixed constant, so it contributes a further deterministic")
print("   mismatch rot(kappa3) != kappa3 unless kappa3 is rotation-invariant.)")

print()
print("=== (4) slide / translation along the round axis ===")
eq = [(d, sum(1 for r in range(64 - d) if K[r] == K[r + d])) for d in range(1, 21)]
print(f"  pairs with K_r == K_(r+d), d=1..20: {[c for _, c in eq]}  (all zero -> "
      f"the round function is never self-similar under a shift)")
print(f"  distinct K_r among the 20 used at R=20: {len(set(K[:20]))} of 20")
print("  message EXPANSION is a constant-coefficient recurrence, hence exactly")
print("  translation-covariant: W'_i = W_(i+d) is again a valid schedule.")
print("  So the only obstructions to a slide are K_r and the IV alignment.")
IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
print("  IV alignment for a slide by d=1 (state after round 1 == IV) forces")
print(f"    IV0==IV1 {IV[0]==IV[1]}, IV1==IV2 {IV[1]==IV[2]}, IV2==IV3 {IV[2]==IV[3]},")
print(f"    IV4==IV5 {IV[4]==IV[5]}, IV5==IV6 {IV[5]==IV[6]}, IV6==IV7 {IV[6]==IV[7]}")
print("    -> 6 word equalities required, 0 hold: a d=1 slide is IMPOSSIBLE, not rare.")
print("  for d>=4 all eight state words are message-dependent: 8 x 32 = 256 bit")
print("  conditions, i.e. finding a slid pair is itself a 2^256 problem, and even")
print("  then it relates preimages of two DIFFERENT digests.")
