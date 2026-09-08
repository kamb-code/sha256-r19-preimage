#!/usr/bin/env python3
"""E1: which SHA-256 components commute with a cyclic rotation of the bit axis.

For each component f and each rotation amount g in 1..31, estimate
    P[ f(x <<< g, ...) == f(x,...) <<< g ]
over random inputs, at full 32-bit width.  Exact for the linear/bitwise ones.
"""
import numpy as np

N = 1 << 22
rng = np.random.default_rng(20260908)
M = np.uint32(0xFFFFFFFF)


def rotr(x, n):
    n = int(n) % 32
    if n == 0:
        return x
    return ((x >> np.uint32(n)) | (x << np.uint32(32 - n))).astype(np.uint32)


def rotl(x, n):
    return rotr(x, (32 - (int(n) % 32)) % 32)


def S0(x): return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
def S1(x): return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
def s0(x): return rotr(x, 7) ^ rotr(x, 18) ^ (x >> np.uint32(3))
def s1(x): return rotr(x, 17) ^ rotr(x, 19) ^ (x >> np.uint32(10))
def Ch(e, f, g): return (e & f) ^ (~e & g)
def Maj(a, b, c): return (a & b) ^ (a & c) ^ (b & c)


X = rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32)
Y = rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32)
Z = rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32)

print("component      " + "".join(f"g={g:<7}" for g in (1, 2, 3, 8, 16, 31)))
rows = {}
for name, fn in [
    ("Sigma0", lambda g: (S0(rotl(X, g)) == rotl(S0(X), g))),
    ("Sigma1", lambda g: (S1(rotl(X, g)) == rotl(S1(X), g))),
    ("sigma0", lambda g: (s0(rotl(X, g)) == rotl(s0(X), g))),
    ("sigma1", lambda g: (s1(rotl(X, g)) == rotl(s1(X), g))),
    ("Ch", lambda g: (Ch(rotl(X, g), rotl(Y, g), rotl(Z, g)) == rotl(Ch(X, Y, Z), g))),
    ("Maj", lambda g: (Maj(rotl(X, g), rotl(Y, g), rotl(Z, g)) == rotl(Maj(X, Y, Z), g))),
    ("XOR", lambda g: ((rotl(X, g) ^ rotl(Y, g)) == rotl(X ^ Y, g))),
    ("ADD mod 2^32", lambda g: (((rotl(X, g) + rotl(Y, g)) & M) == rotl((X + Y) & M, g))),
    ("ADD 3-term", lambda g: (((rotl(X, g) + rotl(Y, g) + rotl(Z, g)) & M)
                              == rotl((X + Y + Z) & M, g))),
]:
    vals = []
    for g in range(1, 32):
        p = float(fn(g).mean())
        vals.append(p)
    rows[name] = vals
    sel = [vals[g - 1] for g in (1, 2, 3, 8, 16, 31)]
    print(f"{name:<14}" + "".join(f"{p:<9.5f}" for p in sel))

print()
print("Daum/KN prediction for ADD, p(g) = (1/4)(1 + 2^(g-32) + 2^(-g) + 2^(-32)):")
for g in (1, 2, 3, 8, 16, 31):
    pred = 0.25 * (1 + 2.0 ** (g - 32) + 2.0 ** (-g) + 2.0 ** (-32))
    print(f"  g={g:2d}  predicted {pred:.6f}   measured {rows['ADD mod 2^32'][g-1]:.6f}")

# exact defect structure of sigma0 / sigma1 under rotation
print()
print("Exact defect of the SHR terms:  (x>>s) <<< g   vs   (x <<< g) >> s")
for name, s in (("sigma0 SHR3", 3), ("sigma1 SHR10", 10)):
    for g in (1, 2, 8, 16, 31):
        a = rotl((X >> np.uint32(s)), g)
        b = (rotl(X, g) >> np.uint32(s))
        diff = a ^ b
        mask = np.bitwise_or.reduce(diff)
        # how many bit positions can ever differ, and empirical equality rate
        nbits = bin(int(mask)).count("1")
        print(f"  {name} g={g:2d}: differing positions {nbits:2d} "
              f"(mask {int(mask):08x}), P[equal] = {float((a==b).mean()):.6f} "
              f"(2^-{nbits} = {2.0**-nbits:.3e})")
