#!/usr/bin/env python3
"""E3: candidate symmetries of the 3D state (round r) x (word) x (bit).

(1) rotation of the bit axis through the full compression, constants on/off
(2) complementation, component by component and through the rounds
(3) symmetric states: words with a nontrivial bit-period d | 32
(4) constants: are any K_r or IV_i rotation-invariant / period-d / self-complementary
"""
import numpy as np

rng = np.random.default_rng(4242)
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
IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]


def rotr(x, n):
    n = int(n) % 32
    return x if n == 0 else ((x >> np.uint32(n)) | (x << np.uint32(32 - n))).astype(np.uint32)


def rotl(x, n): return rotr(x, (32 - (int(n) % 32)) % 32)
def S0(x): return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
def S1(x): return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
def s0(x): return rotr(x, 7) ^ rotr(x, 18) ^ (x >> np.uint32(3))
def s1(x): return rotr(x, 17) ^ rotr(x, 19) ^ (x >> np.uint32(10))
def Ch(e, f, g): return (e & f) ^ (~e & g)
def Maj(a, b, c): return (a & b) ^ (a & c) ^ (b & c)


def rounds(W16, R, iv, use_K=True):
    """W16: list of 16 uint32 arrays. Returns dicts a, e over r = -4..R-1."""
    W = list(W16)
    for t in range(16, R):
        W.append((s1(W[t - 2]) + W[t - 7] + s0(W[t - 15]) + W[t - 16]) & M)
    a = {-1: iv[0], -2: iv[1], -3: iv[2], -4: iv[3]}
    e = {-1: iv[4], -2: iv[5], -3: iv[6], -4: iv[7]}
    for r in range(R):
        kk = np.uint32(K[r]) if use_K else np.uint32(0)
        T1 = (e[r - 4] + S1(e[r - 1]) + Ch(e[r - 1], e[r - 2], e[r - 3]) + kk + W[r]) & M
        a[r] = (T1 + S0(a[r - 1]) + Maj(a[r - 1], a[r - 2], a[r - 3])) & M
        e[r] = (a[r - 4] + T1) & M
    return a, e, W


N = 1 << 22
R = 20

print("=== (1) rotation of the bit axis, full compression, R = 20 ===")
print("A rotational pair needs W' = W<<<g AND a rotated IV (IV<<<g) AND, with")
print("constants on, K_r<<<g == K_r.  Per-round survival, measured:")
for use_K, tag in ((False, "K removed, IV rotated"), (True, "real K, IV rotated")):
    for g in (1, 31):
        W = [rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32) for _ in range(16)]
        Wr = [rotl(x, g) for x in W]
        ivA = [np.uint32(v) for v in IV]
        ivB = [rotl(np.uint32(v), g) for v in IV]
        aA, eA, WA = rounds(W, R, ivA, use_K)
        aB, eB, WB = rounds(Wr, R, ivB, use_K)
        alive = np.ones(N, dtype=bool)
        surv = []
        for r in range(R):
            ok = (aB[r] == rotl(aA[r], g)) & (eB[r] == rotl(eA[r], g))
            alive = alive & ok
            surv.append(int(alive.sum()))
            if alive.sum() == 0:
                break
        print(f"  {tag}, g={g}: alive after rounds 0.. = {surv[:8]} "
              f"(N={N:,}) -> dies at round {len(surv)-1 if surv[-1]==0 else '>'+str(len(surv)-1)}")

print()
print("=== (2) complementation x -> ~x ===")
X = rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32)
Y = rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32)
Z = rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32)
print(f"  Sigma0(~x) == ~Sigma0(x)        : {float((S0(~X)==(~S0(X))).mean()):.6f}")
print(f"  Sigma1(~x) == ~Sigma1(x)        : {float((S1(~X)==(~S1(X))).mean()):.6f}")
d0 = np.bitwise_or.reduce(s0(~X) ^ (~s0(X)))
d1 = np.bitwise_or.reduce(s1(~X) ^ (~s1(X)))
c0 = np.unique(s0(~X) ^ (~s0(X)))
c1 = np.unique(s1(~X) ^ (~s1(X)))
print(f"  sigma0(~x) ^ ~sigma0(x)         : constant? {len(c0)==1} value {int(c0[0]):08x}")
print(f"  sigma1(~x) ^ ~sigma1(x)         : constant? {len(c1)==1} value {int(c1[0]):08x}")
print(f"  Maj(~a,~b,~c) == ~Maj(a,b,c)    : {float((Maj(~X,~Y,~Z)==(~Maj(X,Y,Z))).mean()):.6f}")
print(f"  Ch(~e,~f,~g) == ~Ch(e,f,g)      : {float((Ch(~X,~Y,~Z)==(~Ch(X,Y,Z))).mean()):.6e}")
print(f"  Ch(~e,f,g)   == Ch(e,g,f)       : {float((Ch(~X,Y,Z)==Ch(X,Z,Y)).mean()):.6f}")
add_ok = (((~X).astype(np.uint32) + (~Y).astype(np.uint32)) & M) == ((~((X + Y) & M)).astype(np.uint32) - np.uint32(1)) & M
print(f"  ~x + ~y == ~(x+y) - 1  (exact)  : {float(add_ok.mean()):.6f}")
# per-bit rate for Ch complementation
bits = 0
tot = 0
for e in (0, 1):
    for f in (0, 1):
        for gg in (0, 1):
            lhs = ((1 - e) & (1 - f)) ^ (e & (1 - gg))
            rhs = 1 - ((e & f) ^ ((1 - e) & gg))
            tot += 1
            bits += (lhs == rhs)
print(f"  per-bit rate of Ch complementation: {bits}/{tot}  -> word rate 2^-32 = {2.0**-32:.3e}")

print()
print("=== (3) symmetric states: words of bit-period d | 32 ===")
print("The period-d subspace is closed under Sigma0/Sigma1 (rotations), Ch, Maj;")
print("sigma0/sigma1 leave it (shifts); addition leaves it unless carries agree.")
for d in (1, 2, 4, 8, 16):
    n = 1 << 20
    # random period-d words
    blk = rng.integers(0, 1 << d, n, dtype=np.uint64)
    x = np.zeros(n, dtype=np.uint64)
    for i in range(32 // d):
        x = (x << np.uint64(d)) | blk
    x = x.astype(np.uint32)
    blk2 = rng.integers(0, 1 << d, n, dtype=np.uint64)
    y = np.zeros(n, dtype=np.uint64)
    for i in range(32 // d):
        y = (y << np.uint64(d)) | blk2
    y = y.astype(np.uint32)

    def is_period(v, d):
        return rotl(v, d) == v
    pa = float(is_period(((x + y) & M), d).mean())
    ps0 = float(is_period(s0(x), d).mean())
    ps1 = float(is_period(s1(x), d).mean())
    pS0 = float(is_period(S0(x), d).mean())
    pCh = float(is_period(Ch(x, y, x ^ y), d).mean())
    print(f"  d={d:2d}: P[x+y period-d]={pa:.4f}  P[sigma0(x) period-d]={ps0:.4f}  "
          f"P[sigma1(x) period-d]={ps1:.4f}  Sigma0={pS0:.4f}  Ch={pCh:.4f}")

print()
print("=== (4) the constants ===")
rot_inv = [(r, k) for r, k in enumerate(K) for g in range(1, 32)
           if ((k >> g) | (k << (32 - g))) & Mi == k]
print(f"  K_r with any nontrivial rotational symmetry: {len(set(r for r,_ in rot_inv))} of 64")
per = []
for r, k in enumerate(K):
    for d in (1, 2, 4, 8, 16):
        if ((k >> d) | (k << (32 - d))) & Mi == k:
            per.append((r, d))
print(f"  K_r with a nontrivial bit-period d|32: {len(per)} of 64*5 tests -> {per}")
selfc = [r for r, k in enumerate(K) if (k ^ Mi) == k]
print(f"  K_r self-complementary: {len(selfc)}")
print(f"  IV_i with any nontrivial rotational symmetry: "
      f"{sum(1 for v in IV for g in range(1,32) if ((v>>g)|(v<<(32-g)))&Mi==v)}")
eqK = sum(1 for r in range(63) if K[r] == K[r + 1])
print(f"  adjacent equal round constants K_r == K_(r+1): {eqK} of 63 (slide needs all)")
print(f"  distinct K_r among the first 20: {len(set(K[:20]))}")
print()
print("  IV equalities a slide by one round would require "
      "(state after round 1 == IV):")
names = ["a_-1==a_-2", "a_-2==a_-3", "a_-3==a_-4", "e_-1==e_-2", "e_-2==e_-3", "e_-3==e_-4"]
vals = [IV[0] == IV[1], IV[1] == IV[2], IV[2] == IV[3],
        IV[4] == IV[5], IV[5] == IV[6], IV[6] == IV[7]]
print("   " + ", ".join(f"{n}:{v}" for n, v in zip(names, vals))
      + f"  -> {sum(vals)}/6 hold")
