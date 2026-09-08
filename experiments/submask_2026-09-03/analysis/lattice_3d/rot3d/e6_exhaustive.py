#!/usr/bin/env python3
"""E6: (i) EXHAUSTIVE count over all 2^32 choices of W0 of the round-0 rotational
condition for the real SHA-256 (real IV, real K).  (ii) reduced-width validation
of the two laws the estimates rest on."""
import math
import numpy as np

M32 = np.uint32(0xFFFFFFFF)
Mi = 0xFFFFFFFF
IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
K0 = 0x428a2f98


def rotr_i(x, n, w=32): n %= w; return ((x >> n) | (x << (w - n))) & ((1 << w) - 1)
def rotl_i(x, n, w=32): return rotr_i(x, (w - n % w) % w, w)


def rotl_a(x, n):
    n = int(n) % 32
    return x if n == 0 else ((x << np.uint32(n)) | (x >> np.uint32(32 - n))).astype(np.uint32)


def S1i(x): return rotr_i(x, 6) ^ rotr_i(x, 11) ^ rotr_i(x, 25)
def S0i(x): return rotr_i(x, 2) ^ rotr_i(x, 13) ^ rotr_i(x, 22)
def Chi(e, f, g): return ((e & f) ^ (~e & g)) & Mi
def Maji(a, b, c): return (a & b) ^ (a & c) ^ (b & c)


# round 0 of the real function: T1 = C + W0, a0 = T1 + D, e0 = IV[3] + T1
C = (IV[7] + S1i(IV[4]) + Chi(IV[4], IV[5], IV[6]) + K0) & Mi
D = (S0i(IV[0]) + Maji(IV[0], IV[1], IV[2])) & Mi
print("=== (i) exhaustive over all 2^32 W0: round-0 rotational pairs, real IV/K ===")
print(f"  C = {C:08x}   D = {D:08x}   IV[3] = {IV[3]:08x}")
CH = 1 << 24
for g in (1, 2, 3, 8, 16, 24, 31):
    tot_T1 = tot_all = 0
    for base in range(0, 1 << 32, CH):
        x = (np.arange(base, base + CH, dtype=np.uint64) & 0xFFFFFFFF).astype(np.uint32)
        T1a = (x + np.uint32(C)) & M32
        T1b = (rotl_a(x, g) + np.uint32(C)) & M32
        okT1 = T1b == rotl_a(T1a, g)
        tot_T1 += int(okT1.sum())
        a0a = (T1a + np.uint32(D)) & M32
        a0b = (T1b + np.uint32(D)) & M32
        e0a = (np.uint32(IV[3]) + T1a) & M32
        e0b = (np.uint32(IV[3]) + T1b) & M32
        tot_all += int((okT1 & (a0b == rotl_a(a0a, g)) & (e0b == rotl_a(e0a, g))).sum())
    print(f"  g={g:2d}: W0 with T1 rotational: {tot_T1:,} of 2^32; "
          f"with a0 and e0 rotational too: {tot_all:,}")

print()
print("=== (ii) reduced-width validation ===")


def add_rot_prob_exhaustive(w, g):
    mw = (1 << w) - 1
    x = np.arange(1 << w, dtype=np.uint64)
    X, Y = np.meshgrid(x, x, indexing="ij")
    X = X.ravel(); Y = Y.ravel()

    def rl(v, n):
        n %= w
        return ((v << np.uint64(n)) | (v >> np.uint64(w - n))) & np.uint64(mw)
    lhs = rl((X + Y) & np.uint64(mw), g)
    rhs = (rl(X, g) + rl(Y, g)) & np.uint64(mw)
    return float((lhs == rhs).mean())


print("  Daum law  P[(x+y)<<<g == (x<<<g)+(y<<<g)] = (1/4)(1+2^(g-w)+2^-g+2^-w)")
for w in (8, 12):
    for g in (1, 2, 3, w // 2, w - 1):
        obs = add_rot_prob_exhaustive(w, g)
        pred = 0.25 * (1 + 2.0 ** (g - w) + 2.0 ** (-g) + 2.0 ** (-w))
        flag = "OK" if abs(obs - pred) < 1e-12 else "MISMATCH"
        print(f"    w={w:2d} g={g:2d}: exhaustive {obs:.10f}  predicted {pred:.10f}  {flag}")

print()
print("  SHR defect law  #differing positions of (x>>s)<<<g vs (x<<<g)>>s  ==  2*min(g,s,w-g)")
bad = 0
for w in (8, 12, 16, 32):
    mw = (1 << w) - 1
    for s in (1, 2, 3, 5, 10):
        if s >= w:
            continue
        for g in range(1, w):
            mask = 0
            # exact: the union over x of the difference support
            for b in range(w):
                x = 1 << b
                a1 = rotl_i(x >> s, g, w)
                a2 = rotl_i(x, g, w) >> s
                mask |= (a1 ^ a2)
            n = bin(mask).count("1")
            pred = 2 * min(g, s, w - g)
            if n != pred:
                bad += 1
                if bad < 8:
                    print(f"    w={w} s={s} g={g}: observed {n} predicted {pred}")
print(f"    mismatches over all (w,s,g) tested: {bad}")

print()
print("  reduced-width schedule step, EXHAUSTIVE at w=6 over all 2^24 inputs")
w = 6
mw = (1 << w) - 1
# scaled rotation amounts: sigma0 (7,18,3)->(1,3,1), sigma1 (17,19,10)->(3,4,2)


def rl6(v, n):
    n %= w
    return ((v << np.uint64(n)) | (v >> np.uint64(w - n))) & np.uint64(mw)


def rr6(v, n): return rl6(v, (w - n % w) % w)
def s0_6(v): return rr6(v, 1) ^ rr6(v, 3) ^ (v >> np.uint64(1))
def s1_6(v): return rr6(v, 3) ^ rr6(v, 4) ^ (v >> np.uint64(2))


idx = np.arange(1 << (4 * w), dtype=np.uint64)
A = (idx >> np.uint64(3 * w)) & np.uint64(mw)   # W14
B = (idx >> np.uint64(2 * w)) & np.uint64(mw)   # W9
Cc = (idx >> np.uint64(w)) & np.uint64(mw)      # W1
Dd = idx & np.uint64(mw)                        # W0
print("    g   P[step rotational]  -log2   factorised prediction")
for g in range(1, w):
    Wa = (s1_6(A) + B + s0_6(Cc) + Dd) & np.uint64(mw)
    Wb = (s1_6(rl6(A, g)) + rl6(B, g) + s0_6(rl6(Cc, g)) + rl6(Dd, g)) & np.uint64(mw)
    p = float((Wb == rl6(Wa, g)).mean())
    # factorised model
    ps0 = 2.0 ** (-2 * min(g, 1, w - g))
    ps1 = 2.0 ** (-2 * min(g, 2, w - g))
    # 4-term addition rotational probability at width w, measured
    r = np.random.default_rng(5)
    n = 1 << 22
    q = [r.integers(0, 1 << w, n, dtype=np.uint64) for _ in range(4)]
    lhs = rl6(sum(q) & np.uint64(mw), g)
    rhs = (rl6(q[0], g) + rl6(q[1], g) + rl6(q[2], g) + rl6(q[3], g)) & np.uint64(mw)
    padd = float((lhs == rhs).mean())
    print(f"    {g}   {p:.8f}        {-math.log2(p) if p else float('inf'):6.3f}   "
          f"sig0 {ps0:.4f} * sig1 {ps1:.4f} * add4 {padd:.4f} = {ps0*ps1*padd:.8f}")
