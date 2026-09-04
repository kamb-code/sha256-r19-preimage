#!/usr/bin/env python3
"""INDEPENDENT re-derivation of the R=19 'submask family' algebra.

Nothing here imports the attack code.  SHA-256 is re-implemented straight from
FIPS 180-4 and pinned against hashlib.  Every algebraic claim is then checked as
an identity over random inputs, and the probability claim is checked by
exhaustive enumeration at reduced word width.
"""
import hashlib, itertools, random, struct

M = 0xFFFFFFFF

# ---------------------------------------------------------------- FIPS 180-4
Kc = [
 0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dbA5 & M,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
 0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
 0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
 0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
 0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
 0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
 0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
 0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]
IV = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]

rotr = lambda x, n: ((x >> n) | (x << (32 - n))) & M
S0   = lambda x: rotr(x,2) ^ rotr(x,13) ^ rotr(x,22)     # Sigma0
S1   = lambda x: rotr(x,6) ^ rotr(x,11) ^ rotr(x,25)     # Sigma1
s0   = lambda x: rotr(x,7) ^ rotr(x,18) ^ (x >> 3)       # sigma0
s1   = lambda x: rotr(x,17) ^ rotr(x,19) ^ (x >> 10)     # sigma1
Ch   = lambda x,y,z: ((x & y) ^ ((~x & M) & z)) & M
Maj  = lambda x,y,z: ((x & y) ^ (x & z) ^ (y & z)) & M
T2f  = lambda a,b,c: (S0(a) + Maj(a,b,c)) & M


def compress(W, R=64):
    """FIPS 180-4 compression, R rounds, returning the a_r / e_r indexing of the
    paper: a_{-1}=IV0..a_{-4}=IV3, e_{-1}=IV4..e_{-4}=IV7."""
    Wf = list(W)
    for t in range(16, R):
        Wf.append((s1(Wf[t-2]) + Wf[t-7] + s0(Wf[t-15]) + Wf[t-16]) & M)
    a = {-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]}
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(R):
        T1 = (e[r-4] + S1(e[r-1]) + Ch(e[r-1], e[r-2], e[r-3]) + Kc[r] + Wf[r]) & M
        a[r] = (T1 + T2f(a[r-1], a[r-2], a[r-3])) & M
        e[r] = (a[r-4] + T1) & M
    return a, e, Wf


def digest(W, R=64):
    a, e, _ = compress(W, R)
    st = [a[R-1], a[R-2], a[R-3], a[R-4], e[R-1], e[R-2], e[R-3], e[R-4]]
    return b"".join(struct.pack(">I", (x + y) & M) for x, y in zip(st, IV))


# pin against hashlib at 64 rounds
for _ in range(20):
    blk = bytes(random.getrandbits(8) for _ in range(55))
    pad = blk + b"\x80" + b"\x00" * 0 + struct.pack(">Q", 55 * 8)
    assert len(pad) == 64
    Wp = [struct.unpack(">I", pad[4*i:4*i+4])[0] for i in range(16)]
    assert digest(Wp, 64) == hashlib.sha256(blk).digest()
print("[A] my SHA-256 == hashlib at 64 rounds (20 random blocks)          OK")

rnd = lambda: random.getrandbits(32)

# ------------------------------------------------------------------ (a) W9
# Claim A1:  e_r = a_{r-4} + a_r - T2(a_{r-1},a_{r-2},a_{r-3})
# Claim A2:  W_r = a_r - T2_r - e_{r-4} - Sigma1(e_{r-1}) - Ch(e_{r-1},e_{r-2},e_{r-3}) - K_r
# Claim A3:  W_9 = W9base - Sigma1(e8) - Ch(e8,e7,e6) - T1_5 - a_1,  with
#            W9base = a_9 - T2(a_8,a_7,a_6) - K_9,   T1_5 = a_5 - Sigma0(a_4) - Maj(a_4,a_3,a_2)
for _ in range(2000):
    W = [rnd() for _ in range(16)]
    a, e, Wf = compress(W, 19)
    for r in range(19):
        assert e[r] == (a[r-4] + a[r] - T2f(a[r-1], a[r-2], a[r-3])) & M
        assert Wf[r] == (a[r] - T2f(a[r-1],a[r-2],a[r-3]) - e[r-4] - S1(e[r-1])
                         - Ch(e[r-1], e[r-2], e[r-3]) - Kc[r]) & M
    W9base = (a[9] - T2f(a[8], a[7], a[6]) - Kc[9]) & M
    T1_5   = (a[5] - S0(a[4]) - Maj(a[4], a[3], a[2])) & M
    assert Wf[9] == (W9base - S1(e[8]) - Ch(e[8], e[7], e[6]) - T1_5 - a[1]) & M
    # the a_2 / a_3 entry points, written out
    assert e[6] == (a[2] + a[6] - S0(a[5]) - Maj(a[5], a[4], a[3])) & M
    assert e[7] == (a[3] + a[7] - S0(a[6]) - Maj(a[6], a[5], a[4])) & M
    assert e[5] == (a[1] + T1_5) & M
print("[a] W9 = W9base - S1(e8) - Ch(e8,e7,e6) - T1_5 - a1   (2000 random)  OK")

# W9conv := the a1-free part of W9 as a function of (a2,a3) with a4..a9 fixed.
def W9conv(a2, a3, ctx):
    a4, a5, a6, a7, a8, a9 = (ctx[i] for i in range(4, 10))
    W9base = (a9 - T2f(a8, a7, a6) - Kc[9]) & M
    e8 = (a4 + a8 - T2f(a7, a6, a5)) & M
    e7 = (a3 + a7 - S0(a6) - Maj(a6, a5, a4)) & M
    e6 = (a2 + a6 - S0(a5) - Maj(a5, a4, a3)) & M
    T1_5 = (a5 - S0(a4) - Maj(a4, a3, a2)) & M
    return (W9base - S1(e8) - Ch(e8, e7, e6) - T1_5) & M

# cross-check W9conv against the real cipher: W9 = W9conv - a1
for _ in range(500):
    W = [rnd() for _ in range(16)]
    a, e, Wf = compress(W, 19)
    assert Wf[9] == (W9conv(a[2], a[3], a) - a[1]) & M
print("[a] W9conv(a2,a3) - a1 == true W9                     (500 random)   OK")

# general residual, no family conditions:
#   eps = Maj(a4,a3,a2) - [ Ch(e8,e7,e6) - Ch(e8,e7|a3=0, e6|a2=a3=0) ]
for _ in range(2000):
    ctx = {i: rnd() for i in range(4, 10)}
    a2, a3 = rnd(), rnd()
    a4, a5, a6, a7, a8 = (ctx[i] for i in range(4, 9))
    e8  = (a4 + a8 - T2f(a7, a6, a5)) & M
    e7  = (a3 + a7 - S0(a6) - Maj(a6, a5, a4)) & M
    e7h = (0  + a7 - S0(a6) - Maj(a6, a5, a4)) & M
    e6  = (a2 + a6 - S0(a5) - Maj(a5, a4, a3)) & M
    e6h = (0  + a6 - S0(a5) - Maj(a5, a4, 0 )) & M
    eps = (W9conv(a2, a3, ctx) - W9conv(0, 0, ctx)) & M
    assert eps == (Maj(a4, a3, a2) - (Ch(e8,e7,e6) - Ch(e8,e7h,e6h))) & M
print("[a] general eps = Maj(a4,a3,a2) - dCh                 (2000 random)  OK")

# ------------------------------------------------------- (b) eps under e8=-1
def ctx_family(v, rng):
    """a4=a5=v, a6,a7 free, a8 chosen so e8=-1, a9 chosen so e9=-1."""
    a4 = a5 = v
    a6, a7 = rng.getrandbits(32), rng.getrandbits(32)
    a8 = (M - a4 + T2f(a7, a6, a5)) & M          # e8 = a4 + a8 - T2(a7,a6,a5) = -1
    a9 = (M - a5 + T2f(a8, a7, a6)) & M          # e9 = a5 + a9 - T2(a8,a7,a6) = -1
    return {4: a4, 5: a5, 6: a6, 7: a7, 8: a8, 9: a9}

rng = random.Random(1)
bad_v = bad_a5 = 0
for _ in range(4000):
    v = rng.getrandbits(32)
    ctx = ctx_family(v, rng)
    e8 = (ctx[4] + ctx[8] - T2f(ctx[7], ctx[6], ctx[5])) & M
    e9 = (ctx[5] + ctx[9] - T2f(ctx[8], ctx[7], ctx[6])) & M
    assert e8 == M and e9 == M
    a2, a3 = rng.getrandbits(32), rng.getrandbits(32)
    eps = (W9conv(a2, a3, ctx) - W9conv(0, 0, ctx)) & M
    if eps != (Maj(v, a3, a2) - a3) & M: bad_v += 1
    # a5 free (a4=v only, e8 still forced to -1), a5 != a4
    ctx2 = dict(ctx); ctx2[5] = rng.getrandbits(32)
    ctx2[8] = (M - ctx2[4] + T2f(ctx2[7], ctx2[6], ctx2[5])) & M
    ctx2[9] = (M - ctx2[5] + T2f(ctx2[8], ctx2[7], ctx2[6])) & M
    eps2 = (W9conv(a2, a3, ctx2) - W9conv(0, 0, ctx2)) & M
    if eps2 != (Maj(ctx2[4], a3, a2) - a3) & M: bad_a5 += 1
print(f"[b] eps == Maj(a4,a3,a2) - a3 when e8=-1: failures {bad_v}/4000 "
      f"(a4=a5=v) and {bad_a5}/4000 (a5 free)                              "
      f"{'OK' if bad_v==bad_a5==0 else 'FAIL'}")

# and it is genuinely e8 that does it: break e8 only
bad = 0
for _ in range(2000):
    v = rng.getrandbits(32)
    ctx = ctx_family(v, rng)
    ctx[8] = rng.getrandbits(32)                 # e8 now generic
    a2, a3 = rng.getrandbits(32), rng.getrandbits(32)
    eps = (W9conv(a2, a3, ctx) - W9conv(0, 0, ctx)) & M
    if eps == (Maj(v, a3, a2) - a3) & M: bad += 1
print(f"[b] with e8 generic the identity holds only by accident: {bad}/2000")

# is eps == 0 exactly the bitwise condition Maj(v,a3,a2) == a3 ?
# x - y == 0 (mod 2^32) <=> x == y as 32-bit words: no borrow can rescue it.
bad = 0
for _ in range(200000):
    v, a2, a3 = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32)
    if (((Maj(v,a3,a2) - a3) & M) == 0) != (Maj(v,a3,a2) == a3): bad += 1
print(f"[b] (Maj-a3 mod 2^32 == 0) <=> (Maj == a3) bitwise: mismatches {bad}/200000")

# at v = 0 the difference is even borrow-free: Maj(0,a3,a2)-a3 = -(a3 & ~a2)
bad = 0
for _ in range(200000):
    a2, a3 = rng.getrandbits(32), rng.getrandbits(32)
    if ((Maj(0, a3, a2) - a3) & M) != ((-(a3 & ~a2 & M)) & M): bad += 1
print(f"[b] at v=0: eps = -(a3 & ~a2) exactly: mismatches {bad}/200000")

# ---------------------------------------------------- (b) exact probability
# exhaustive at reduced width n: #{(a2,a3) in [0,2^n)^2 : Maj(v,a3,a2)==a3} = 3^n
def maj_n(x, y, z): return (x & y) ^ (x & z) ^ (y & z)
print("[b] exhaustive count of Maj(v,a3,a2)==a3 at reduced width:")
for n in (1, 2, 3, 6, 8):
    lo, hi = 4**n, 0
    for v in range(2**n):
        c = sum(1 for a2 in range(2**n) for a3 in range(2**n) if maj_n(v, a3, a2) == a3)
        lo, hi = min(lo, c), max(hi, c)
    print(f"      n={n:2d}: count over all v in [{lo},{hi}], 3^n={3**n}, "
          f"p={lo/4**n:.6f}, (3/4)^n={(3/4)**n:.6f}  "
          f"{'OK' if lo==hi==3**n else 'FAIL'}")
import math
print(f"      => p = (3/4)^32 = {(3/4)**32:.6e} = 2^{math.log2((3/4)**32):.4f}  vs 2^-32 = {2**-32:.6e}  "
      f"enhancement 2^{math.log2((3/4)**32 * 2**32):.2f}")

# empirical at full width, several v
for v in (0, 1, M, 0x9e3779b9, 0xffff0000):
    n, hits = 1 << 22, 0
    for _ in range(n):
        a2, a3 = rng.getrandbits(32), rng.getrandbits(32)
        if Maj(v, a3, a2) == a3: hits += 1
    print(f"      v=0x{v:08x}: {hits}/{n} = {hits/n:.4e}   (3/4)^32={(3/4)**32:.4e}")

# ------------------------------------------------------------- (c) C1 target
# W10 = W10base - e6 - Ch(e9,e8,e7) - K10 folded, with
# W10base = a10 - T2(a9,a8,a7) - Sigma1(e9) - K10.   e6 = a2 + c6 - Maj(a5,a4,a3).
# So the C1 target for the sigma0(W2)-W2 lookup carries
#     D(a3) = c6 - Maj(a5,a4,a3) + Ch(e9,e8,e7(a3)),  c6 = a6 - Sigma0(a5).
for _ in range(500):
    W = [rnd() for _ in range(16)]
    a, e, Wf = compress(W, 19)
    W10base = (a[10] - T2f(a[9], a[8], a[7]) - S1(e[9]) - Kc[10]) & M
    assert Wf[10] == (W10base - e[6] - Ch(e[9], e[8], e[7])) & M
    c6 = (a[6] - S0(a[5])) & M
    D  = (c6 - Maj(a[5], a[4], a[3]) + Ch(e[9], e[8], e[7])) & M
    assert Wf[10] == (W10base - a[2] - D) & M
print("[c] W10 = W10base - a2 - D(a3),  D = c6 - Maj(a5,a4,a3) + Ch(e9,e8,e7)  OK")

def D_of(a3, ctx, a10):
    a4, a5, a6, a7, a8, a9 = (ctx[i] for i in range(4, 10))
    e8 = (a4 + a8 - T2f(a7, a6, a5)) & M
    e9 = (a5 + a9 - T2f(a8, a7, a6)) & M
    e7 = (a3 + a7 - S0(a6) - Maj(a6, a5, a4)) & M
    c6 = (a6 - S0(a5)) & M
    return (c6 - Maj(a5, a4, a3) + Ch(e9, e8, e7)) & M

def const_D(ctx, trials=64):
    a10 = 0
    vals = {D_of(rng.getrandbits(32), ctx, a10) for _ in range(trials)}
    return len(vals) == 1

full = both = only45 = only_e9 = 0
for _ in range(300):
    v = rng.getrandbits(32)
    ctx = ctx_family(v, rng)                       # a4=a5=v, e8=e9=-1
    both += const_D(ctx)
    c = dict(ctx); c[5] = rng.getrandbits(32)      # a5 != a4, e9 still -1
    c[8] = (M - c[4] + T2f(c[7], c[6], c[5])) & M
    c[9] = (M - c[5] + T2f(c[8], c[7], c[6])) & M
    only_e9 += const_D(c)
    c = ctx_family(v, rng); c[9] = rng.getrandbits(32)   # a4=a5=v but e9 generic
    only45 += const_D(c)
print(f"[c] D(a3) constant:  a4=a5 AND e9=-1 -> {both}/300 ;  "
      f"e9=-1 only -> {only_e9}/300 ;  a4=a5 only -> {only45}/300")

# closed form of the constant
bad = 0
for _ in range(500):
    v = rng.getrandbits(32); ctx = ctx_family(v, rng)
    want = (ctx[6] - S0(v) - v + M) & M           # c6 - Maj(v,v,a3) + e8,  Maj(x,x,y)=x, e8=-1
    if D_of(rng.getrandbits(32), ctx, 0) != want: bad += 1
print(f"[c] D == a6 - Sigma0(v) - v - 1  (mod 2^32): failures {bad}/500")
print("DONE")
