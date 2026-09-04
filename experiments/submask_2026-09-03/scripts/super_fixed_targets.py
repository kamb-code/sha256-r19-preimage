#!/usr/bin/env python3
"""The 'submask' context family.

Take the iteration-free family further.  Impose FOUR context conditions:

    a4 = 0,  a5 = 0,  e8 = 0xFFFFFFFF (choose a8),  e9 = 0xFFFFFFFF (choose a9)

leaving a6, a7, a10 (and a11 at R=20) free.  Then, writing T1_5 for the round-5
quantity and Ch for the choice function,

    T1_5 = a5 - Sigma0(a4) - Maj(a4,a3,a2) = -(a3 & a2)          [a4 = a5 = 0]
    Ch(e8,e7,e6) = e7 = a3 + T1_7                                [e8 = -1]
    Maj(a5,a4,a3) = 0  and  Ch(e9,e8,e7) = e8                    [a5 = a4 = 0, e9 = -1]

so (i) the C1 target loses its a3-dependence, hence C0/C1/C2 solve triangularly
in three table lookups per swept a0, and (ii) the W9 consistency residual
collapses to

    eps = W9_conv - W9_hat = (a3 & a2) - a3 = -(a3 & ~a2),

so the final 32-bit check becomes the BITWISE condition a3 & ~a2 == 0, i.e.
a3 is a submask of a2.  For near-uniform a2, a3 that holds with probability
(3/4)^32 = 1.0037e-4, not 2^-32 -- an enhancement of 2^18.7.

Verifies every hit against a from-scratch SHA-256.  Usage:
    python3 super_degenerate.py [R] [N_a0] [n_targets]
"""
import hashlib, os, struct, sys, time
import numpy as np

M = 0xFFFFFFFF
K = [0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
     0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
     0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
     0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
     0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
     0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
     0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
     0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]
IV = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]
U32 = np.uint32; MISS = U32(M); Z = U32(0)

def rotr(x, n):
    if isinstance(x, np.ndarray): return ((x >> U32(n)) | (x << U32(32 - n))) & MISS
    return ((x >> n) | (x << (32 - n))) & M
def S0(x): return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
def S1(x): return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
def s0(x): return rotr(x, 7) ^ rotr(x, 18) ^ (x >> (U32(3) if isinstance(x, np.ndarray) else 3))
def s1(x): return rotr(x, 17) ^ rotr(x, 19) ^ (x >> (U32(10) if isinstance(x, np.ndarray) else 10))
def Ch(e, f, g): return ((e & f) ^ (~e & g)) if isinstance(e, np.ndarray) else (((e & f) ^ ((~e) & g)) & M)
def Maj(a, b, cc): return (a & b) ^ (a & cc) ^ (b & cc)
def T2(a, b, cc): return (S0(a) + Maj(a, b, cc)) & (MISS if isinstance(a, np.ndarray) else M)
def c(x): return U32(x & M)

def forward(W, R):
    a = {-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]}; e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    Wf = list(W)
    for t in range(16, R): Wf.append((s1(Wf[t-2]) + Wf[t-7] + s0(Wf[t-15]) + Wf[t-16]) & M)
    for r in range(R):
        T1 = (e[r-4] + S1(e[r-1]) + Ch(e[r-1], e[r-2], e[r-3]) + K[r] + Wf[r]) & M
        a[r] = (T1 + T2(a[r-1], a[r-2], a[r-3])) & M; e[r] = (a[r-4] + T1) & M
    return a, e, Wf
def digest(W, R):
    a, e, _ = forward(W, R); s = [a[R-1], a[R-2], a[R-3], a[R-4], e[R-1], e[R-2], e[R-3], e[R-4]]
    return b"".join(struct.pack(">I", (x + y) & M) for x, y in zip(s, IV))
def recW(a, e, r):
    return (a[r] - T2(a[r-1], a[r-2], a[r-3]) - e[r-4] - S1(e[r-1]) - Ch(e[r-1], e[r-2], e[r-3]) - K[r]) & M
def bc(h, R):
    s = [(struct.unpack(">I", h[4*i:4*i+4])[0] - IV[i]) & M for i in range(8)]
    a = {R-1: s[0], R-2: s[1], R-3: s[2], R-4: s[3]}; e = {R-1: s[4], R-2: s[5], R-3: s[6], R-4: s[7]}
    for r in (R-1, R-2, R-3, R-4):
        a[r-4] = (e[r] - ((a[r] - T2(a[r-1], a[r-2], a[r-3])) & M)) & M
    return a, e

blk = os.urandom(55); pad = blk + b"\x80" + b"\x00"*(56-1-55) + struct.pack(">Q", 55*8)
assert digest([struct.unpack(">I", pad[4*i:4*i+4])[0] for i in range(16)], 64) == hashlib.sha256(blk).digest()
print("forward model matches hashlib at 64 rounds", flush=True)

t0 = time.time()
Tinv = np.load('/nvme0n1-disk/Kamvid/sigma0_u_table.npy', mmap_mode='r').view(np.uint32)
print(f"sigma0 table mapped {time.time()-t0:.0f}s", flush=True)

R = int(sys.argv[1]) if len(sys.argv) > 1 else 19
N = int(sys.argv[2]) if len(sys.argv) > 2 else (1 << 22)
NT = int(sys.argv[3]) if len(sys.argv) > 3 else 4

def make_ctx(rng, R):
    """a4=a5=0; a6,a7,a10(,a11) free; a8,a9 solve e8=e9=0xFFFFFFFF."""
    ctx = {4: 0, 5: 0}
    ctx[6] = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    ctx[7] = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    a4, a5, a6, a7 = ctx[4], ctx[5], ctx[6], ctx[7]
    ctx[8] = (M - a4 + S0(a7) + Maj(a7, a6, a5)) & M          # e8 = a4+a8-T2(a7,a6,a5) = -1
    a8 = ctx[8]
    ctx[9] = (M - a5 + S0(a8) + Maj(a8, a7, a6)) & M          # e9 = a5+a9-T2(a8,a7,a6) = -1
    ctx[10] = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    if R >= 20:
        ctx[11] = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    return ctx

def run(h, ctx, N, seed, label, R):
    ab, eb = bc(h, R)
    a = dict(ab); a.update(ctx); a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(8, R): e[r] = (a[r-4] + a[r] - T2(a[r-1], a[r-2], a[r-3])) & M
    for r in range(R-4, R): assert e[r] == eb[r], f"e{r}"
    a4, a5, a6, a7, a8, a9, a10 = (a[i] for i in range(4, 11))
    a11 = a[11]
    am1, am2, am3, am4 = a[-1], a[-2], a[-3], a[-4]; em1, em2, em3, em4 = e[-1], e[-2], e[-3], e[-4]
    e8 = e[8]; e9v = e[9]; e10 = e[10]
    assert e8 == M and e9v == M and a4 == 0 and a5 == 0, "family conditions violated"
    c7 = (a7 - S0(a6)) & M; c6 = (a6 - S0(a5)) & M
    T1_7 = (a7 - T2(a6, a5, a4)) & M
    W9base = ((a9 - T2(a8, a7, a6)) - K[9]) & M
    W10base = ((a10 - T2(a9, a8, a7)) - S1(e9v) - K[10]) & M
    W11base = ((a11 - T2(a10, a9, a8)) - S1(e10) - K[11]) & M
    Wr = {r: recW(a, e, r) for r in range(12, R)}
    K0p = (Wr[16] - s1(Wr[14])) & M; K1p = (Wr[17] - s1(Wr[15])) & M; K2p = (Wr[18] - s1(Wr[16])) & M
    K3p = ((Wr[19] - s1(Wr[17]) - Wr[12]) & M) if R >= 20 else None
    T2iv = T2(am1, am2, am3)
    C0c = (-T2iv - em4 - S1(em1) - Ch(em1, em2, em3) - K[0]) & M; Ce0 = (am4 - T2iv) & M
    # W9hat at a2=a3=0
    T15h = (a5 - S0(a4) - Maj(a4, 0, 0)) & M
    e7h = (0 + T1_7) & M; e6h = (0 + c6 - Maj(a5, a4, 0)) & M
    W9hat = (W9base - T15h - S1(e8) - Ch(e8, e7h, e6h)) & M
    # constant part of the C1 target (a3 drops out)
    D = ((a6 - S0(a5) - Maj(a5, a4, 0)) + Ch(e9v, e8, e7h)) & M

    r = np.random.default_rng(seed)
    tot = dict(surv=0, fp=0, sub=0, eps0=0, c3lo=0, ver=0, c3=0)
    hits = []
    B = 1 << 20
    for b in range(max(N // B, 1)):
        A0 = r.integers(0, 1 << 32, size=B, dtype=np.uint64).astype(U32)
        E0 = (A0 + c(Ce0)) & MISS; W0 = (A0 + c(C0c)) & MISS
        G = (-(S0(A0) + Maj(A0, c(am1), c(am2))) - c(em3) - S1(E0) - Ch(E0, c(em1), c(em2)) - c(K[1])) & MISS
        F0 = (c(K0p) - W0 - c(W9hat) - G) & MISS
        W1 = np.asarray(Tinv[F0]); ok = W1 != MISS
        A0, E0, W0, G, W1 = (x[ok] for x in (A0, E0, W0, G, W1)); tot['surv'] += A0.size
        A1 = (W1 - G) & MISS
        E1 = (c(am3) + A1 - (S0(A0) + Maj(A0, c(am1), c(am2)))) & MISS
        F12 = (-(S0(A1) + Maj(A1, A0, c(am1))) - c(em2) - S1(E1) - Ch(E1, E0, c(em1)) - c(K[2])) & MISS
        R1 = (c(K1p) - W1 - F12 - c(W10base) + c(D)) & MISS
        W2 = np.asarray(Tinv[R1]); ok = W2 != MISS
        A0, A1, E0, E1, W1, F12, W2 = (x[ok] for x in (A0, A1, E0, E1, W1, F12, W2))
        A2 = (W2 - F12) & MISS
        e2 = (c(am2) + A2 - (S0(A1) + Maj(A1, A0, c(am1)))) & MISS
        F23 = (-(S0(A2) + Maj(A2, A1, A0)) - c(em1) - S1(e2) - Ch(e2, E1, E0) - c(K[3])) & MISS
        R2 = (c(K2p) - c(W11base) + c(T1_7) + c(Ch(e10, e9v, e8)) - W2 - F23) & MISS
        W3 = np.asarray(Tinv[R2]); ok = W3 != MISS
        A0, A1, A2, E0, E1, e2, F23, W3 = (x[ok] for x in (A0, A1, A2, E0, E1, e2, F23, W3))
        A3 = (W3 - F23) & MISS
        tot['fp'] += A0.size
        if A0.size == 0: continue
        # the collapsed consistency check: eps = -(a3 & ~a2)
        sub = (A3 & ~A2) == Z
        tot['sub'] += int(sub.sum())
        # cross-check against the general eps formula
        e6 = (A2 + c(c6) - Maj(c(a5), c(a4), A3)) & MISS
        e7 = (A3 + c(T1_7)) & MISS
        T15 = (c(a5) - c(S0(a4)) - Maj(c(a4), A3, A2)) & MISS
        W9conv = (c(W9base) - T15 - c(S1(e8)) - Ch(c(e8), e7, e6)) & MISS
        eps = (W9conv - c(W9hat)) & MISS
        assert np.array_equal((eps == Z), sub), "collapsed eps disagrees with the general formula"
        tot['eps0'] += int((eps == Z).sum())
        idx = np.nonzero(sub)[0]
        if R >= 20 and idx.size:
            a0f, a1f, a2f, a3f = A0[idx], A1[idx], A2[idx], A3[idx]
            E0f, E1f, e2f, W3f = E0[idx], E1[idx], e2[idx], W3[idx]
            e3 = (c(am1) + a3f - (S0(a2f) + Maj(a2f, a1f, a0f))) & MISS
            W4 = (c(a4) - (S0(a3f) + Maj(a3f, a2f, a1f)) - E0f - S1(e3) - Ch(e3, e2f, E1f) - c(K[4])) & MISS
            c3 = (s0(W4) + W3f - c(K3p)) & MISS
            tot['c3lo'] += int(((c3 & U32(0xFFFF)) == Z).sum()); tot['c3'] += int((c3 == Z).sum())
            idx = idx[np.nonzero(c3 == Z)[0]]
        for j in idx[:50]:
            aa = dict(a); aa.update({0: int(A0[j]), 1: int(A1[j]), 2: int(A2[j]), 3: int(A3[j])})
            ee = dict(e)
            for rr in range(0, R): ee[rr] = (aa[rr-4] + aa[rr] - T2(aa[rr-1], aa[rr-2], aa[rr-3])) & M
            Wm = [recW(aa, ee, rr) for rr in range(16)]
            if digest(Wm, R) == h:
                tot['ver'] += 1
                if len(hits) < 3: hits.append((int(A0[j]), Wm))
    fp = max(tot['fp'], 1)
    print(f"[{label}] R={R} a0 {N:,}: survivors {tot['surv']:,}  triangular solutions {tot['fp']:,} "
          f"({tot['fp']/max(tot['surv'],1):.4f}/survivor)", flush=True)
    print(f"    a3 submask of a2 : {tot['sub']:,}  rate {tot['sub']/fp:.4e}   "
          f"(predicted (3/4)^32 = 1.0037e-04, uniform 2.33e-10)", flush=True)
    if R >= 20:
        print(f"    C3 == 0          : {tot['c3']:,}   C3_lo==0 {tot['c3lo']:,} "
              f"(rate {tot['c3lo']/max(tot['sub'],1):.3e} vs uniform 1.526e-05)", flush=True)
    print(f"    VERIFIED PREIMAGES: {tot['ver']:,}   -> {tot['ver']/N:.3e} per swept a0 "
          f"= 2^{np.log2(max(tot['ver'],1)/N):.1f}", flush=True)
    for a0v, Wm in hits[:2]:
        print(f"      a0=0x{a0v:08x} words={' '.join(f'{w:08x}' for w in Wm)}", flush=True)
    return tot


# fixed targets: the all-ones digest (Zaikin's) and the all-zero digest (Davydov's)
R = 19
N = 1 << 22
for name, h in (("all-ones (Zaikin target)", bytes([0xff]*32)), ("all-zero (Davydov target)", bytes(32))):
    print(f"=== {name}: {h.hex()}", flush=True)
    for k in range(3):
        rng = np.random.default_rng(9000 + k)
        t = run(h, make_ctx(rng, R), N, 5000 + k, f"{name[:8]} ctx{k}", R)
        if t['ver']: break
print("DONE", flush=True)
