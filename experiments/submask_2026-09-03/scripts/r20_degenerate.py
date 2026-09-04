#!/usr/bin/env python3
"""R=20 in the iteration-free (degenerate) context family.

Same construction as r19_degenerate.py but at 20 rounds: backward chain gives
a12..a19, context is a4..a11 with a5 = a4 and a9 chosen so e9 = 0xFFFFFFFF, and
C0/C1/C2 solve triangularly (a1, then a2, then a3, one table lookup each, no
iteration).  Two 32-bit conditions then remain:

    eps  = W9_conv - W9_hat          (the R=19 consistency check)
    C3   = sigma0(W4) + W3 - K3'     (the fourth schedule constraint)

Because the family produces ~0.4 solutions of C0/C1/C2 per swept a0, we can
generate ~10^7 candidates cheaply and measure BOTH residual distributions --
and, for the first time at full width, whether the two are independent.

Reports, over N swept a0:
  * fixed-point rate (expect 0.634^2 = 0.402)
  * P(eps == 0), P(eps_lo == 0), P(C3 == 0), P(C3_lo == 0) against uniform
  * the 2x2 joint table on the low halves and a chi-squared independence test
  * any full (eps == 0 and C3 == 0) solution, forward-verified
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
U32 = np.uint32; MISS = U32(M); LO16 = U32(0xFFFF)

def rotr(x, n):
    if isinstance(x, np.ndarray): return ((x >> U32(n)) | (x << U32(32 - n))) & MISS
    return ((x >> n) | (x << (32 - n))) & M
def S0(x): return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
def S1(x): return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
def s0(x): return rotr(x, 7) ^ rotr(x, 18) ^ (x >> (U32(3) if isinstance(x, np.ndarray) else 3))
def s1(x): return rotr(x, 17) ^ rotr(x, 19) ^ (x >> (U32(10) if isinstance(x, np.ndarray) else 10))
def Ch(e, f, g): return ((e & f) ^ (~e & g)) if isinstance(e, np.ndarray) else (((e & f) ^ ((~e) & g)) & M)
def Maj(a, b, c_): return (a & b) ^ (a & c_) ^ (b & c_)
def T2(a, b, c_): return (S0(a) + Maj(a, b, c_)) & (MISS if isinstance(a, np.ndarray) else M)
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

blk = os.urandom(55); pad = blk + b"\x80" + b"\x00"*(56-1-55) + struct.pack(">Q", 55*8)
assert digest([struct.unpack(">I", pad[4*i:4*i+4])[0] for i in range(16)], 64) == hashlib.sha256(blk).digest()

R = 20
def backward_chain(h, R):
    s = [(struct.unpack(">I", h[4*i:4*i+4])[0] - IV[i]) & M for i in range(8)]
    a = {R-1: s[0], R-2: s[1], R-3: s[2], R-4: s[3]}; e = {R-1: s[4], R-2: s[5], R-3: s[6], R-4: s[7]}
    for r in (R-1, R-2, R-3, R-4):
        a[r-4] = (e[r] - ((a[r] - T2(a[r-1], a[r-2], a[r-3])) & M)) & M
    return a, e

t0 = time.time()
Tinv = np.load('/nvme0n1-disk/Kamvid/sigma0_u_table.npy', mmap_mode='r').view(np.uint32)
print(f"sigma0 table mapped {time.time()-t0:.0f}s", flush=True)

def run(h, ctx, N, seed, label):
    ab, eb = backward_chain(h, R)
    a = dict(ab); a.update(ctx); a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(8, R):
        e[r] = (a[r-4] + a[r] - T2(a[r-1], a[r-2], a[r-3])) & M
    for r in range(R-4, R): assert e[r] == eb[r], f"e{r} mismatch"
    a4, a5, a6, a7, a8, a9, a10, a11 = (a[i] for i in range(4, 12))
    am1, am2, am3, am4 = a[-1], a[-2], a[-3], a[-4]; em1, em2, em3, em4 = e[-1], e[-2], e[-3], e[-4]
    c8 = (a8 - S0(a7) - Maj(a7, a6, a5)) & M; c7 = (a7 - S0(a6)) & M; c6 = (a6 - S0(a5)) & M
    e8 = (a4 + c8) & M; e9v = (a5 + a9 - T2(a8, a7, a6)) & M; e10 = (a6 + a10 - T2(a9, a8, a7)) & M
    T1_7 = (a7 - T2(a6, a5, a4)) & M
    W9base = ((a9 - T2(a8, a7, a6)) - K[9]) & M
    W10base = ((a10 - T2(a9, a8, a7)) - S1(e9v) - K[10]) & M
    W11base = ((a11 - T2(a10, a9, a8)) - S1(e10) - K[11]) & M
    Wr = {r: recW(a, e, r) for r in range(12, R)}
    K0p = (Wr[16] - s1(Wr[14])) & M; K1p = (Wr[17] - s1(Wr[15])) & M
    K2p = (Wr[18] - s1(Wr[16])) & M; K3p = (Wr[19] - s1(Wr[17]) - Wr[12]) & M
    T2iv = T2(am1, am2, am3)
    C0c = (-T2iv - em4 - S1(em1) - Ch(em1, em2, em3) - K[0]) & M; Ce0 = (am4 - T2iv) & M
    T15h = (a5 - S0(a4) - Maj(a4, 0, 0)) & M
    e7h = (c7 - Maj(a6, a5, a4)) & M; e6h = (c6 - Maj(a5, a4, 0)) & M
    W9hat = (W9base - T15h - S1(e8) - Ch(e8, e7h, e6h)) & M
    print(f"[{label}] a5==a4 {a5==a4}  e9=0x{e9v:08x}  W9hat=0x{W9hat:08x}", flush=True)

    r = np.random.default_rng(seed)
    tot = dict(surv=0, fp=0, eps0=0, epslo=0, c30=0, c3lo=0, both_lo=0, ver=0)
    B = 1 << 20
    for b in range(max(N // B, 1)):
        A0 = r.integers(0, 1 << 32, size=B, dtype=np.uint64).astype(U32)
        E0 = (A0 + c(Ce0)) & MISS; W0 = (A0 + c(C0c)) & MISS
        G = (-(S0(A0) + Maj(A0, c(am1), c(am2))) - c(em3) - S1(E0) - Ch(E0, c(em1), c(em2)) - c(K[1])) & MISS
        F0 = (c(K0p) - W0 - c(W9hat) - G) & MISS
        W1 = np.asarray(Tinv[F0]); ok0 = W1 != MISS
        A0 = A0[ok0]; E0 = E0[ok0]; W0 = W0[ok0]; G = G[ok0]; W1 = W1[ok0]
        n = A0.size; tot['surv'] += n
        A1 = (W1 - G) & MISS
        E1 = (c(am3) + A1 - (S0(A0) + Maj(A0, c(am1), c(am2)))) & MISS
        F12 = (-(S0(A1) + Maj(A1, A0, c(am1))) - c(em2) - S1(E1) - Ch(E1, E0, c(em1)) - c(K[2])) & MISS
        # C1 -> a2 (D is constant in this family: a3 drops out)
        A3z = U32(0)
        e7c = (A3z + c(c7) - c(Maj(a6, a5, a4))) & MISS
        D = ((c(a6) - c(S0(a5)) - Maj(c(a5), c(a4), A3z)) + Ch(c(e9v), c(e8), e7c)) & MISS
        R1 = (c(K1p) - W1 - F12 - c(W10base) + D) & MISS
        W2 = np.asarray(Tinv[R1]); h1 = W2 != MISS
        A0, A1, E0, E1, W0, W1, F12, W2 = (x[h1] for x in (A0, A1, E0, E1, W0, W1, F12, W2))
        A2 = (W2 - F12) & MISS
        # C2 -> a3
        e2 = (c(am2) + A2 - (S0(A1) + Maj(A1, A0, c(am1)))) & MISS
        F23 = (-(S0(A2) + Maj(A2, A1, A0)) - c(em1) - S1(e2) - Ch(e2, E1, E0) - c(K[3])) & MISS
        R2 = (c(K2p) - c(W11base) + c(T1_7) + c(Ch(e10, e9v, e8)) - W2 - F23) & MISS
        W3 = np.asarray(Tinv[R2]); h2 = W3 != MISS
        A0, A1, A2, E0, E1, e2, F23, W3 = (x[h2] for x in (A0, A1, A2, E0, E1, e2, F23, W3))
        A3 = (W3 - F23) & MISS
        m = A0.size; tot['fp'] += m
        if m == 0: continue
        # eps residual (W9 consistency)
        e6 = (A2 + c(c6) - Maj(c(a5), c(a4), A3)) & MISS
        e7 = (A3 + c(c7) - c(Maj(a6, a5, a4))) & MISS
        T15 = (c(a5) - c(S0(a4)) - Maj(c(a4), A3, A2)) & MISS
        W9conv = (c(W9base) - T15 - c(S1(e8)) - Ch(c(e8), e7, e6)) & MISS
        eps = (W9conv - c(W9hat)) & MISS
        # C3 residual
        e3 = (c(am1) + A3 - (S0(A2) + Maj(A2, A1, A0))) & MISS
        W4 = (c(a4) - (S0(A3) + Maj(A3, A2, A1)) - E0 - S1(e3) - Ch(e3, e2, E1) - c(K[4])) & MISS
        c3 = (s0(W4) + W3 - c(K3p)) & MISS
        eps_lo = (eps & LO16) == U32(0); c3_lo = (c3 & LO16) == U32(0)
        tot['eps0'] += int((eps == U32(0)).sum()); tot['epslo'] += int(eps_lo.sum())
        tot['c30'] += int((c3 == U32(0)).sum()); tot['c3lo'] += int(c3_lo.sum())
        tot['both_lo'] += int((eps_lo & c3_lo).sum())
        full = (eps == U32(0)) & (c3 == U32(0))
        for j in np.nonzero(full)[0]:
            aa = dict(a); aa.update({0: int(A0[j]), 1: int(A1[j]), 2: int(A2[j]), 3: int(A3[j])})
            ee = dict(e)
            for rr in range(0, R): ee[rr] = (aa[rr-4] + aa[rr] - T2(aa[rr-1], aa[rr-2], aa[rr-3])) & M
            Wm = [recW(aa, ee, rr) for rr in range(16)]
            ok = digest(Wm, R) == h
            tot['ver'] += int(ok)
            print(f"  *** R=20 FULL MATCH verified={ok} words={[hex(w) for w in Wm]}", flush=True)
    fp = max(tot['fp'], 1)
    print(f"[{label}] a0 {N:,}: survivors {tot['surv']:,}  fixed points {tot['fp']:,} "
          f"({tot['fp']/max(tot['surv'],1):.4f}/survivor)", flush=True)
    print(f"    eps==0     {tot['eps0']:>8,}  rate {tot['eps0']/fp:.3e}  (uniform 2.33e-10, enh {tot['eps0']/fp/2.3283e-10:.0f}x)", flush=True)
    print(f"    eps_lo==0  {tot['epslo']:>8,}  rate {tot['epslo']/fp:.3e}  (uniform 1.526e-05, enh {tot['epslo']/fp/1.5259e-5:.1f}x)", flush=True)
    print(f"    C3==0      {tot['c30']:>8,}  rate {tot['c30']/fp:.3e}", flush=True)
    print(f"    C3_lo==0   {tot['c3lo']:>8,}  rate {tot['c3lo']/fp:.3e}  (uniform 1.526e-05, enh {tot['c3lo']/fp/1.5259e-5:.2f}x)", flush=True)
    exp_both = tot['epslo'] * tot['c3lo'] / fp
    print(f"    both_lo    {tot['both_lo']:>8,}  vs {exp_both:.2f} expected under independence  "
          f"(ratio {tot['both_lo']/max(exp_both,1e-9):.2f})", flush=True)
    print(f"    verified   {tot['ver']}", flush=True)
    return tot

N = int(sys.argv[1]) if len(sys.argv) > 1 else (1 << 24)
grand = dict(surv=0, fp=0, eps0=0, epslo=0, c30=0, c3lo=0, both_lo=0, ver=0)
t_start = time.time()
for tseed in range(300, 306):
    rng = np.random.default_rng(tseed)
    msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
    h = digest([struct.unpack(">I", (msg + b"\x80" + b"\x00"*(56-1-55) + struct.pack(">Q", 55*8))[4*i:4*i+4])[0] for i in range(16)], R)
    ctx = {r: int(x) for r, x in zip(range(4, 12), rng.integers(0, 1 << 32, 8, dtype=np.uint64))}
    ctx[5] = ctx[4]
    a5, a6, a7, a8 = ctx[5], ctx[6], ctx[7], ctx[8]
    ctx[9] = (M - a5 + T2(a8, a7, a6)) & M
    print(f"=== R=20 target {tseed}: {h.hex()}", flush=True)
    t = run(h, ctx, N, tseed * 13, f"deg t{tseed}")
    for k in grand: grand[k] += t[k]
fp = max(grand['fp'], 1)
print("")
print(f"R=20 GRAND TOTAL (6 targets, {6*N:,} a0, degenerate family):")
print(f"  fixed points {grand['fp']:,} ({grand['fp']/max(grand['surv'],1):.4f} per survivor)")
print(f"  eps_lo==0 {grand['epslo']:,} -> {grand['epslo']/fp:.3e} ({grand['epslo']/fp/1.5259e-5:.1f}x uniform)")
print(f"  C3_lo==0  {grand['c3lo']:,} -> {grand['c3lo']/fp:.3e} ({grand['c3lo']/fp/1.5259e-5:.2f}x uniform)")
eb = grand['epslo'] * grand['c3lo'] / fp
print(f"  both_lo   {grand['both_lo']:,} vs {eb:.2f} expected under independence (ratio {grand['both_lo']/max(eb,1e-9):.2f})")
print(f"  eps==0 (full 32-bit) {grand['eps0']:,} -> {grand['eps0']/fp:.3e}")
print(f"  verified R=20 preimages {grand['ver']}")
print(f"  [{time.time()-t_start:.0f}s]")
print("DONE", flush=True)
