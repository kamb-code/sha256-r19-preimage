#!/usr/bin/env python3
"""R=19: 'iteration-free' contexts.  With a5 = a4 and e9 = 0xFFFFFFFF (chosen via
a9), the a3-dependence of the C1 target D(a3) = (a6 - S0(a5) - Maj(a5,a4,a3))
+ Ch(e9, e8, a3 + T1_7) vanishes, so per a0 the chain is three table lookups
with no iteration: a1 from C0 (provisional W9hat), a2 from C1, a3 from C2.
The only remaining condition is the W9 consistency eps(a2,a3) = 0.

Measures, for a random target: fixed points / lo-passes / hi-passes per a0 in
(i) a degenerate context and (ii) a random context, and forward-verifies any
full match.  Uses the published sigma0 table (16 GB in RAM).
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
U32 = np.uint32; MISS = U32(M)

def rotr(x, n):
    if isinstance(x, np.ndarray): return ((x >> U32(n)) | (x << U32(32 - n))) & MISS
    return ((x >> n) | (x << (32 - n))) & M
def S0(x): return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
def S1(x): return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
def s0(x): return rotr(x, 7) ^ rotr(x, 18) ^ (x >> (U32(3) if isinstance(x, np.ndarray) else 3))
def s1(x): return rotr(x, 17) ^ rotr(x, 19) ^ (x >> (U32(10) if isinstance(x, np.ndarray) else 10))
def Ch(e, f, g): return ((e & f) ^ (~e & g)) if isinstance(e, np.ndarray) else (((e & f) ^ ((~e) & g)) & M)
def Maj(a, b, c): return (a & b) ^ (a & c) ^ (b & c)
def T2(a, b, c): return (S0(a) + Maj(a, b, c)) & (MISS if isinstance(a, np.ndarray) else M)
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

R = 19
def backward_chain(h):
    """From the 19-round digest recover a11..a18 (and e15..e18)."""
    s = [(struct.unpack(">I", h[4*i:4*i+4])[0] - IV[i]) & M for i in range(8)]
    a = {18: s[0], 17: s[1], 16: s[2], 15: s[3]}; e = {18: s[4], 17: s[5], 16: s[6], 15: s[7]}
    for r in (18, 17, 16, 15):                       # a_{r-4} = e_r - T1_r, T1_r = a_r - T2(a_{r-1},a_{r-2},a_{r-3})
        T1 = (a[r] - T2(a[r-1], a[r-2], a[r-3])) & M
        a[r-4] = (e[r] - T1) & M
    return a, e

t0 = time.time()
Tinv = np.load('/nvme0n1-disk/Kamvid/sigma0_u_table.npy', mmap_mode='r').view(np.uint32)
print(f"sigma0 table loaded {time.time()-t0:.0f}s", flush=True)

def run_context(h, ctx, N, seed, label):
    """ctx: dict a4..a10.  R19 frame A loop from (0,0), 8 iterations, over N random a0."""
    ab, eb = backward_chain(h)
    a = dict(ab); a.update(ctx); a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(8, 19):                            # e_r for r >= 8 needs a_{r-4}.. : all known for r>=8
        e[r] = (a[r-4] + a[r] - T2(a[r-1], a[r-2], a[r-3])) & M
    for r in (15, 16, 17, 18): assert e[r] == eb[r]
    a4, a5, a6, a7, a8, a9, a10, a11 = (a[i] for i in range(4, 12))
    am1, am2, am3, am4 = a[-1], a[-2], a[-3], a[-4]; em1, em2, em3, em4 = e[-1], e[-2], e[-3], e[-4]
    c8 = (a8 - S0(a7) - Maj(a7, a6, a5)) & M; c7 = (a7 - S0(a6)) & M; c6 = (a6 - S0(a5)) & M
    e8 = (a4 + c8) & M; e9 = (a5 + a9 - T2(a8, a7, a6)) & M; e10 = (a6 + a10 - T2(a9, a8, a7)) & M
    W9base = ((a9 - T2(a8, a7, a6)) - K[9]) & M
    W10base = ((a10 - T2(a9, a8, a7)) - S1(e9) - K[10]) & M
    W11base = ((a11 - T2(a10, a9, a8)) - S1(e10) - K[11]) & M
    Wr = {r: recW(a, e, r) for r in range(14, 19)}
    K0p = (Wr[16] - s1(Wr[14])) & M; K1p = (Wr[17] - s1(Wr[15])) & M; K2p = (Wr[18] - s1(Wr[16])) & M
    T2iv = T2(am1, am2, am3)
    C0c = (-T2iv - em4 - S1(em1) - Ch(em1, em2, em3) - K[0]) & M; Ce0 = (am4 - T2iv) & M
    # provisional W9hat at a2=a3=0
    T15h = (a5 - S0(a4) - Maj(a4, 0, 0)) & M
    e7h = (c7 - Maj(a6, a5, a4)) & M; e6h = (c6 - Maj(a5, a4, 0)) & M
    W9hat = (W9base - T15h - S1(e8) - Ch(e8, e7h, e6h)) & M
    print(f"[{label}] a5==a4: {a5 == a4}  e9=0x{e9:08x}  W9hat=0x{W9hat:08x}", flush=True)

    r = np.random.default_rng(seed)
    tot = dict(surv=0, fp=0, lo=0, hi=0, ver=0)
    B = 1 << 20
    for b in range(N // B):
        A0 = r.integers(0, 1 << 32, size=B, dtype=np.uint64).astype(U32)
        E0 = (A0 + c(Ce0)) & MISS; W0 = (A0 + c(C0c)) & MISS
        G = (-(S0(A0) + Maj(A0, c(am1), c(am2))) - c(em3) - S1(E0) - Ch(E0, c(em1), c(em2)) - c(K[1])) & MISS
        F0 = (c(K0p) - W0 - c(W9hat) - G) & MISS
        W1 = Tinv[F0]; ok0 = W1 != MISS
        A0 = A0[ok0]; E0 = E0[ok0]; W0 = W0[ok0]; G = G[ok0]; W1 = W1[ok0]
        n = A0.size; tot['surv'] += n
        A1 = (W1 - G) & MISS
        E1 = (c(am3) + A1 - (S0(A0) + Maj(A0, c(am1), c(am2)))) & MISS
        F12 = (-(S0(A1) + Maj(A1, A0, c(am1))) - c(em2) - S1(E1) - Ch(E1, E0, c(em1)) - c(K[2])) & MISS
        A2 = np.zeros(n, U32); A3 = np.zeros(n, U32)
        conv = np.zeros(n, bool)
        for it in range(8):
            # C1 -> a2 from current a3
            e7 = (A3 + c(c7) - c(Maj(a6, a5, a4))) & MISS
            D = ((c(a6) - c(S0(a5)) - Maj(c(a5), c(a4), A3)) + Ch(c(e9), c(e8), e7)) & MISS
            R1 = (c(K1p) - W1 - F12 - c(W10base) + D) & MISS
            W2 = Tinv[R1]; h1 = W2 != MISS
            A2n = (W2 - F12) & MISS
            # C2 -> a3 from the NEW a2 (production order)
            e2 = (c(am2) + A2n - (S0(A1) + Maj(A1, A0, c(am1)))) & MISS
            F23 = (-(S0(A2n) + Maj(A2n, A1, A0)) - c(em1) - S1(e2) - Ch(e2, E1, E0) - c(K[3])) & MISS
            W2v = np.where(h1, W2, U32(0))
            R2 = (c(K2p) - c(W11base) + c((a7 - T2(a6, a5, a4)) & M) - W2v - F23) & MISS   # -W11 - W3 + ... : W11 = W11base - e7 - Ch(e10,e9,e8), e7 = a3 + T1_7
            R2 = (R2 + c(Ch(e10, e9, e8))) & MISS
            W3 = Tinv[R2]; h2 = W3 != MISS
            A3n = (W3 - F23) & MISS
            both = h1 & h2
            A2n = np.where(both, A2n, A2); A3n = np.where(both, A3n, A3)
            fixed = both & (A2n == A2) & (A3n == A3)
            newfp = fixed & ~conv; conv |= fixed
            A2 = np.where(both, A2n, A2); A3 = np.where(both, A3n, A3)
        tot['fp'] += int(conv.sum())
        # W9 consistency on fixed points
        idx = np.nonzero(conv)[0]
        if idx.size:
            a2f, a3f, a0f, a1f = A2[idx], A3[idx], A0[idx], A1[idx]
            e6 = (a2f + c(c6) - Maj(c(a5), c(a4), a3f)) & MISS
            e7 = (a3f + c(c7) - c(Maj(a6, a5, a4))) & MISS
            T15 = (c(a5) - c(S0(a4)) - Maj(c(a4), a3f, a2f)) & MISS
            W9conv = (c(W9base) - T15 - c(S1(e8)) - Ch(c(e8), e7, e6)) & MISS
            lo = (W9conv & U32(0xFFFF)) == U32(W9hat & 0xFFFF)
            hi = (W9conv >> U32(16)) == U32(W9hat >> 16)
            tot['lo'] += int(lo.sum()); tot['hi'] += int((lo & hi).sum())
            for j in np.nonzero(lo & hi)[0]:
                aa = dict(a); aa.update({0: int(a0f[j]), 1: int(a1f[j]), 2: int(a2f[j]), 3: int(a3f[j])})
                ee = dict(e)
                for rr in range(0, 19): ee[rr] = (aa[rr-4] + aa[rr] - T2(aa[rr-1], aa[rr-2], aa[rr-3])) & M
                Wm = [recW(aa, ee, rr) for rr in range(16)]
                ok = digest(Wm, R) == h
                tot['ver'] += int(ok)
                print(f"  FULL MATCH a0=0x{int(a0f[j]):08x}: verified={ok} words={[hex(w) for w in Wm]}", flush=True)
    n_a0 = N
    print(f"[{label}] a0 tried {n_a0:,}: C0 survivors {tot['surv']:,}, fixed points {tot['fp']:,} "
          f"({tot['fp']/max(tot['surv'],1):.3e} per survivor), lo-pass {tot['lo']}, full {tot['hi']}, verified {tot['ver']}", flush=True)
    return tot


# ---- scale run: many targets, degenerate family, collect every preimage ----
N = 1 << 24
found_all = []
tot = dict(surv=0, fp=0, lo=0, hi=0, ver=0)
t_start = time.time()
for tseed in range(200, 208):
    rng = np.random.default_rng(tseed)
    msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
    h = digest([struct.unpack(">I", (msg + b"\x80" + b"\x00"*(56-1-55) + struct.pack(">Q", 55*8))[4*i:4*i+4])[0] for i in range(16)], R)
    ctx = {r: int(x) for r, x in zip(range(4, 11), rng.integers(0, 1 << 32, 7, dtype=np.uint64))}
    ctx[5] = ctx[4]
    a5, a6, a7, a8 = ctx[5], ctx[6], ctx[7], ctx[8]
    ctx[9] = (M - a5 + T2(a8, a7, a6)) & M
    print(f"=== target {tseed}: {h.hex()}", flush=True)
    t = run_context(h, ctx, N, tseed * 7, f"deg t{tseed}")
    for k in tot: tot[k] += t[k]
print("")
print(f"SCALE TOTAL (8 targets, {8*N:,} a0, degenerate family):")
print(f"  C0 survivors {tot['surv']:,}")
print(f"  fixed points {tot['fp']:,}  = {tot['fp']/tot['surv']:.4f} per survivor")
print(f"  lo-passes    {tot['lo']:,}  = {tot['lo']/tot['fp']:.3e} per fixed point (uniform 1.526e-05, enh {tot['lo']/tot['fp']/1.5259e-5:.1f}x)")
print(f"  full matches {tot['hi']:,}  = {tot['hi']/max(tot['lo'],1):.3e} per lo-pass")
print(f"  verified     {tot['ver']:,}")
print(f"  PREIMAGES PER a0: {tot['ver']/(8*N):.3e} = 2^{np.log2(max(tot['ver'],1)/(8*N)):.1f}")
print(f"  [{time.time()-t_start:.0f}s]")
print("DONE", flush=True)
