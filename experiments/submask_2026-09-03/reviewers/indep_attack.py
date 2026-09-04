#!/usr/bin/env python3
"""INDEPENDENT re-implementation of the 'submask family' R=19 attack.

Builds its own sigma0(u)-u inversion table, derives the triangular C0/C1/C2
solve from scratch, and verifies every claimed preimage with a from-scratch
SHA-256 pinned against hashlib.  Also runs the two isolation controls
(e8 != -1, and a4 != a5) and a sweep over the family parameter v.
"""
import hashlib, math, os, struct, sys, time
import numpy as np

M = 0xFFFFFFFF
U = np.uint32
MISS = U(M)

Kc = [0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]
IV = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]

# ---- scalar (python int) primitives, straight from FIPS 180-4
rot = lambda x,n: ((x >> n) | (x << (32-n))) & M
S0  = lambda x: rot(x,2) ^ rot(x,13) ^ rot(x,22)
S1  = lambda x: rot(x,6) ^ rot(x,11) ^ rot(x,25)
s0  = lambda x: rot(x,7) ^ rot(x,18) ^ (x >> 3)
s1  = lambda x: rot(x,17) ^ rot(x,19) ^ (x >> 10)
Ch  = lambda x,y,z: ((x & y) ^ ((~x & M) & z)) & M
Maj = lambda x,y,z: ((x & y) ^ (x & z) ^ (y & z)) & M
T2  = lambda a,b,c: (S0(a) + Maj(a,b,c)) & M

# ---- vector (uint32 ndarray) primitives
def vrot(x,n): return ((x >> U(n)) | (x << U(32-n)))
def vS0(x): return vrot(x,2) ^ vrot(x,13) ^ vrot(x,22)
def vs0(x): return vrot(x,7) ^ vrot(x,18) ^ (x >> U(3))
def vS1(x): return vrot(x,6) ^ vrot(x,11) ^ vrot(x,25)
def vCh(x,y,z): return (x & y) ^ (~x & z)
def vMaj(x,y,z): return (x & y) ^ (x & z) ^ (y & z)
def vT2(a,b,c): return vS0(a) + vMaj(a,b,c)
def c(x): return U(x & M)

def compress(W, R=64):
    Wf = list(W)
    for t in range(16, R):
        Wf.append((s1(Wf[t-2]) + Wf[t-7] + s0(Wf[t-15]) + Wf[t-16]) & M)
    a = {-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]}
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(R):
        T1 = (e[r-4] + S1(e[r-1]) + Ch(e[r-1],e[r-2],e[r-3]) + Kc[r] + Wf[r]) & M
        a[r] = (T1 + T2(a[r-1],a[r-2],a[r-3])) & M
        e[r] = (a[r-4] + T1) & M
    return a, e, Wf

def digest(W, R=64):
    a, e, _ = compress(W, R)
    st = [a[R-1],a[R-2],a[R-3],a[R-4],e[R-1],e[R-2],e[R-3],e[R-4]]
    return b"".join(struct.pack(">I",(x+y)&M) for x,y in zip(st, IV))

blk = os.urandom(55)
pad = blk + b"\x80" + struct.pack(">Q", 55*8)
assert digest([struct.unpack(">I",pad[4*i:4*i+4])[0] for i in range(16)], 64) == hashlib.sha256(blk).digest()
print("my SHA-256 == hashlib @64 rounds", flush=True)

# ------------------------------------------------------------------ table
def build_table(order="asc"):
    t0 = time.time()
    T = np.full(1 << 32, MISS, dtype=U)
    CH = 1 << 26
    rng_order = range(0, 1 << 32, CH)
    for base in rng_order:
        u = np.arange(base, base + CH, dtype=np.uint64).astype(U)
        T[vs0(u) - u] = u
        del u
    print(f"  table built ({order}) in {time.time()-t0:.0f}s", flush=True)
    return T

T = build_table()
cov = 0
rr = np.random.default_rng(7)
probe = rr.integers(0, 1<<32, size=1<<22, dtype=np.uint64).astype(U)
val = T[probe]
hit = val != MISS
print(f"  coverage {hit.mean():.5f} (1-1/e = {1-math.exp(-1):.5f})", flush=True)
# verify the table: sigma0(T[x]) - T[x] == x for every hit
chk = vs0(val[hit]) - val[hit]
assert np.array_equal(chk, probe[hit]), "table is wrong"
print(f"  table verified on {hit.sum():,} random lookups", flush=True)
# bit statistics of the returned preimages (build-order dependent)
mean_bits = np.unpackbits(val[hit].view(np.uint8)).mean()
print(f"  mean bit density of stored preimages: {mean_bits:.5f} (uniform 0.5)", flush=True)
sub = ((val[hit][:1<<20] & ~val[hit][1<<20:2<<20]) == U(0)).mean()
print(f"  P(T[x] submask of T[y]) = {sub:.4e}   (3/4)^32 = {(3/4)**32:.4e}", flush=True)

# --------------------------------------------------------------- backward
def backward(h, R):
    s = [(struct.unpack(">I",h[4*i:4*i+4])[0] - IV[i]) & M for i in range(8)]
    a = {R-1:s[0], R-2:s[1], R-3:s[2], R-4:s[3]}
    e = {R-1:s[4], R-2:s[5], R-3:s[6], R-4:s[7]}
    for r in (R-1, R-2, R-3, R-4):
        a[r-4] = (e[r] - (a[r] - T2(a[r-1],a[r-2],a[r-3]))) & M
    return a, e

def recW(a, e, r):
    return (a[r] - T2(a[r-1],a[r-2],a[r-3]) - e[r-4] - S1(e[r-1])
            - Ch(e[r-1],e[r-2],e[r-3]) - Kc[r]) & M

def make_ctx(v, rng, force_e8=True, force_e9=True, a5_equal=True):
    a4 = v
    a5 = v if a5_equal else int(rng.integers(0,1<<32,dtype=np.uint64))
    a6 = int(rng.integers(0,1<<32,dtype=np.uint64))
    a7 = int(rng.integers(0,1<<32,dtype=np.uint64))
    a8 = ((M - a4 + T2(a7,a6,a5)) & M) if force_e8 else int(rng.integers(0,1<<32,dtype=np.uint64))
    a9 = ((M - a5 + T2(a8,a7,a6)) & M) if force_e9 else int(rng.integers(0,1<<32,dtype=np.uint64))
    a10 = int(rng.integers(0,1<<32,dtype=np.uint64))
    return {4:a4,5:a5,6:a6,7:a7,8:a8,9:a9,10:a10}

def run(h, ctx, N, seed, R=19, batch=1<<22, verify_all=False):
    ab, eb = backward(h, R)
    a = dict(ab); a.update(ctx)
    a.update({-1:IV[0], -2:IV[1], -3:IV[2], -4:IV[3]})
    e = {-1:IV[4], -2:IV[5], -3:IV[6], -4:IV[7]}
    for r in range(8, R):
        e[r] = (a[r-4] + a[r] - T2(a[r-1],a[r-2],a[r-3])) & M
    for r in range(R-4, R): assert e[r] == eb[r]
    a4,a5,a6,a7,a8,a9,a10 = (a[i] for i in range(4,11))
    am1,am2,am3,am4 = a[-1],a[-2],a[-3],a[-4]
    em1,em2,em3,em4 = e[-1],e[-2],e[-3],e[-4]
    e8, e9, e10 = e[8], e[9], e[10]

    W = {r: recW(a,e,r) for r in range(12, R)}
    K0p = (W[16] - s1(W[14])) & M
    K1p = (W[17] - s1(W[15])) & M
    K2p = (W[18] - s1(W[16])) & M

    T1_7    = (a7 - T2(a6,a5,a4)) & M                 # e7 = a3 + T1_7
    c6      = (a6 - S0(a5)) & M                       # e6 = a2 + c6 - Maj(a5,a4,a3)
    W9base  = (a9 - T2(a8,a7,a6) - Kc[9]) & M
    W10base = (a10 - T2(a9,a8,a7) - S1(e9) - Kc[10]) & M
    W11base = (a[11] - T2(a10,a9,a8) - S1(e10) - Kc[11]) & M

    # W9hat: the a1-free part of W9 evaluated at a2 = a3 = 0
    T15h = (a5 - S0(a4) - Maj(a4,0,0)) & M
    e7h  = T1_7
    e6h  = (c6 - Maj(a5,a4,0)) & M
    W9hat = (W9base - T15h - S1(e8) - Ch(e8,e7h,e6h)) & M
    # D(a3) of the C1 target, frozen at a3 = 0
    D = (c6 - Maj(a5,a4,0) + Ch(e9,e8,e7h)) & M

    T2iv = T2(am1,am2,am3)
    C0c  = (-T2iv - em4 - S1(em1) - Ch(em1,em2,em3) - Kc[0]) & M
    Ce0  = (am4 - T2iv) & M

    rng = np.random.default_rng(seed)
    tot = dict(surv=0, tri=0, sub=0, eps0=0, ver=0, c0ok=0)
    hits = []
    done = 0
    while done < N:
        B = min(batch, N - done); done += B
        A0 = rng.integers(0,1<<32,size=B,dtype=np.uint64).astype(U)
        E0 = A0 + c(Ce0)
        W0 = A0 + c(C0c)
        G  = -(vT2(A0, c(am1), c(am2))) - c(em3) - vS1(E0) - vCh(E0, c(em1), c(em2)) - c(Kc[1])
        F0 = c(K0p) - W0 - c(W9hat) - G
        W1 = T[F0]; ok = W1 != MISS
        A0,E0,W0,G,W1 = (x[ok] for x in (A0,E0,W0,G,W1))
        tot['surv'] += A0.size
        assert np.array_equal(vs0(W1) - W1, F0[ok])
        A1 = W1 - G
        E1 = c(am3) + A1 - vT2(A0, c(am1), c(am2))
        F12 = -(vT2(A1, A0, c(am1))) - c(em2) - vS1(E1) - vCh(E1, E0, c(em1)) - c(Kc[2])
        R1 = c(K1p) - W1 - F12 - c(W10base) + c(D)
        W2 = T[R1]; ok = W2 != MISS
        A0,A1,E0,E1,W1,F12,W2 = (x[ok] for x in (A0,A1,E0,E1,W1,F12,W2))
        A2 = W2 - F12
        E2 = c(am2) + A2 - vT2(A1, A0, c(am1))
        F23 = -(vT2(A2, A1, A0)) - c(em1) - vS1(E2) - vCh(E2, E1, E0) - c(Kc[3])
        R2 = c(K2p) - W2 - F23 - c(W11base) + c(T1_7) + c(Ch(e10,e9,e8))
        W3 = T[R2]; ok = W3 != MISS
        A0,A1,A2,E0,E1,E2,W1,W2,W3,F23 = (x[ok] for x in (A0,A1,A2,E0,E1,E2,W1,W2,W3,F23))
        A3 = W3 - F23
        tot['tri'] += A0.size
        if A0.size == 0: continue
        # true eps, computed from the general formula (no family shortcut)
        e6 = A2 + c(c6) - vMaj(c(a5), c(a4), A3)
        e7 = A3 + c(T1_7)
        T15 = c(a5) - c(S0(a4)) - vMaj(c(a4), A3, A2)
        W9conv = c(W9base) - T15 - c(S1(e8)) - vCh(c(e8), e7, e6)
        eps = W9conv - c(W9hat)
        tot['eps0'] += int((eps == U(0)).sum())
        tot['sub']  += int((vMaj(c(a4), A3, A2) == A3).sum())
        tot['bits'] = tot.get('bits',0) + int(np.unpackbits(A2[:1<<16].view(np.uint8)).sum()) \
                                        + int(np.unpackbits(A3[:1<<16].view(np.uint8)).sum())
        tot['nbits'] = tot.get('nbits',0) + 64 * min(A2.size, 1<<16)
        idx = np.nonzero(eps == U(0))[0]
        if verify_all:
            cand = np.unique(np.concatenate([idx, np.arange(min(A0.size, 2000))]))
        else:
            cand = idx[:4000]
        if verify_all:
            tot['seen'] = tot.get('seen',0) + cand.size
            tot['seen_eps'] = tot.get('seen_eps',0) + int((eps[cand] == U(0)).sum())
        for j in cand:
            aa = dict(a); aa.update({0:int(A0[j]),1:int(A1[j]),2:int(A2[j]),3:int(A3[j])})
            ee = dict(e)
            for rr_ in range(0, R):
                ee[rr_] = (aa[rr_-4] + aa[rr_] - T2(aa[rr_-1],aa[rr_-2],aa[rr_-3])) & M
            Wm = [recW(aa, ee, rr_) for rr_ in range(16)]
            # independent schedule check
            ok0 = ((s1(Wm[14]) + Wm[9] + s0(Wm[1]) + Wm[0]) & M) == W[16]
            ok1 = ((s1(Wm[15]) + Wm[10] + s0(Wm[2]) + Wm[1]) & M) == W[17]
            ok2 = ((s1(W[16])  + Wm[11] + s0(Wm[3]) + Wm[2]) & M) == W[18]
            if ok0: tot['c0ok'] += 1
            tot['c1ok'] = tot.get('c1ok',0) + int(ok1)
            tot['c2ok'] = tot.get('c2ok',0) + int(ok2)
            tot['seen2'] = tot.get('seen2',0) + 1
            if STRICT: assert ok1 and ok2, "C1/C2 should hold identically"
            if digest(Wm, R) == h:
                tot['ver'] += 1
                assert ok0 and ok1 and ok2
                if len(hits) < 2: hits.append((int(A0[j]), Wm))
    return tot, hits

R = 19
STRICT = (len(sys.argv) < 3 or sys.argv[2] != "controls")
ONLY_CTRL = not STRICT
N = int(sys.argv[1]) if len(sys.argv) > 1 else (1 << 24)

def report(tag, tot, N, extra=""):
    tri = max(tot['tri'],1)
    print(f"  [{tag}] a0={N:,}  surv={tot['surv']:,}  triangular={tot['tri']:,} "
          f"({tot['tri']/N:.4f}/a0)  eps==0: {tot['eps0']:,} ({tot['eps0']/tri:.4e}) "
          f"VERIFIED={tot['ver']:,} {extra}", flush=True)

def target_for(seed, R):
    g = np.random.default_rng(seed)
    m = bytes(g.integers(0,256,55,dtype=np.uint8).tolist())
    p = m + b"\x80" + struct.pack(">Q", 55*8)
    return digest([struct.unpack(">I",p[4*i:4*i+4])[0] for i in range(16)], R)

ONES = bytes.fromhex("ff"*32)

if not ONLY_CTRL:
    print("\n=== structural check: for EVERY triangular solution, C1 and C2 hold "
          "identically and C0 holds iff eps==0 ===", flush=True)
    h = target_for(500, R)
    rng = np.random.default_rng(4242)
    tot, hits = run(h, make_ctx(0x9e3779b9, rng), 1 << 24, seed=5, R=R, batch=1<<20, verify_all=True)
    print(f"  triangular {tot['tri']:,}; fully recomputed a sample of {tot.get('seen',0):,} of them: "
          f"C1 and C2 held for ALL (asserted), C0 held for {tot['c0ok']:,}, "
          f"eps==0 for {tot.get('seen_eps',0):,}, verified preimages {tot['ver']:,} "
          f"-> C0 <=> eps==0 <=> preimage: "
          f"{tot['c0ok'] == tot.get('seen_eps',0) == tot['ver']}", flush=True)

    print("\n=== main family sweep over v (a4=a5=v, e8=e9=-1) ===", flush=True)
    grand = {}
    for v in (0x00000000, 0x00000001, 0x9e3779b9, 0xffffffff, 0x0000ffff):
        G = dict(surv=0,tri=0,sub=0,eps0=0,ver=0,c0ok=0,bits=0,nbits=0)
        t0 = time.time()
        for ti in range(4):
            h = target_for(500+ti, R)
            rng = np.random.default_rng(1000*ti + (v & 0xffff))
            ctx = make_ctx(v, rng)
            tot, hits = run(h, ctx, N, seed=7*ti+3, R=R)
            for k in G: G[k] += tot.get(k,0)
        tri = max(G['tri'],1)
        print(f"v=0x{v:08x}: triangular {G['tri']:,} ({G['tri']/(4*N):.4f}/a0)  "
              f"eps==0 {G['eps0']:,} rate {G['eps0']/tri:.4e} (pred {(3/4)**32:.4e})  "
              f"Maj==a3 {G['sub']:,}  bitdens(a2,a3) {G['bits']/max(G['nbits'],1):.5f}  "
              f"VERIFIED {G['ver']:,}  "
              f"a0/preimage {4*N/max(G['ver'],1):,.0f} = 2^{math.log2(4*N/max(G['ver'],1)):.2f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        grand[v] = G

print("\n=== controls ===", flush=True)
for name, kw in (("e8 NOT forced (a4=a5, e9=-1)", dict(force_e8=False)),
                 ("a5 free (e8=e9=-1)",           dict(a5_equal=False)),
                 ("e9 NOT forced (a4=a5, e8=-1)", dict(force_e9=False))):
    G = dict(surv=0,tri=0,sub=0,eps0=0,ver=0,c0ok=0,bits=0,nbits=0,c1ok=0,c2ok=0,seen2=0)
    for ti in range(2):
        h = target_for(500+ti, R)
        rng = np.random.default_rng(31337+ti)
        ctx = make_ctx(0, rng, **kw)
        tot, hits = run(h, ctx, N, seed=99*ti+1, R=R, verify_all=True)
        for k in G: G[k] += tot.get(k,0)
    tri = max(G['tri'],1)
    print(f"  {name:32s}: triangular {G['tri']:,}  eps==0 {G['eps0']:,} "
          f"({G['eps0']/tri:.3e})  VERIFIED {G['ver']:,}  "
          f"of {G.get('seen2',0):,} recomputed: C1 held {G.get('c1ok',0):,} "
          f"C2 held {G.get('c2ok',0):,} C0 held {G['c0ok']:,}", flush=True)

print("\n=== all-ones digest, v=0 ===", flush=True)
G = dict(surv=0,tri=0,sub=0,eps0=0,ver=0,c0ok=0,bits=0,nbits=0); t0=time.time()
rng = np.random.default_rng(2026)
ctx = make_ctx(0, rng)
tot, hits = run(ONES, ctx, 1<<22, seed=11, R=R)
print(f"  a0={1<<22:,} in {time.time()-t0:.1f}s: triangular {tot['tri']:,} "
      f"eps==0 {tot['eps0']:,} VERIFIED {tot['ver']:,}", flush=True)
for a0v, Wm in hits:
    print("  W =", " ".join(f"{w:08x}" for w in Wm), flush=True)
    assert digest(Wm, 19) == ONES
print("DONE", flush=True)
