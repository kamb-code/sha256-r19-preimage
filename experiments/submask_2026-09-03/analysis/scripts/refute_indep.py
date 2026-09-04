#!/usr/bin/env python3
"""FULLY INDEPENDENT re-implementation for refutation attempt.

Written from FIPS 180-4.  IV and K derived from prime roots, not transcribed.
Targets are os.urandom (NOT digests of any message we know).
Contexts drawn from an RNG seeded independently of the target.
Exact swept counter (no N//B rounding).
Every hit re-verified by a *third* implementation (publish/code/verify_r19.py logic,
re-implemented here in pure python from the spec, no numpy, no shared code path).
"""
import os, sys, struct, time, hashlib
import numpy as np

M = 0xFFFFFFFF

# ---- derive IV and K from prime roots (do not transcribe) ----
def primes(n):
    ps = []
    x = 2
    while len(ps) < n:
        if all(x % p for p in ps if p*p <= x):
            ps.append(x)
        x += 1
    return ps
def iroot(n, k):
    lo, hi = 0, 1 << ((n.bit_length() // k) + 2)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid**k <= n: lo = mid
        else: hi = mid - 1
    return lo
P = primes(64)
IV = [iroot(p << 64, 2) & M for p in P[:8]]
K  = [iroot(p << 96, 3) & M for p in P[:64]]

# sanity: these must match the published constants
assert IV[0] == 0x6a09e667 and IV[7] == 0x5be0cd19, [hex(v) for v in IV]
assert K[0] == 0x428a2f98 and K[63] == 0xc67178f2, [hex(v) for v in K[:4]]

def rr(x, n): return ((x >> n) | (x << (32 - n))) & M
def S0(x): return rr(x,2)^rr(x,13)^rr(x,22)
def S1(x): return rr(x,6)^rr(x,11)^rr(x,25)
def s0(x): return rr(x,7)^rr(x,18)^(x>>3)
def s1(x): return rr(x,17)^rr(x,19)^(x>>10)
def Ch(x,y,z): return ((x&y)^((~x)&z)) & M
def Maj(x,y,z): return (x&y)^(x&z)^(y&z)

# ---- pure-python reference compression, textbook (a..h) form, NO a_r/e_r algebra ----
def compress_R(Win, R):
    """Textbook loop.  Returns 8-word digest = IV + state after R rounds."""
    W = list(Win)
    for t in range(16, R):
        W.append((s1(W[t-2]) + W[t-7] + s0(W[t-15]) + W[t-16]) & M)
    a,b,cc,d,e,f,g,h = IV
    for t in range(R):
        T1 = (h + S1(e) + Ch(e,f,g) + K[t] + W[t]) & M
        T2 = (S0(a) + Maj(a,b,cc)) & M
        h,g,f,e,d,cc,b,a = g,f,e,(d+T1)&M,cc,b,a,(T1+T2)&M
    st = [a,b,cc,d,e,f,g,h]
    return [(x+y)&M for x,y in zip(st, IV)]

# pin against hashlib at 64 rounds
_blk = os.urandom(55)
_pad = _blk + b"\x80" + b"\x00"*0 + struct.pack(">Q", 55*8)
assert len(_pad) == 64
_w = [struct.unpack(">I", _pad[4*i:4*i+4])[0] for i in range(16)]
assert b"".join(struct.pack(">I", x) for x in compress_R(_w, 64)) == hashlib.sha256(_blk).digest()
print("independent pure-python model pinned to hashlib @64", flush=True)

U32 = np.uint32; MISS = U32(M); Z = U32(0)
def nrr(x,n): return ((x >> U32(n)) | (x << U32(32-n))) & MISS
def nS0(x): return nrr(x,2)^nrr(x,13)^nrr(x,22)
def nS1(x): return nrr(x,6)^nrr(x,11)^nrr(x,25)
def ns0(x): return nrr(x,7)^nrr(x,18)^(x >> U32(3))
def nCh(x,y,z): return (x&y)^((~x)&z)
def nMaj(x,y,z): return (x&y)^(x&z)^(y&z)
def c(x): return U32(x & M)

Tinv = np.load('/nvme0n1-disk/Kamvid/sigma0_u_table.npy', mmap_mode='r').view(np.uint32)
assert Tinv.size == (1 << 32)

# ---------- attack, re-derived independently ----------
def T2f(x,y,z): return (S0(x) + Maj(x,y,z)) & M

def back_chain(h32, R):
    """a_{R-1..R-4}, e_{R-1..R-4} from target, then a_{R-5..R-8} by e_r = a_{r-4}+a_r-T2."""
    s = [(struct.unpack(">I", h32[4*i:4*i+4])[0] - IV[i]) & M for i in range(8)]
    a = {R-1:s[0], R-2:s[1], R-3:s[2], R-4:s[3]}
    e = {R-1:s[4], R-2:s[5], R-3:s[6], R-4:s[7]}
    for r in (R-1,R-2,R-3,R-4):
        a[r-4] = (e[r] - a[r] + T2f(a[r-1],a[r-2],a[r-3])) & M
    return a, e

def make_ctx(rng, v):
    a4 = a5 = v
    a6 = int(rng.integers(0, 1<<32, dtype=np.uint64))
    a7 = int(rng.integers(0, 1<<32, dtype=np.uint64))
    a8 = (M - a4 + T2f(a7,a6,a5)) & M      # e8 = a4 + a8 - T2(a7,a6,a5) = 0xFFFFFFFF
    a9 = (M - a5 + T2f(a8,a7,a6)) & M      # e9 = -1
    a10 = int(rng.integers(0, 1<<32, dtype=np.uint64))
    return {4:a4,5:a5,6:a6,7:a7,8:a8,9:a9,10:a10}

def attack(h32, ctx, N, seed, R=19, batch=1<<20, verify_all=True):
    ab, _eb = back_chain(h32, R)
    a = dict(ab); a.update(ctx)
    a.update({-1:IV[0],-2:IV[1],-3:IV[2],-4:IV[3]})
    e = {-1:IV[4],-2:IV[5],-3:IV[6],-4:IV[7]}
    for r in range(8, R):
        e[r] = (a[r-4] + a[r] - T2f(a[r-1],a[r-2],a[r-3])) & M
    v = a[4]
    assert a[5] == v and e[8] == M and e[9] == M

    am1,am2,am3,am4 = a[-1],a[-2],a[-3],a[-4]
    em1,em2,em3,em4 = e[-1],e[-2],e[-3],e[-4]
    a6,a7,a8,a9,a10 = a[6],a[7],a[8],a[9],a[10]
    e8, e9, e10 = e[8], e[9], e[10]

    def recW(A,E,r):
        return (A[r] - T2f(A[r-1],A[r-2],A[r-3]) - E[r-4] - S1(E[r-1])
                - Ch(E[r-1],E[r-2],E[r-3]) - K[r]) & M
    Wr = {r: recW(a,e,r) for r in range(12,R)}
    K0p = (Wr[16] - s1(Wr[14])) & M
    K1p = (Wr[17] - s1(Wr[15])) & M
    K2p = (Wr[18] - s1(Wr[16])) & M

    T1_7 = (a7 - T2f(a6,a5 if False else a[5], a[4])) & M
    c6 = (a6 - S0(a[5])) & M
    W9base  = ((a9  - T2f(a8,a7,a6)) - K[9]) & M
    W10base = ((a10 - T2f(a9,a8,a7)) - S1(e9) - K[10]) & M
    W11base = ((a[11] - T2f(a10,a9,a8)) - S1(e10) - K[11]) & M

    # provisional W9 at a2=a3=0 (a1-independent part)
    T15h = (v - S0(v) - Maj(v,0,0)) & M
    e7h  = T1_7
    e6h  = (c6 - Maj(v,v,0)) & M
    W9hat = (W9base - T15h - S1(e8) - Ch(e8,e7h,e6h)) & M
    D = (c6 - Maj(v,v,0) + Ch(e9,e8,e7h)) & M

    T2iv = T2f(am1,am2,am3)
    C0c = (-T2iv - em4 - S1(em1) - Ch(em1,em2,em3) - K[0]) & M
    Ce0 = (am4 - T2iv) & M

    rng = np.random.default_rng(seed)
    swept = 0; surv = 0; tri = 0; sub = 0; ver = 0; bad = 0
    hits = []
    left = N
    while left > 0:
        B = min(batch, left); left -= B
        A0 = rng.integers(0, 1<<32, size=B, dtype=np.uint64).astype(U32)
        swept += B
        E0 = (A0 + c(Ce0)); W0 = (A0 + c(C0c))
        G = (-(nS0(A0) + nMaj(A0,c(am1),c(am2))) - c(em3) - nS1(E0)
             - nCh(E0,c(em1),c(em2)) - c(K[1]))
        F0 = (c(K0p) - W0 - c(W9hat) - G)
        W1 = np.asarray(Tinv[F0]); ok = W1 != MISS
        A0,E0,W0,G,W1 = (x[ok] for x in (A0,E0,W0,G,W1)); surv += A0.size
        A1 = (W1 - G)
        E1 = (c(am3) + A1 - (nS0(A0) + nMaj(A0,c(am1),c(am2))))
        F12 = (-(nS0(A1) + nMaj(A1,A0,c(am1))) - c(em2) - nS1(E1)
               - nCh(E1,E0,c(em1)) - c(K[2]))
        R1 = (c(K1p) - W1 - F12 - c(W10base) + c(D))
        W2 = np.asarray(Tinv[R1]); ok = W2 != MISS
        A0,A1,E0,E1,W1,F12,W2 = (x[ok] for x in (A0,A1,E0,E1,W1,F12,W2))
        A2 = (W2 - F12)
        e2 = (c(am2) + A2 - (nS0(A1) + nMaj(A1,A0,c(am1))))
        F23 = (-(nS0(A2) + nMaj(A2,A1,A0)) - c(em1) - nS1(e2)
               - nCh(e2,E1,E0) - c(K[3]))
        R2 = (c(K2p) - c(W11base) + c(T1_7) + c(Ch(e10,e9,e8)) - W2 - F23)
        W3 = np.asarray(Tinv[R2]); ok = W3 != MISS
        A0,A1,A2,F23,W3 = (x[ok] for x in (A0,A1,A2,F23,W3))
        A3 = (W3 - F23)
        tri += A0.size
        if A0.size == 0: continue
        # general-v collapsed condition: Maj(v,a3,a2) == a3
        good = nMaj(c(v), A3, A2) == A3
        sub += int(good.sum())
        idx = np.nonzero(good)[0]
        for j in idx:                      # NO CAP
            aa = dict(a); aa.update({0:int(A0[j]),1:int(A1[j]),2:int(A2[j]),3:int(A3[j])})
            ee = dict(e)
            for rrr in range(0, R):
                ee[rrr] = (aa[rrr-4] + aa[rrr] - T2f(aa[rrr-1],aa[rrr-2],aa[rrr-3])) & M
            Wm = [recW(aa,ee,rrr) for rrr in range(16)]
            dg = b"".join(struct.pack(">I", x) for x in compress_R(Wm, R))
            if dg == h32:
                ver += 1
                if len(hits) < 4: hits.append(Wm)
            else:
                bad += 1
    return dict(swept=swept, surv=surv, tri=tri, sub=sub, ver=ver, bad=bad, hits=hits)

if __name__ == "__main__":
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    N  = int(sys.argv[2]) if len(sys.argv) > 2 else (1 << 22)
    mode = sys.argv[3] if len(sys.argv) > 3 else "urandom"
    tot = dict(swept=0,surv=0,tri=0,sub=0,ver=0,bad=0)
    allhits = []
    t0 = time.time()
    for i in range(NT):
        if mode == "urandom":
            h32 = os.urandom(32)
        elif mode == "ones":
            h32 = b"\xff"*32
        elif mode == "zeros":
            h32 = b"\x00"*32
        else:
            h32 = hashlib.sha256(mode.encode()+bytes([i])).digest()
        # context RNG seeded from system entropy, independent of the target
        rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
        v = int(rng.integers(0, 1<<32, dtype=np.uint64))   # random nonzero v
        ctx = make_ctx(rng, v)
        r = attack(h32, ctx, N, int.from_bytes(os.urandom(8),"little"))
        for k in tot: tot[k] += r[k]
        allhits += [(h32, w) for w in r['hits']]
        print(f"t{i} v=0x{v:08x} target={h32.hex()[:16]}... swept={r['swept']:,} "
              f"tri={r['tri']:,} sub={r['sub']} ver={r['ver']} bad={r['bad']}", flush=True)
    print("")
    print(f"TOTAL swept={tot['swept']:,}  surv={tot['surv']:,} ({tot['surv']/tot['swept']:.5f}/a0)")
    print(f"  tri={tot['tri']:,} ({tot['tri']/tot['swept']:.5f}/a0, ={tot['surv']/tot['swept']:.5f}^3 pred "
          f"{(tot['surv']/tot['swept'])**3:.5f})")
    print(f"  sub={tot['sub']:,} rate/tri={tot['sub']/max(tot['tri'],1):.4e} pred (3/4)^32={3**32/2**64:.6e}")
    print(f"  VERIFIED={tot['ver']:,}  FALSE(bad)={tot['bad']:,}")
    print(f"  per swept a0 = {tot['ver']/tot['swept']:.4e} = 2^{np.log2(max(tot['ver'],1)/tot['swept']):.3f}"
          f"  -> {tot['swept']/max(tot['ver'],1):,.0f} a0/preimage")
    print(f"  [{time.time()-t0:.0f}s]")
    import json
    with open("/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/refute_hits.json","w") as f:
        json.dump([[h.hex(), w] for h,w in allhits], f)
    print("wrote refute_hits.json", len(allhits))
