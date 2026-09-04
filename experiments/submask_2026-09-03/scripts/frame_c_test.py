#!/usr/bin/env python3
"""Frame C test at R=20: sweep (a0,a1), absorb a4 by C0 through a second table
P0(x) = Sigma0(x) - Sigma1(x + c8), absorb a2 by C1 and a3 by C2 (global
sigma0(u)-u table), C3 as the final check.  Measures the convergence of the
inner (a2,a3,a4) loop on a PLANTED solution, against the R=19-style (a2,a3)
loop on the same instance as calibration.

Everything is self-contained (own SHA-256 in the a-representation, checked
against hashlib).  Needs ~34 GB RAM: the published sigma0 table (16 GB) plus
the P0 table built here (16 GB).
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

# ---- scalar / vector primitives (work on python ints and uint32 numpy arrays) ----
def rotr(x, n):
    if isinstance(x, np.ndarray):
        return ((x >> np.uint32(n)) | (x << np.uint32(32 - n))) & np.uint32(M)
    return ((x >> n) | (x << (32 - n))) & M
def S0(x): return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
def S1(x): return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
def s0(x): return rotr(x, 7) ^ rotr(x, 18) ^ (x >> (np.uint32(3) if isinstance(x, np.ndarray) else 3))
def s1(x): return rotr(x, 17) ^ rotr(x, 19) ^ (x >> (np.uint32(10) if isinstance(x, np.ndarray) else 10))
def Ch(e, f, g): return (e & f) ^ (~e & g) if isinstance(e, np.ndarray) else ((e & f) ^ ((~e) & g)) & M
def Maj(a, b, c): return (a & b) ^ (a & c) ^ (b & c)
def T2(a, b, c): return (S0(a) + Maj(a, b, c)) & (np.uint32(M) if isinstance(a, np.ndarray) else M)
def T(u): return (s0(u) - u) & (np.uint32(M) if isinstance(u, np.ndarray) else M)

# ---- forward SHA-256 in the a-representation (paper indexing) ----
def forward(W, R):
    """W: 16 words.  Returns dicts a[r], e[r] for r in -4..R-1 (a_{-1}=IV[0], e_{-1}=IV[4])."""
    a = {-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]}
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    Wf = list(W)
    for t in range(16, R):
        Wf.append((s1(Wf[t-2]) + Wf[t-7] + s0(Wf[t-15]) + Wf[t-16]) & M)
    for r in range(R):
        T1 = (e[r-4] + S1(e[r-1]) + Ch(e[r-1], e[r-2], e[r-3]) + K[r] + Wf[r]) & M
        a[r] = (T1 + T2(a[r-1], a[r-2], a[r-3])) & M
        e[r] = (a[r-4] + T1) & M
    return a, e, Wf

def digest(W, R):
    a, e, _ = forward(W, R)
    s = [a[R-1], a[R-2], a[R-3], a[R-4], e[R-1], e[R-2], e[R-3], e[R-4]]
    return b"".join(struct.pack(">I", (x + y) & M) for x, y in zip(s, IV))

def recW(a, e, r):
    return (a[r] - T2(a[r-1], a[r-2], a[r-3]) - e[r-4] - S1(e[r-1]) - Ch(e[r-1], e[r-2], e[r-3]) - K[r]) & M

# ---- self-check against hashlib ----
blk = os.urandom(55); pad = blk + b"\x80" + b"\x00"*(56-1-55) + struct.pack(">Q", 55*8)
W64 = [struct.unpack(">I", pad[4*i:4*i+4])[0] for i in range(16)]
assert digest(W64, 64) == hashlib.sha256(blk).digest(), "forward model wrong"
print("forward model matches hashlib at 64 rounds", flush=True)

R = 20
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 7)

def plant(seed_rng):
    """Random block with W8 chosen so that c8 = a8 - S0(a7) - Maj(a7,a6,a5) = 0."""
    W = [int(x) for x in seed_rng.integers(0, 1 << 32, size=16, dtype=np.uint64)]
    a, e, _ = forward(W, 8)                   # states up to a7,e7
    A8 = (S0(a[7]) + Maj(a[7], a[6], a[5])) & M            # wanted a8
    T1_8 = (A8 - T2(a[7], a[6], a[5])) & M
    W[8] = (T1_8 - e[4] - S1(e[7]) - Ch(e[7], e[6], e[5]) - K[8]) & M
    a, e, Wf = forward(W, R)
    assert ((a[8] - S0(a[7]) - Maj(a[7], a[6], a[5])) & M) == 0
    for r in range(16, R):                    # recovered words equal the schedule
        assert recW(a, e, r) == Wf[r], r
    for r in range(16):
        assert recW(a, e, r) == W[r], r
    return W, a, e

W, a, e = plant(rng)
print("planted instance: schedule-consistent, c8 = 0", flush=True)

# ---- tables ----
t0 = time.time()
Tinv = np.load('/nvme0n1-disk/Kamvid/sigma0_u_table.npy').view(np.uint32)     # sentinel 0xFFFFFFFF
print(f"sigma0 table loaded in {time.time()-t0:.0f}s", flush=True)
t0 = time.time()
P0inv = np.full(1 << 32, M, dtype=np.uint32)
for s in range(0, 1 << 32, 1 << 24):
    u = np.arange(s, s + (1 << 24), dtype=np.uint64).astype(np.uint32)
    v = (S0(u) - S1(u)) & np.uint32(M)        # c8 = 0
    P0inv[v] = u
print(f"P0 table built in {time.time()-t0:.0f}s; coverage {(P0inv != np.uint32(M)).mean()*100:.2f}%", flush=True)
MISS = np.uint32(M)

# ---- context constants (paper indexing; context a5..a11, target-fixed a12..a19) ----
am1, am2, am3, am4 = a[-1], a[-2], a[-3], a[-4]; em1, em2, em3, em4 = e[-1], e[-2], e[-3], e[-4]
a0, a1, a2t, a3t, a4t = a[0], a[1], a[2], a[3], a[4]
a5, a6, a7, a8, a9, a10, a11 = (a[i] for i in range(5, 12))
e0 = (am4 + a0 - T2(am1, am2, am3)) & M
e1 = (am3 + a1 - T2(a0, am1, am2)) & M
W0 = recW(a, e, 0); W1 = recW(a, e, 1)
Wr = {r: recW(a, e, r) for r in range(12, 20)}   # W12..W19 need a4.. (W12 needs a4; handle below)
# context-only pieces
c8 = (a8 - S0(a7) - Maj(a7, a6, a5)) & M; assert c8 == 0
c7 = (a7 - S0(a6)) & M; c6 = (a6 - S0(a5)) & M
e9 = (a5 + a9 - T2(a8, a7, a6)) & M; e10 = (a6 + a10 - T2(a9, a8, a7)) & M; e11 = (a7 + a11 - T2(a10, a9, a8)) & M
T1_9 = (a9 - T2(a8, a7, a6)) & M; T1_10 = (a10 - T2(a9, a8, a7)) & M; T1_11 = (a11 - T2(a10, a9, a8)) & M
T1_12 = (a[12] - T2(a11, a10, a9)) & M
W9base = (T1_9 - K[9]) & M
W10base = (T1_10 - S1(e9) - K[10]) & M
W11base = (T1_11 - S1(e10) - K[11]) & M
W12base = (T1_12 - c8 - S1(e11) - Ch(e11, e10, e9) - K[12]) & M   # W12 = W12base - a4
K0p = (Wr[16] - s1(Wr[14])) & M; K1p = (Wr[17] - s1(Wr[15])) & M
K2p = (Wr[18] - s1(Wr[16])) & M; K3p = (Wr[19] - s1(Wr[17])) & M
assert Wr[12] == (W12base - a4t) & M
F12 = (recW(a, e, 2) - a2t) & M              # depends on a0,a1 only
W9target = (K0p - s0(W1) - W0) & M
assert recW(a, e, 9) == W9target             # C0 holds at the true point

def maps(A2, A3, A4):
    """Vectorised absorber maps.  Returns (a3_new, a4_new, a2_new) computed in that order,
    each from the CURRENT other values (Jacobi-style), plus the C3 residual."""
    A2 = A2.astype(np.uint32); A3 = A3.astype(np.uint32); A4 = A4.astype(np.uint32)
    e2 = (np.uint32(am2) + A2 - np.uint32(T2(a1, a0, am1))) & MISS
    e6 = (A2 + np.uint32(c6) - Maj(np.uint32(a5), A4, A3)) & MISS
    e7 = (A3 + np.uint32(c7) - Maj(np.uint32(a6), np.uint32(a5), A4)) & MISS
    e8 = (A4 + np.uint32(c8)) & MISS
    F23 = (- (S0(A2) + Maj(A2, np.uint32(a1), np.uint32(a0))) - np.uint32(em1)
           - S1(e2) - Ch(e2, np.uint32(e1), np.uint32(e0)) - np.uint32(K[3])) & MISS   # W3 - a3
    W2 = (A2 + np.uint32(F12)) & MISS
    # C2 -> a3
    R2 = (np.uint32(K2p) - W2 - F23 - np.uint32(W11base) + np.uint32(c7)
          - Maj(np.uint32(a6), np.uint32(a5), A4) + Ch(np.uint32(e10), np.uint32(e9), e8)) & MISS
    W3 = Tinv[R2]; ok3 = W3 != MISS
    A3n = (W3 - F23) & MISS
    # C0 -> a4  (P0(a4) = S0(a4) - S1(a4 + c8))
    R0 = (np.uint32(W9target) - np.uint32(W9base) + np.uint32(a1) + np.uint32(a5)
          - Maj(A4, A3, A2) + Ch(e8, e7, e6)) & MISS
    A4n = P0inv[R0]; ok4 = A4n != MISS
    # C1 -> a2
    R1 = (np.uint32(K1p) - np.uint32(W1) - np.uint32(F12) - np.uint32(W10base) + np.uint32(c6)
          - Maj(np.uint32(a5), A4, A3) + Ch(np.uint32(e9), e8, e7)) & MISS
    W2n = Tinv[R1]; ok2 = W2n != MISS
    A2n = (W2n - np.uint32(F12)) & MISS
    return A3n, A4n, A2n, ok3 & ok4 & ok2

def c3_residual(A2, A3, A4):
    A2 = A2.astype(np.uint32); A3 = A3.astype(np.uint32); A4 = A4.astype(np.uint32)
    e2 = (np.uint32(am2) + A2 - np.uint32(T2(a1, a0, am1))) & MISS
    e3 = (np.uint32(am1) + A3 - (S0(A2) + Maj(A2, np.uint32(a1), np.uint32(a0)))) & MISS
    W3 = (A3 - (S0(A2) + Maj(A2, np.uint32(a1), np.uint32(a0))) - np.uint32(em1) - S1(e2)
          - Ch(e2, np.uint32(e1), np.uint32(e0)) - np.uint32(K[3])) & MISS
    W4 = (A4 - (S0(A3) + Maj(A3, A2, np.uint32(a1))) - np.uint32(e0) - S1(e3)
          - Ch(e3, e2, np.uint32(e1)) - np.uint32(K[4])) & MISS
    W12 = (np.uint32(W12base) - A4) & MISS
    return (s0(W4) + W3 + W12 - np.uint32(K3p)) & MISS

# ---- exactness checks at the planted point ----
tA2 = np.array([a2t], dtype=np.uint32); tA3 = np.array([a3t], dtype=np.uint32); tA4 = np.array([a4t], dtype=np.uint32)
A3n, A4n, A2n, ok = maps(tA2, tA3, tA4)
print("true point through the maps (table-based): a3 %s a4 %s a2 %s  (lookups ok=%s)" % (
    "OK" if A3n[0] == a3t else "REP-DIFF", "OK" if A4n[0] == a4t else "REP-DIFF", "OK" if A2n[0] == a2t else "REP-DIFF", bool(ok[0])), flush=True)
# algebraic exactness independent of table representatives: T(true W) == R
print("C3 residual at true point:", hex(int(c3_residual(tA2, tA3, tA4)[0])), flush=True)

# ---- convergence experiments ----
def run_loop(N, iters, mode, seed=1):
    r = np.random.default_rng(seed)
    A2 = r.integers(0, 1 << 32, size=N, dtype=np.uint64).astype(np.uint32)
    A3 = r.integers(0, 1 << 32, size=N, dtype=np.uint64).astype(np.uint32)
    A4 = (r.integers(0, 1 << 32, size=N, dtype=np.uint64).astype(np.uint32) if mode == "C"
          else np.full(N, a4t, dtype=np.uint32))
    converged = np.zeros(N, dtype=bool); true_hit = np.zeros(N, dtype=bool)
    for it in range(iters):
        A3n, A4n, A2n, ok = maps(A2, A3, A4)
        if mode == "R19":                      # a4 fixed at its true value, C0 not used
            A4n = A4
        fixed = ok & (A3n == A3) & (A4n == A4) & (A2n == A2)
        converged |= fixed
        true_hit |= fixed & (A2 == a2t) & (A3 == a3t) & (A4 == a4t)
        # Gauss-Seidel style update: a3 first, then a4 from new a3, then a2 from new (a3,a4)
        A3 = np.where(ok, A3n, A3)
        if mode == "C":
            _, A4n2, _, ok2 = maps(A2, A3, A4)
            A4 = np.where(ok2, A4n2, A4)
        _, _, A2n2, ok3 = maps(A2, A3, A4)
        A2 = np.where(ok3, A2n2, A2)
    return converged.sum(), true_hit.sum()

N = 1 << 20
for mode in ("R19", "C"):
    t0 = time.time()
    conv, true_hit = run_loop(N, 16, mode)
    print(f"mode {mode:>3}: {N:,} random seeds, 16 iterations: fixed points {conv:,} "
          f"({conv/N:.2e} per seed), true point {true_hit:,}   [{time.time()-t0:.0f}s]", flush=True)

# basin probe: seeds near the true point (a2,a3 random, a4 true / a4 near true)
r = np.random.default_rng(3)
for label, A2, A3, A4 in (
    ("a4 true, a2 a3 random", r.integers(0,1<<32,N,dtype=np.uint64).astype(np.uint32), r.integers(0,1<<32,N,dtype=np.uint64).astype(np.uint32), np.full(N, a4t, np.uint32)),
    ("a2 true, a3 a4 random", np.full(N, a2t, np.uint32), r.integers(0,1<<32,N,dtype=np.uint64).astype(np.uint32), r.integers(0,1<<32,N,dtype=np.uint64).astype(np.uint32)),
    ("a2 a3 true, a4 random", np.full(N, a2t, np.uint32), np.full(N, a3t, np.uint32), r.integers(0,1<<32,N,dtype=np.uint64).astype(np.uint32)),
):
    conv = np.zeros(N, bool); hit = np.zeros(N, bool)
    for it in range(16):
        A3n, A4n, A2n, ok = maps(A2, A3, A4)
        fixed = ok & (A3n == A3) & (A4n == A4) & (A2n == A2); conv |= fixed
        hit |= fixed & (A2 == a2t) & (A3 == a3t) & (A4 == a4t)
        A3 = np.where(ok, A3n, A3); _, A4n2, _, ok2 = maps(A2, A3, A4); A4 = np.where(ok2, A4n2, A4)
        _, _, A2n2, ok3 = maps(A2, A3, A4); A2 = np.where(ok3, A2n2, A2)
    print(f"basin probe [{label}]: fixed points {conv.sum():,} true {hit.sum():,} of {N:,}", flush=True)
print("DONE", flush=True)
