#!/usr/bin/env python3
"""Frame C at R=20, like-for-like with the paper's harness.

Calibration A (the paper's R=19 loop, frame A): for N random a0, compute a1
from C0 with the provisional W9hat (a2=a3=0), then iterate (a2,a3) from the
zero seed with a4 = its true (context) value, 8 iterations.  Paper: about
2.3e-5 of a0 values converge.

Experiment B (frame C): for N random (a0,a1) pairs, iterate (a2,a3,a4) from
the zero seed, with a4 absorbed by C0 through the P0 table, 16 iterations.
Fixed points are exact solutions of C0,C1,C2; C3 is then checked.

All constants are python ints reduced mod 2^32 before becoming uint32 arrays.
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
U32 = np.uint32
MISS = U32(M)

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
def T(u): return (s0(u) - u) & (MISS if isinstance(u, np.ndarray) else M)
def c(x): return U32(x & M)                      # python int -> uint32 scalar, safely

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
print("forward model matches hashlib", flush=True)

R = 20
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
rng = np.random.default_rng(seed)
N = 1 << 20

def plant(seed_rng):
    W = [int(x) for x in seed_rng.integers(0, 1 << 32, size=16, dtype=np.uint64)]
    a, e, _ = forward(W, 8)
    A8 = (S0(a[7]) + Maj(a[7], a[6], a[5])) & M
    W[8] = (((A8 - T2(a[7], a[6], a[5])) & M) - e[4] - S1(e[7]) - Ch(e[7], e[6], e[5]) - K[8]) & M
    a, e, Wf = forward(W, R)
    assert ((a[8] - S0(a[7]) - Maj(a[7], a[6], a[5])) & M) == 0
    for r in range(16, R): assert recW(a, e, r) == Wf[r]
    return W, a, e
W, a, e = plant(rng)

# ---- tables ----
t0 = time.time()
Tinv = np.load('/nvme0n1-disk/Kamvid/sigma0_u_table.npy').view(np.uint32)
print(f"sigma0 table loaded {time.time()-t0:.0f}s", flush=True)
P0PATH = '/nvme0n1-disk/Kamvid/p0_inv_c8_0.npy'
t0 = time.time()
if os.path.exists(P0PATH):
    P0inv = np.load(P0PATH); print(f"P0 table loaded {time.time()-t0:.0f}s", flush=True)
else:
    P0inv = np.full(1 << 32, M, dtype=np.uint32)
    for s in range(0, 1 << 32, 1 << 24):
        u = np.arange(s, s + (1 << 24), dtype=np.uint64).astype(np.uint32)
        P0inv[(S0(u) - S1(u)) & MISS] = u
    np.save(P0PATH, P0inv); print(f"P0 table built+saved {time.time()-t0:.0f}s", flush=True)

# ---- context constants (python ints) ----
am1, am2, am3, am4 = a[-1], a[-2], a[-3], a[-4]; em1, em2, em3, em4 = e[-1], e[-2], e[-3], e[-4]
a0t, a1t, a2t, a3t, a4t = a[0], a[1], a[2], a[3], a[4]
a5, a6, a7, a8, a9, a10, a11 = (a[i] for i in range(5, 12))
c8 = (a8 - S0(a7) - Maj(a7, a6, a5)) & M; assert c8 == 0
c7 = (a7 - S0(a6)) & M; c6 = (a6 - S0(a5)) & M
e9 = (a5 + a9 - T2(a8, a7, a6)) & M; e10 = (a6 + a10 - T2(a9, a8, a7)) & M; e11 = (a7 + a11 - T2(a10, a9, a8)) & M
W9base = ((a9 - T2(a8, a7, a6)) - K[9]) & M
W10base = ((a10 - T2(a9, a8, a7)) - S1(e9) - K[10]) & M
W11base = ((a11 - T2(a10, a9, a8)) - S1(e10) - K[11]) & M
W12base = ((a[12] - T2(a11, a10, a9)) - c8 - S1(e11) - Ch(e11, e10, e9) - K[12]) & M
Wr = {r: recW(a, e, r) for r in range(14, 20)}
K0p = (Wr[16] - s1(Wr[14])) & M; K1p = (Wr[17] - s1(Wr[15])) & M
K2p = (Wr[18] - s1(Wr[16])) & M; K3p = (Wr[19] - s1(Wr[17])) & M
T2iv = T2(am1, am2, am3)
C0c = (-T2iv - em4 - S1(em1) - Ch(em1, em2, em3) - K[0]) & M          # W0 = a0 + C0c
Ce0 = (am4 - T2iv) & M                                                 # e0 = a0 + Ce0

# ---- vectorised pieces over arrays A0, A1 (pairs) and candidates A2,A3,A4 ----
def pair_consts(A0, A1):
    """Everything that depends only on (a0,a1)."""
    E0 = (A0 + c(Ce0)) & MISS
    W0 = (A0 + c(C0c)) & MISS
    G = (-(S0(A0) + Maj(A0, c(am1), c(am2))) - c(em3) - S1(E0) - Ch(E0, c(em1), c(em2)) - c(K[1])) & MISS  # W1 - a1
    W1 = (A1 + G) & MISS
    E1 = (c(am3) + A1 - (S0(A0) + Maj(A0, c(am1), c(am2)))) & MISS
    F12 = (-(S0(A1) + Maj(A1, A0, c(am1))) - c(em2) - S1(E1) - Ch(E1, E0, c(em1)) - c(K[2])) & MISS       # W2 - a2
    return E0, W0, G, W1, E1, F12

def step(A0, A1, P, A2, A3, A4, use_c0):
    E0, W0, G, W1, E1, F12 = P
    e2 = (c(am2) + A2 - (S0(A1) + Maj(A1, A0, c(am1)))) & MISS
    e6 = (A2 + c(c6) - Maj(c(a5), A4, A3)) & MISS
    e7 = (A3 + c(c7) - Maj(c(a6), c(a5), A4)) & MISS
    e8 = (A4 + c(c8)) & MISS
    F23 = (-(S0(A2) + Maj(A2, A1, A0)) - c(em1) - S1(e2) - Ch(e2, E1, E0) - c(K[3])) & MISS   # W3 - a3
    W2 = (A2 + F12) & MISS
    R2 = (c(K2p) - W2 - F23 - c(W11base) + c(c7) - Maj(c(a6), c(a5), A4) + Ch(c(e10), c(e9), e8)) & MISS
    W3 = Tinv[R2]; ok = W3 != MISS
    A3n = (W3 - F23) & MISS
    if use_c0:
        W9t = (c(K0p) - s0(W1) - W0) & MISS
        R0 = (W9t - c(W9base) + A1 + c(a5) - Maj(A4, A3, A2) + Ch(e8, e7, e6)) & MISS
        A4n = P0inv[R0]; ok &= A4n != MISS
    else:
        A4n = A4
    R1 = (c(K1p) - W1 - F12 - c(W10base) + c(c6) - Maj(c(a5), A4, A3) + Ch(c(e9), e8, e7)) & MISS
    W2n = Tinv[R1]; ok &= W2n != MISS
    A2n = (W2n - F12) & MISS
    return A3n, A4n, A2n, ok

def c3_res(A0, A1, P, A2, A3, A4):
    E0, W0, G, W1, E1, F12 = P
    e2 = (c(am2) + A2 - (S0(A1) + Maj(A1, A0, c(am1)))) & MISS
    e3 = (c(am1) + A3 - (S0(A2) + Maj(A2, A1, A0))) & MISS
    W3 = (A3 - (S0(A2) + Maj(A2, A1, A0)) - c(em1) - S1(e2) - Ch(e2, E1, E0) - c(K[3])) & MISS
    W4 = (A4 - (S0(A3) + Maj(A3, A2, A1)) - E0 - S1(e3) - Ch(e3, e2, E1) - c(K[4])) & MISS
    return (s0(W4) + W3 + (c(W12base) - A4) - c(K3p)) & MISS

# ---- exactness at the planted point, with python-int algebra (no table) ----
A0 = np.array([a0t], U32); A1 = np.array([a1t], U32); P = pair_consts(A0, A1)
A2 = np.array([a2t], U32); A3 = np.array([a3t], U32); A4 = np.array([a4t], U32)
E0, W0, G, W1, E1, F12 = P
assert int(W1[0]) == recW(a, e, 1) and int(W0[0]) == recW(a, e, 0) and int(E1[0]) == e[1]
assert int((A2 + F12)[0]) == recW(a, e, 2)
A3n, A4n, A2n, ok = step(A0, A1, P, A2, A3, A4, True)
# check the targets algebraically: T(true W) must equal R
Wt2, Wt3 = recW(a, e, 2), recW(a, e, 3)
print("C3 residual at true point:", hex(int(c3_res(A0, A1, P, A2, A3, A4)[0])))
print("true point maps: a3 %s, a4 %s, a2 %s (table representatives)" % (
      "OK" if A3n[0] == a3t else "rep-diff", "OK" if A4n[0] == a4t else "rep-diff", "OK" if A2n[0] == a2t else "rep-diff"), flush=True)

# ---- Calibration A: the paper's R=19 loop (frame A), per random a0, seed (0,0), 8 iterations ----
def calibration_A(N, iters=8, seed=11):
    r = np.random.default_rng(seed)
    A0 = r.integers(0, 1 << 32, size=N, dtype=np.uint64).astype(U32)
    A4 = np.full(N, a4t, U32)
    # provisional W9hat with a2=a3=0 (and a4 true)
    e6h = (c(c6) - Maj(c(a5), A4, U32(0))) & MISS
    e7h = (c(c7) - Maj(c(a6), c(a5), A4)) & MISS
    e8 = (A4 + c(c8)) & MISS
    T15h = (c(a5) - S0(A4) - Maj(A4, U32(0), U32(0))) & MISS
    W9hat = (c(W9base) - T15h - S1(e8) - Ch(e8, e7h, e6h)) & MISS
    E0 = (A0 + c(Ce0)) & MISS; W0 = (A0 + c(C0c)) & MISS
    G = (-(S0(A0) + Maj(A0, c(am1), c(am2))) - c(em3) - S1(E0) - Ch(E0, c(em1), c(em2)) - c(K[1])) & MISS
    F0 = (c(K0p) - W0 - W9hat - G) & MISS            # T(W1) = K0' - W0 - W9hat - g(a0)
    W1 = Tinv[F0]; ok0 = W1 != MISS
    A1 = (W1 - G) & MISS
    P = pair_consts(A0, A1)
    A2 = np.zeros(N, U32); A3 = np.zeros(N, U32)
    conv = np.zeros(N, bool)
    for it in range(iters):
        A3n, _, _, ok = step(A0, A1, P, A2, A3, A4, False)
        A3 = np.where(ok, A3n, A3)
        _, _, A2n, ok2 = step(A0, A1, P, A2, A3, A4, False)
        A3c, _, A2c, okc = step(A0, A1, P, A2n, A3, A4, False)
        fixed = ok0 & ok2 & okc & (A2c == A2n) & (A3c == A3)
        conv |= fixed
        A2 = np.where(ok2, A2n, A2)
    return conv.sum(), ok0.sum()

t0 = time.time(); conv, ok0 = calibration_A(N)
print(f"Calibration A (paper's R19 loop, frame A): {N:,} random a0, C0 survivors {ok0:,}, "
      f"converged {conv:,} = {conv/N:.2e} per a0 (paper: ~2.3e-5 per survivor)  [{time.time()-t0:.0f}s]", flush=True)

# ---- Experiment B: frame C, random (a0,a1) pairs, seed (0,0,0), 16 iterations ----
def experiment_B(N, iters=16, seed=12, a4_seed=None):
    r = np.random.default_rng(seed)
    A0 = r.integers(0, 1 << 32, size=N, dtype=np.uint64).astype(U32)
    A1 = r.integers(0, 1 << 32, size=N, dtype=np.uint64).astype(U32)
    P = pair_consts(A0, A1)
    A2 = np.zeros(N, U32); A3 = np.zeros(N, U32)
    A4 = np.zeros(N, U32) if a4_seed is None else np.full(N, a4_seed, U32)
    conv = np.zeros(N, bool); c3pass = np.zeros(N, bool)
    for it in range(iters):
        A3n, _, _, ok = step(A0, A1, P, A2, A3, A4, True); A3 = np.where(ok, A3n, A3)
        _, A4n, _, ok = step(A0, A1, P, A2, A3, A4, True); A4 = np.where(ok, A4n, A4)
        _, _, A2n, ok = step(A0, A1, P, A2, A3, A4, True); A2 = np.where(ok, A2n, A2)
        A3c, A4c, A2c, okc = step(A0, A1, P, A2, A3, A4, True)
        fixed = okc & (A3c == A3) & (A4c == A4) & (A2c == A2)
        conv |= fixed
        c3pass |= fixed & (c3_res(A0, A1, P, A2, A3, A4) == 0)
    return conv.sum(), c3pass.sum()

t0 = time.time(); conv, c3 = experiment_B(N)
print(f"Experiment B (frame C, per random (a0,a1) pair, seed 0): {N:,} pairs, "
      f"converged {conv:,} = {conv/N:.2e} per pair, C3 passes {c3}  [{time.time()-t0:.0f}s]", flush=True)

# B2: the true pair, many seeds (basin size of the planted point / any fixed point)
def basin(N, iters=16, seed=13):
    r = np.random.default_rng(seed)
    A0 = np.full(N, a0t, U32); A1 = np.full(N, a1t, U32); P = pair_consts(A0, A1)
    A2 = r.integers(0, 1 << 32, N, dtype=np.uint64).astype(U32)
    A3 = r.integers(0, 1 << 32, N, dtype=np.uint64).astype(U32)
    A4 = r.integers(0, 1 << 32, N, dtype=np.uint64).astype(U32)
    conv = np.zeros(N, bool); hit = np.zeros(N, bool)
    for it in range(iters):
        A3n, _, _, ok = step(A0, A1, P, A2, A3, A4, True); A3 = np.where(ok, A3n, A3)
        _, A4n, _, ok = step(A0, A1, P, A2, A3, A4, True); A4 = np.where(ok, A4n, A4)
        _, _, A2n, ok = step(A0, A1, P, A2, A3, A4, True); A2 = np.where(ok, A2n, A2)
        A3c, A4c, A2c, okc = step(A0, A1, P, A2, A3, A4, True)
        fixed = okc & (A3c == A3) & (A4c == A4) & (A2c == A2); conv |= fixed
        hit |= fixed & (A2 == a2t) & (A3 == a3t) & (A4 == a4t)
    return conv.sum(), hit.sum()
t0 = time.time(); conv, hit = basin(N)
print(f"B2 (true pair, {N:,} random seeds): fixed points {conv:,}, true point {hit:,}  [{time.time()-t0:.0f}s]", flush=True)
print("DONE", flush=True)
