#!/usr/bin/env python3
"""E2/E3: rotational propagation through the MESSAGE SCHEDULE ALONE (no constants),
and through the full compression function with the constants removed / present.

A "rotational pair" is (W, W') with W'_i = W_i <<< g for i = 0..15.  We ask how
often the expanded words stay a rotational pair: W'_t == W_t <<< g for t >= 16.
Then the same for the state words a_r, e_r.
"""
import numpy as np

rng = np.random.default_rng(20260908)
M = np.uint32(0xFFFFFFFF)

K = [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
     0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
     0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
     0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
     0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
     0x06ca6351, 0x14292967]
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


def expand(W, upto):
    W = list(W)
    for t in range(16, upto):
        W.append((s1(W[t - 2]) + W[t - 7] + s0(W[t - 15]) + W[t - 16]) & M)
    return W


def compress(W, R, iv, use_K=True):
    a = {-1: iv[0], -2: iv[1], -3: iv[2], -4: iv[3]}
    e = {-1: iv[4], -2: iv[5], -3: iv[6], -4: iv[7]}
    for r in range(R):
        kk = np.uint32(K[r]) if use_K else np.uint32(0)
        T1 = (e[r - 4] + S1(e[r - 1]) + Ch(e[r - 1], e[r - 2], e[r - 3]) + kk + W[r]) & M
        a[r] = (T1 + S0(a[r - 1]) + Maj(a[r - 1], a[r - 2], a[r - 3])) & M
        e[r] = (a[r - 4] + T1) & M
    return a, e


N = 1 << 24
print(f"=== message schedule alone, N = {N:,} rotational pairs per g ===")
print("g   P[W16 ok]   P[W17|16]  P[W18|17]  P[W19|18]  P[W20|19]   P[W16..W19 all]")
res = {}
for g in (1, 2, 3, 8, 16, 30, 31):
    W = [rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32) for _ in range(16)]
    Wr = [rotl(x, g) for x in W]
    Wa = expand(W, 21)
    Wb = expand(Wr, 21)
    ok = np.ones(N, dtype=bool)
    conds = []
    for t in range(16, 21):
        ct = (Wb[t] == rotl(Wa[t], g))
        conds.append(ct)
    p16 = float(conds[0].mean())
    row = [p16]
    cum = conds[0]
    for t in range(1, 5):
        denom = int(cum.sum())
        num = int((cum & conds[t]).sum())
        row.append(num / denom if denom else float('nan'))
        cum = cum & conds[t]
    all19 = float((conds[0] & conds[1] & conds[2] & conds[3]).mean())
    res[g] = (row, all19)
    print(f"{g:<3d} " + "  ".join(f"{v:9.6f}" for v in row) + f"   {all19:.3e}")

print()
print("prediction per expanded word = P[sig0 rot] * P[sig1 rot] * P[4-term add rot]")
