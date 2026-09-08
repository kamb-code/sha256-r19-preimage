#!/usr/bin/env python3
"""E4: intrinsic rotational cost of the ROUND FUNCTION, with the constant
obstruction removed by fiat (rotate IV and K as well: the "rotated specification"
SHA-256^<<<g).  Whatever fails then is intrinsic: shifts and carries.

Three arms:
  A  rotate everything (IV<<<g, K<<<g, W<<<g)  -- constants no longer obstruct
  B  as A but K = 0                            -- no constants at all
  C  real IV and K, message rotated            -- the true function
"""
import numpy as np

rng = np.random.default_rng(777)
M = np.uint32(0xFFFFFFFF)

K = [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
     0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
     0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
     0x0fc19dc6, 0x240ca1cc]
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


def run(W16, R, cv, Ks):
    W = list(W16)
    for t in range(16, R):
        W.append((s1(W[t - 2]) + W[t - 7] + s0(W[t - 15]) + W[t - 16]) & M)
    a = {-1: cv[0], -2: cv[1], -3: cv[2], -4: cv[3]}
    e = {-1: cv[4], -2: cv[5], -3: cv[6], -4: cv[7]}
    for r in range(R):
        T1 = (e[r - 4] + S1(e[r - 1]) + Ch(e[r - 1], e[r - 2], e[r - 3]) + Ks[r] + W[r]) & M
        a[r] = (T1 + S0(a[r - 1]) + Maj(a[r - 1], a[r - 2], a[r - 3])) & M
        e[r] = (a[r - 4] + T1) & M
    return a, e, W


N = 1 << 21
R = 20
print(f"N = {N:,} rotational pairs, R = {R}")
print()
for arm, tag in (("A", "rotate the whole spec: IV<<<g, K<<<g, W<<<g"),
                 ("B", "K = 0, random chaining value (rotated), W<<<g"),
                 ("C", "real IV and K, only W rotated (the true function)")):
    print(f"--- arm {arm}: {tag} ---")
    for g in (1, 31, 2, 16):
        W = [rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32) for _ in range(16)]
        Wr = [rotl(x, g) for x in W]
        if arm == "B":
            cvA = [rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32)
                   for _ in range(8)]
            cvB = [rotl(x, g) for x in cvA]
            KA = [np.uint32(0)] * R
            KB = [np.uint32(0)] * R
        else:
            cvA = [np.full(N, np.uint32(v)) for v in IV]
            KA = [np.uint32(k) for k in K[:R]]
            if arm == "A":
                cvB = [rotl(x, g) for x in cvA]
                KB = [rotl(np.uint32(k), g) for k in K[:R]]
            else:
                cvB = [np.full(N, np.uint32(v)) for v in IV]
                KB = KA
        aA, eA, WA = run(W, R, cvA, KA)
        aB, eB, WB = run(Wr, R, cvB, KB)
        # schedule first
        sched = np.ones(N, dtype=bool)
        for t in range(16, R):
            sched &= (WB[t] == rotl(WA[t], g))
        alive = np.ones(N, dtype=bool)
        curve = []
        for r in range(R):
            alive &= (aB[r] == rotl(aA[r], g)) & (eB[r] == rotl(eA[r], g))
            curve.append(int(alive.sum()))
            if curve[-1] == 0:
                break
        joint = int((alive & sched).sum())
        pr = [f"{c}" for c in curve[:6]]
        print(f"  g={g:2d}: schedule W16..W19 ok {int(sched.sum()):,}/{N:,} "
              f"({sched.mean():.3e}); state alive after r=0,1,..: {pr} "
              f"-> extinct at round {len(curve)-1 if curve[-1]==0 else 'never<'+str(R)}")
    print()
