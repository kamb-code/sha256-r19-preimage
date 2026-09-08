#!/usr/bin/env python3
"""Exact R=20 constraint algebra, full width and reduced width, vectorised.

Residuals c0..c3 of the four schedule constraints as functions of the four
unknowns (a0,a1,a2,a3), with the context a4..a11 and the digest chain
a12..a19 fixed.  Matches code/submask_family.py exactly; sanity-checked by
planting a real message.
"""
from __future__ import annotations
import struct
import numpy as np

K32 = [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
       0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
       0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
       0x0fc19dc6, 0x240ca1cc]
IV32 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
R = 20


def scaled(w):
    def r(x):
        return int(np.floor(x * w / 32 + 0.5))
    return ((r(2), r(13), r(22)), (r(6), r(11), r(25)),
            (r(7), r(18), max(r(3), 1)), (r(17), r(19), max(r(10), 1)))


class Alg:
    """w-bit model (w=32 is SHA-256 itself).  All ops on python ints or
    numpy uint64 arrays reduced mod 2^w."""

    def __init__(self, w=32):
        self.w = w
        self.M = (1 << w) - 1
        self.S0r, self.S1r, self.s0r, self.s1r = scaled(w)
        self.K = [k & self.M for k in K32]
        self.IV = [x & self.M for x in IV32]

    # ---- primitives -----------------------------------------------------
    def rotr(self, x, n):
        w, M = self.w, self.M
        n %= w
        if n == 0:
            return x & M
        return ((x >> n) | (x << (w - n))) & M

    def S0(self, x):
        a, b, c = self.S0r
        return self.rotr(x, a) ^ self.rotr(x, b) ^ self.rotr(x, c)

    def S1(self, x):
        a, b, c = self.S1r
        return self.rotr(x, a) ^ self.rotr(x, b) ^ self.rotr(x, c)

    def s0(self, x):
        a, b, c = self.s0r
        return self.rotr(x, a) ^ self.rotr(x, b) ^ ((x & self.M) >> c)

    def s1(self, x):
        a, b, c = self.s1r
        return self.rotr(x, a) ^ self.rotr(x, b) ^ ((x & self.M) >> c)

    def Ch(self, e, f, g):
        return ((e & f) ^ ((~e) & g)) & self.M

    def Maj(self, a, b, c):
        return ((a & b) ^ (a & c) ^ (b & c)) & self.M

    def T2(self, a, b, c):
        return (self.S0(a) + self.Maj(a, b, c)) & self.M

    # ---- reference compression -----------------------------------------
    def forward(self, W, rounds=R):
        M = self.M
        a = {-1: self.IV[0], -2: self.IV[1], -3: self.IV[2], -4: self.IV[3]}
        e = {-1: self.IV[4], -2: self.IV[5], -3: self.IV[6], -4: self.IV[7]}
        Wf = list(W)
        for t in range(16, rounds):
            Wf.append((self.s1(Wf[t - 2]) + Wf[t - 7] + self.s0(Wf[t - 15]) + Wf[t - 16]) & M)
        for r in range(rounds):
            T1 = (e[r - 4] + self.S1(e[r - 1]) + self.Ch(e[r - 1], e[r - 2], e[r - 3])
                  + self.K[r] + Wf[r]) & M
            a[r] = (T1 + self.T2(a[r - 1], a[r - 2], a[r - 3])) & M
            e[r] = (a[r - 4] + T1) & M
        return a, e, Wf

    def family(self, v, a6, a7, a10, a11):
        M = self.M
        c = {4: v & M, 5: v & M, 6: a6 & M, 7: a7 & M}
        c[8] = (M - (v & M) + self.S0(c[7]) + self.Maj(c[7], c[6], c[5])) & M
        c[9] = (M - (v & M) + self.S0(c[8]) + self.Maj(c[8], c[7], c[6])) & M
        c[10], c[11] = a10 & M, a11 & M
        return c


class Instance:
    """One (digest chain, context) pair.  residuals(a0,a1,a2,a3) -> c0..c3."""

    def __init__(self, alg: Alg, chain: dict, ctx: dict):
        self.A = alg
        self.M = alg.M
        self.chain = dict(chain)          # a12..a19
        self.ctx = dict(ctx)              # a4..a11
        a = {-1: alg.IV[0], -2: alg.IV[1], -3: alg.IV[2], -4: alg.IV[3]}
        a.update(ctx)
        a.update(chain)
        self.a = a
        M = self.M
        e = {-1: alg.IV[4], -2: alg.IV[5], -3: alg.IV[6], -4: alg.IV[7]}
        for r in range(8, R):
            e[r] = (a[r - 4] + a[r] - alg.T2(a[r - 1], a[r - 2], a[r - 3])) & M
        self.e_hi = e
        # constant message words W12..W19
        self.Wc = {}
        for r in range(12, R):
            self.Wc[r] = (a[r] - alg.T2(a[r - 1], a[r - 2], a[r - 3]) - e[r - 4]
                          - alg.S1(e[r - 1]) - alg.Ch(e[r - 1], e[r - 2], e[r - 3])
                          - alg.K[r]) & M

    def state(self, a0, a1, a2, a3):
        """e0..e7 and W0..W11 as arrays/ints."""
        A, M = self.A, self.M
        a = self.a
        vec = any(isinstance(x, np.ndarray) for x in (a0, a1, a2, a3))
        cast = (lambda x: np.asarray(x, dtype=np.uint64)) if vec else (lambda x: x)
        KK = [cast(k) for k in A.K]
        am1, am2, am3, am4 = (cast(a[-1]), cast(a[-2]), cast(a[-3]), cast(a[-4]))
        em1, em2, em3, em4 = (cast(A.IV[4]), cast(A.IV[5]), cast(A.IV[6]), cast(A.IV[7]))
        e = {-1: em1, -2: em2, -3: em3, -4: em4}
        av = {0: cast(a0) & M, 1: cast(a1) & M, 2: cast(a2) & M, 3: cast(a3) & M}
        for r in range(4, R):
            av[r] = cast(a[r])
        av[-1], av[-2], av[-3], av[-4] = am1, am2, am3, am4
        for r in range(0, 8):
            e[r] = (av[r - 4] + av[r] - A.T2(av[r - 1], av[r - 2], av[r - 3])) & M
        for r in range(8, R):
            e[r] = cast(self.e_hi[r])
        W = {}
        for r in range(0, 12):
            W[r] = (av[r] - A.T2(av[r - 1], av[r - 2], av[r - 3]) - e[r - 4]
                    - A.S1(e[r - 1]) - A.Ch(e[r - 1], e[r - 2], e[r - 3]) - KK[r]) & M
        for r in range(12, R):
            W[r] = cast(self.Wc[r])
        return av, e, W

    def residuals(self, a0, a1, a2, a3):
        A, M = self.A, self.M
        _, _, W = self.state(a0, a1, a2, a3)
        out = []
        for j in range(4):
            out.append((W[16 + j] - A.s1(W[14 + j]) - W[9 + j] - A.s0(W[1 + j]) - W[j]) & M)
        return out

    def c3_of(self, a0, a1, a2, a3):
        return self.residuals(a0, a1, a2, a3)[3]


def plant(alg: Alg, rng):
    """A random 20-round instance from a planted message; returns
    (chain, ctx_true, a0..a3 true)."""
    M = alg.M
    W = [int(rng.integers(0, 1 << alg.w)) for _ in range(16)]
    a, e, Wf = alg.forward(W, R)
    chain = {i: a[i] for i in range(12, R)}
    ctx = {i: a[i] for i in range(4, 12)}
    return chain, ctx, [a[i] for i in range(4)], W


def family_instance(alg: Alg, rng, v=None):
    """A random target digest chain plus a random submask-family context."""
    M = alg.M
    W = [int(rng.integers(0, 1 << alg.w)) for _ in range(16)]
    a, e, Wf = alg.forward(W, R)
    chain = {i: a[i] for i in range(12, R)}
    rv = lambda: int(rng.integers(0, 1 << alg.w))
    if v is None:
        v = rv()
    ctx = alg.family(v, rv(), rv(), rv(), rv())
    return Instance(alg, chain, ctx), v


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    for w in (32, 12, 8):
        alg = Alg(w)
        bad = 0
        for _ in range(200):
            chain, ctx, a03, W = plant(alg, rng)
            inst = Instance(alg, chain, ctx)
            r = inst.residuals(*a03)
            if any(x != 0 for x in r):
                bad += 1
        print(f"w={w:2d}: planted-message residual check {200-bad}/200",
              "OK" if bad == 0 else "FAIL")
    # vectorised check
    alg = Alg(32)
    chain, ctx, a03, W = plant(alg, rng)
    inst = Instance(alg, chain, ctx)
    n = 1000
    A0 = np.full(n, a03[0], dtype=np.uint64)
    A1 = np.full(n, a03[1], dtype=np.uint64)
    A2 = np.full(n, a03[2], dtype=np.uint64)
    A3 = np.full(n, a03[3], dtype=np.uint64)
    rs = inst.residuals(A0, A1, A2, A3)
    print("vectorised at the true point:", [int(x[0]) for x in rs])
