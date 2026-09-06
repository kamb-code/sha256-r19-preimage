#!/usr/bin/env python3
"""w-bit SHA-256 analogue (R = 20) for edge-weight measurements, vectorised.

Rotation/shift amounts scaled by w/32 (round half up) exactly as in
experiments/.../unexamined/model_w.py; at w = 32 the model IS SHA-256 (checked
against code/submask_family.py in selftest()).

Context language (extends elimination_search/a10/numcheck.py `realise`):
    ('free',)            a_i is a free context word (random per state, or fixed by `free` dict)
    ('const', c)         a_i = c
    ('eq', j)            a_i = a_j          (j in 4..19; j may be a digest word)
    ('neq', j)           a_i = ~a_j
    ('rot', j, n)        a_i = rotr(a_j, n)           (rotational tie, new)
    ('rotx', j, n, c)    a_i = rotr(a_j, n) ^ c       (rotational tie with mask, new)
    ('add', j, c)        a_i = a_j + c                 (additive tie, new)
    ('sat_r', t)         e_i = t, solved through a_i           (i >= 8)
    ('sat_h', t)         e_{i+4} = t, solved through a_i       (i+4 <= 19)
Every value is a python int or an int64 numpy array (vectorised over states).
The unknown (a5 here) is an array; context words derived from it become arrays.

Edge weight of unknown u into constraint target T_j:
    T_j = W_{16+j} - s1(W_{14+j}) - W_{9+j} - W_j        (numcheck.py's T_j)
    = the s0-table lookup key up to the u-free word W_{j+1}, so dT_j = dkey.
For each of N random states and each bit b of u: Hamming weight of
T_j(u ^ (1<<b)) - ... XOR ... T_j(u); reported as mean bits per flipped bit.
"""
from __future__ import annotations
import numpy as np

R = 20
K32 = [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
       0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
       0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
       0x0fc19dc6, 0x240ca1cc]
IV32 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]


def scaled(w):
    def r(x):
        return int(np.floor(x * w / 32 + 0.5))
    S0r = (r(2), r(13), r(22)); S1r = (r(6), r(11), r(25))
    s0r = (r(7), r(18), max(r(3), 1)); s1r = (r(17), r(19), max(r(10), 1))
    return S0r, S1r, s0r, s1r


class Model:
    def __init__(self, w):
        self.w = w
        self.M = (1 << w) - 1
        self.S0r, self.S1r, self.s0r, self.s1r = scaled(w)
        self.K = [k & self.M for k in K32]
        self.IV = [x & self.M for x in IV32]
        # popcount table for w <= 16 words via bytes
        self._pc = np.array([bin(i).count('1') for i in range(256)], dtype=np.int64)

    # --- primitives on int64 arrays (values kept in [0, 2^w)) -------------
    def m(self, x):
        return x & self.M

    def rotr(self, x, n):
        if n == 0:
            return x & self.M
        return ((x >> n) | (x << (self.w - n))) & self.M

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
        return ((e & f) ^ (~e & g)) & self.M

    def Maj(self, a, b, c):
        return ((a & b) ^ (a & c) ^ (b & c)) & self.M

    def T2(self, a, b, c):
        return (self.S0(a) + self.Maj(a, b, c)) & self.M

    def popcount(self, x):
        x = np.asarray(x, dtype=np.int64) & self.M
        tot = np.zeros_like(x)
        for sh in range(0, self.w, 8):
            tot += self._pc[(x >> sh) & 0xFF]
        return tot

    # --- reference compression (scalar ints), for selftest ----------------
    def forward(self, W):
        a = {-1: self.IV[0], -2: self.IV[1], -3: self.IV[2], -4: self.IV[3]}
        e = {-1: self.IV[4], -2: self.IV[5], -3: self.IV[6], -4: self.IV[7]}
        Wf = list(W)
        for t in range(16, R):
            Wf.append((self.s1(Wf[t - 2]) + Wf[t - 7] + self.s0(Wf[t - 15]) + Wf[t - 16]) & self.M)
        for r in range(R):
            T1 = (e[r - 4] + self.S1(e[r - 1]) + self.Ch(e[r - 1], e[r - 2], e[r - 3]) + self.K[r] + Wf[r]) & self.M
            a[r] = (T1 + self.T2(a[r - 1], a[r - 2], a[r - 3])) & self.M
            e[r] = (a[r - 4] + T1) & self.M
        return a, e, Wf

    # --- context realisation ---------------------------------------------
    def realise(self, defs, a):
        """a: dict with a0..a3 (arrays), the unknown word(s), digest a12..a19 and
        IV entries -1..-4 present; defs: dict i -> definition for context words
        not already in `a`.  Fills a[i] for i in 4..11 in dependency order."""
        M = self.M
        pend = {i: d for i, d in defs.items() if i not in a}
        while pend:
            prog = False
            for i, d in list(pend.items()):
                k = d[0]
                if k == 'free' or k == 'const':
                    deps = []
                elif k in ('eq', 'neq'):
                    deps = [d[1]]
                elif k in ('rot', 'rotx', 'add'):
                    deps = [d[1]]
                elif k == 'sat_r':
                    deps = [i - 4, i - 1, i - 2, i - 3]
                elif k == 'sat_h':
                    r = i + 4; deps = [r, r - 1, r - 2, r - 3]
                else:
                    raise ValueError(d)
                if not all(j in a for j in deps):
                    continue
                if k == 'free':
                    raise RuntimeError(f"free word a{i} has no value")
                if k == 'const':
                    a[i] = d[1] & M
                elif k == 'eq':
                    a[i] = a[d[1]]
                elif k == 'neq':
                    a[i] = (~a[d[1]]) & M
                elif k == 'rot':
                    a[i] = self.rotr(a[d[1]], d[2])
                elif k == 'rotx':
                    a[i] = self.rotr(a[d[1]], d[2]) ^ (d[3] & M)
                elif k == 'add':
                    a[i] = (a[d[1]] + d[2]) & M
                elif k == 'sat_r':
                    r = i
                    a[i] = (d[1] - a[r - 4] + self.T2(a[r - 1], a[r - 2], a[r - 3])) & M
                elif k == 'sat_h':
                    r = i + 4
                    a[i] = (d[1] - a[r] + self.T2(a[r - 1], a[r - 2], a[r - 3])) & M
                del pend[i]; prog = True
            if not prog:
                raise RuntimeError(f"cyclic/unsatisfiable context definitions: {pend}")
        return a

    def targets(self, a):
        """e_r, W_r for r < 20 and the four lookup targets T_j (arrays)."""
        M, K = self.M, self.K
        e = {-1: self.IV[4], -2: self.IV[5], -3: self.IV[6], -4: self.IV[7]}
        for r in range(R):
            e[r] = (a[r - 4] + a[r] - self.T2(a[r - 1], a[r - 2], a[r - 3])) & M
        W = {}
        for r in range(R):
            W[r] = (a[r] - self.T2(a[r - 1], a[r - 2], a[r - 3]) - e[r - 4] - self.S1(e[r - 1])
                    - self.Ch(e[r - 1], e[r - 2], e[r - 3]) - K[r]) & M
        T = [(W[16 + j] - self.s1(W[14 + j]) - W[9 + j] - W[j]) & M for j in range(4)]
        return e, W, T

    def base_state(self, rng, N, unknown, free_words, fixed=None):
        """Random a0..a3, digest a12..a19, IV, the unknown and the free context
        words as int64 arrays of length N.  `fixed` overrides (scalar or array)."""
        a = {-1: self.IV[0], -2: self.IV[1], -3: self.IV[2], -4: self.IV[3]}
        rnd = lambda: rng.integers(0, 1 << self.w, size=N, dtype=np.int64)
        for i in range(4):
            a[i] = rnd()
        for i in range(12, R):
            a[i] = rnd()
        a[unknown] = rnd()
        for i in free_words:
            a[i] = rnd()
        if fixed:
            for i, v in fixed.items():
                a[i] = (np.broadcast_to(np.asarray(v, dtype=np.int64), (N,)).copy()) & self.M
        return a

    def edge_weights(self, defs, unknown, rng, N=200, fixed=None, return_all=False):
        """Mean Hamming weight of dT_j per flipped bit of `unknown`, over N random
        states and all w bit positions.  Also returns the fraction of (state,bit)
        pairs with dT_j == 0, and the max weight over states."""
        free_words = [i for i, d in defs.items() if d[0] == 'free']
        a0 = self.base_state(rng, N, unknown, free_words, fixed)
        base = dict(a0)
        self.realise(defs, base)
        _, _, T0 = self.targets(base)
        tot = np.zeros(4); zero = np.zeros(4); per_state = np.zeros((4, N))
        for b in range(self.w):
            a1 = {k: v for k, v in a0.items()}
            a1[unknown] = a0[unknown] ^ (1 << b)
            self.realise(defs, a1)
            _, _, T1 = self.targets(a1)
            for j in range(4):
                d = (T1[j] ^ T0[j]) & self.M
                pc = self.popcount(d)
                tot[j] += pc.sum(); zero[j] += (pc == 0).sum(); per_state[j] += pc
        n = N * self.w
        out = dict(mean=tot / n, zero_frac=zero / n, per_state_mean=per_state / self.w)
        return out


def selftest():
    """w=32 model == SHA-256 reference (submask_family), and rotation amounts."""
    import sys, struct
    sys.path.insert(0, "/home/administrator/sha/publish/code")
    import submask_family as sf
    m = Model(32)
    rng = np.random.default_rng(1)
    Wt = [int(x) for x in rng.integers(0, 1 << 32, 16)]
    a, e, Wf = m.forward(Wt)
    a2, e2, Wf2 = sf.forward(Wt, 20)
    assert all(a[r] == a2[r] for r in range(20)) and all(e[r] == e2[r] for r in range(20)), "w=32 model != SHA-256"
    # recovered words and targets from a full state must give W back
    A = {r: np.array([a[r]], dtype=np.int64) for r in range(-4, 20)}
    _, W, T = m.targets(A)
    assert all(int(W[r][0]) == Wf[r] for r in range(20)), "recover_W mismatch"
    assert all(int(T[j][0]) == (sf.s0(Wf[j + 1])) for j in range(4)), "T_j != s0(W_{j+1}) on a real message"
    for w in (8, 12, 16, 32):
        S0r, S1r, s0r, s1r = scaled(w)
        assert len(set(S0r)) == 3 and len(set(S1r)) == 3 and len(set(s0r[:2])) == 2 and len(set(s1r[:2])) == 2, (w, scaled(w))
    print("selftest OK; amounts:", {w: scaled(w) for w in (8, 12, 32)})


if __name__ == "__main__":
    selftest()
