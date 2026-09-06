#!/usr/bin/env python3
"""Reduced-width (w-bit) model of the R=20 submask attack, exhaustive per context.

Rotation amounts scaled by w/32 (round half up); at w=32 they are SHA-256's.
Tables are RELATIONS (every root), so nothing is lost to root policy.

Measures, per frame:
  frame 'ctx'  : v is a context word (the published attack)
  frame 'tie'  : v = a0 + c  (a4 tied to the swept word; context rebuilt per a0)
  frame 'tieS' : v = Sigma0(a0) + c
and reports P(c3 == 0 | candidate) against 2^-w, and the solutions per context.

Also the MITM-sharing count: for one context and one a0, over all 2^w values
of v, how many DISTINCT a1 the C0 relation returns (sharing would be needed
for a meet-in-the-middle over (a3, a4) to cost less than 2^w per a0).
"""
import multiprocessing as mp, sys
import numpy as np


def scaled(w):
    def r(x):
        return int(np.floor(x * w / 32 + 0.5))
    S0r = (r(2), r(13), r(22)); S1r = (r(6), r(11), r(25))
    s0r = (r(7), r(18), max(r(3), 1)); s1r = (r(17), r(19), max(r(10), 1))
    return S0r, S1r, s0r, s1r


K32 = [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
       0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
       0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
       0x0fc19dc6, 0x240ca1cc]
IV32 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
R = 20


class Model:
    def __init__(self, w):
        self.w = w; self.M = (1 << w) - 1
        self.S0r, self.S1r, self.s0r, self.s1r = scaled(w)
        self.K = [k & self.M for k in K32]; self.IV = [x & self.M for x in IV32]
        n = 1 << w
        u = np.arange(n, dtype=np.int64)
        y = (self.s0v(u) - u) & self.M
        self.roots = [[] for _ in range(n)]           # sigma0(u)-u = y  ->  all u
        for uu, yy in zip(u.tolist(), y.tolist()):
            self.roots[yy].append(uu)

    def rotr(self, x, n):
        w = self.w
        return ((x >> n) | (x << (w - n))) & self.M
    def S0(self, x): a, b, c = self.S0r; return self.rotr(x, a) ^ self.rotr(x, b) ^ self.rotr(x, c)
    def S1(self, x): a, b, c = self.S1r; return self.rotr(x, a) ^ self.rotr(x, b) ^ self.rotr(x, c)
    def s0(self, x): a, b, c = self.s0r; return self.rotr(x, a) ^ self.rotr(x, b) ^ (x >> c)
    def s1(self, x): a, b, c = self.s1r; return self.rotr(x, a) ^ self.rotr(x, b) ^ (x >> c)
    def s0v(self, x):                                 # vectorised (int64 arrays)
        a, b, c = self.s0r; w = self.w
        rr = lambda n: ((x >> n) | (x << (w - n))) & self.M
        return rr(a) ^ rr(b) ^ (x >> c)
    def Ch(self, e, f, g): return ((e & f) ^ (~e & g)) & self.M
    def Maj(self, a, b, c): return (a & b) ^ (a & c) ^ (b & c)
    def T2(self, a, b, c): return (self.S0(a) + self.Maj(a, b, c)) & self.M

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

    def recover_W(self, a, e, r):
        return (a[r] - self.T2(a[r - 1], a[r - 2], a[r - 3]) - e[r - 4] - self.S1(e[r - 1])
                - self.Ch(e[r - 1], e[r - 2], e[r - 3]) - self.K[r]) & self.M

    def family(self, v, a6, a7, a10, a11):
        M = self.M
        c = {4: v, 5: v, 6: a6, 7: a7}
        c[8] = (M - v + self.S0(a7) + self.Maj(a7, a6, v)) & M
        c[9] = (M - v + self.S0(c[8]) + self.Maj(c[8], a7, a6)) & M
        c[10], c[11] = a10, a11
        return c

    def constants(self, chain, ctx):
        M, K = self.M, self.K
        a = dict(chain); a.update(ctx)
        a.update({-1: self.IV[0], -2: self.IV[1], -3: self.IV[2], -4: self.IV[3]})
        e = {-1: self.IV[4], -2: self.IV[5], -3: self.IV[6], -4: self.IV[7]}
        for r in range(8, R):
            e[r] = (a[r - 4] + a[r] - self.T2(a[r - 1], a[r - 2], a[r - 3])) & M
        a4, a5, a6, a7, a8, a9, a10 = (a[i] for i in range(4, 11))
        e8, e9, e10 = e[8], e[9], e[10]
        T1_7 = (a7 - self.T2(a6, a5, a4)) & M
        c6 = (a6 - self.S0(a5)) & M
        W9base = ((a9 - self.T2(a8, a7, a6)) - K[9]) & M
        W10base = ((a10 - self.T2(a9, a8, a7)) - self.S1(e9) - K[10]) & M
        W11base = ((a[11] - self.T2(a10, a9, a8)) - self.S1(e10) - K[11]) & M
        Wr = {r: self.recover_W(a, e, r) for r in range(12, R)}
        K0p = (Wr[16] - self.s1(Wr[14])) & M
        K1p = (Wr[17] - self.s1(Wr[15])) & M
        K2p = (Wr[18] - self.s1(Wr[16])) & M
        K3p = (Wr[19] - self.s1(Wr[17]) - Wr[12]) & M
        W9hat = (W9base - (a5 - self.S0(a4)) - self.S1(e8)
                 - self.Ch(e8, T1_7, (c6 - self.Maj(a5, a4, 0)) & M)) & M
        D = ((a6 - self.S0(a5) - self.Maj(a5, a4, 0)) + self.Ch(e9, e8, T1_7)) & M
        return dict(KC0=(K0p - W9hat) & M, KC1=(K1p - W10base + D) & M,
                    KC2=(K2p - W11base + T1_7 + self.Ch(e10, e9, e8)) & M, K3p=K3p, a=a, e=e)

    def sweep(self, chain, ctx_fn, a0_list, h=None):
        """ctx_fn(a0) -> context dict.  Returns (candidates, c3zero, verified, sols)."""
        M, K, IV = self.M, self.K, self.IV
        am1, am2, am3, am4 = IV[0], IV[1], IV[2], IV[3]
        em1, em2, em3, em4 = IV[4], IV[5], IV[6], IV[7]
        T2iv = self.T2(am1, am2, am3)
        C0c = (-T2iv - em4 - self.S1(em1) - self.Ch(em1, em2, em3) - K[0]) & M
        Ce0 = (am4 - T2iv) & M
        cand = c3z = ver = 0
        cache = {}
        for a0 in a0_list:
            ctx = ctx_fn(a0)
            key = tuple(sorted(ctx.items()))
            if key not in cache:
                cache[key] = self.constants(chain, ctx)
            c = cache[key]
            a4 = ctx[4]
            E0 = (a0 + Ce0) & M; W0 = (a0 + C0c) & M
            G = (-(self.S0(a0) + self.Maj(a0, am1, am2)) - em3 - self.S1(E0) - self.Ch(E0, em1, em2) - K[1]) & M
            for W1 in self.roots[(c['KC0'] - W0 - G) & M]:
                A1 = (W1 - G) & M
                E1 = (am3 + A1 - (self.S0(a0) + self.Maj(a0, am1, am2))) & M
                F12 = (-(self.S0(A1) + self.Maj(A1, a0, am1)) - em2 - self.S1(E1) - self.Ch(E1, E0, em1) - K[2]) & M
                for W2 in self.roots[(c['KC1'] - W1 - F12) & M]:
                    A2 = (W2 - F12) & M
                    e2 = (am2 + A2 - (self.S0(A1) + self.Maj(A1, a0, am1))) & M
                    F23 = (-(self.S0(A2) + self.Maj(A2, A1, a0)) - em1 - self.S1(e2) - self.Ch(e2, E1, E0) - K[3]) & M
                    for W3 in self.roots[(c['KC2'] - W2 - F23) & M]:
                        A3 = (W3 - F23) & M
                        if self.Maj(a4, A3, A2) != A3:
                            continue
                        cand += 1
                        e3 = (am1 + A3 - (self.S0(A2) + self.Maj(A2, A1, a0))) & M
                        W4 = (a4 - (self.S0(A3) + self.Maj(A3, A2, A1)) - E0 - self.S1(e3) - self.Ch(e3, e2, E1) - K[4]) & M
                        c3 = (self.s0(W4) + W3 - c['K3p']) & M
                        if c3 == 0:
                            c3z += 1
                            if h is not None:
                                aa = dict(c['a']); aa.update({0: a0, 1: A1, 2: A2, 3: A3})
                                ee = dict(c['e'])
                                for rr in range(R):
                                    ee[rr] = (aa[rr - 4] + aa[rr] - self.T2(aa[rr - 1], aa[rr - 2], aa[rr - 3])) & M
                                Wm = [self.recover_W(aa, ee, rr) for rr in range(16)]
                                af, ef, _ = self.forward(Wm)
                                if all(af[r] == chain[r] for r in range(12, 20)):
                                    ver += 1
        return cand, c3z, ver


def job(args):
    w, seed, frame, n_ctx = args
    m = Model(w); M = m.M
    rng = np.random.default_rng(seed)
    r = lambda: int(rng.integers(0, 1 << w))
    tot = np.zeros(3, np.int64)
    for _ in range(n_ctx):
        Wt = [r() for _ in range(16)]
        af, ef, _ = m.forward(Wt)
        chain = {i: af[i] for i in range(12, 20)}
        a6, a7, a10, a11, c, v = (r() for _ in range(6))
        if frame == 'ctx':
            fn = lambda a0: m.family(v, a6, a7, a10, a11)
        elif frame == 'tie':
            fn = lambda a0: m.family((a0 + c) & M, a6, a7, a10, a11)
        elif frame == 'tieS':
            fn = lambda a0: m.family((m.S0(a0) + c) & M, a6, a7, a10, a11)
        tot += np.array(m.sweep(chain, fn, range(1 << w), h=True))
    return frame, tot, n_ctx


def sharing(w, seed=1):
    """for one context and one a0: distinct a1 over all v (family maintained)."""
    m = Model(w); M = m.M
    rng = np.random.default_rng(seed)
    r = lambda: int(rng.integers(0, 1 << w))
    Wt = [r() for _ in range(16)]
    af, ef, _ = m.forward(Wt)
    chain = {i: af[i] for i in range(12, 20)}
    a6, a7, a10, a11 = (r() for _ in range(4))
    IV = m.IV; K = m.K
    am1, am2, am3, am4 = IV[0], IV[1], IV[2], IV[3]
    em1, em2, em3, em4 = IV[4], IV[5], IV[6], IV[7]
    T2iv = m.T2(am1, am2, am3)
    C0c = (-T2iv - em4 - m.S1(em1) - m.Ch(em1, em2, em3) - K[0]) & M
    Ce0 = (am4 - T2iv) & M
    out = []
    for a0 in [r() for _ in range(8)]:
        E0 = (a0 + Ce0) & M; W0 = (a0 + C0c) & M
        G = (-(m.S0(a0) + m.Maj(a0, am1, am2)) - em3 - m.S1(E0) - m.Ch(E0, em1, em2) - K[1]) & M
        seen = set(); nroots = 0
        for v in range(1 << w):
            c = m.constants(chain, m.family(v, a6, a7, a10, a11))
            for W1 in m.roots[(c['KC0'] - W0 - G) & M]:
                seen.add((W1 - G) & M); nroots += 1
        out.append((nroots, len(seen)))
    return out


if __name__ == "__main__":
    ws = [int(x) for x in sys.argv[1:]] or [8, 10, 12]
    print("rotation amounts at w=32:", scaled(32))
    for w in ws:
        print(f"\n=== w = {w}: amounts S0{scaled(w)[0]} S1{scaled(w)[1]} s0{scaled(w)[2]} s1{scaled(w)[3]}")
        n = {8: 400, 10: 400, 12: 300}.get(w, 20)      # contexts per job
        jobs = [(w, 1000 * w + j, fr, n) for j in range(8) for fr in ('ctx', 'tie', 'tieS')]
        agg = {}
        with mp.Pool(24) as pool:
            for fr, tot, nc in pool.imap_unordered(job, jobs):
                a = agg.setdefault(fr, [np.zeros(3, np.int64), 0]); a[0] += tot; a[1] += nc
        for fr in ('ctx', 'tie', 'tieS'):
            (cand, c3z, ver), nc = agg[fr]
            p = c3z / max(cand, 1)
            print(f"  frame {fr:5s}: {nc} contexts, {cand:,} candidates, c3==0: {c3z} "
                  f"(verified preimages {ver}); P(c3==0|cand) = {p:.3e} = 2^-w x {p * (1 << w):.3f} "
                  f"+- {np.sqrt(max(c3z,1)) / max(cand,1) * (1 << w):.3f}")
        print(f"  MITM sharing at w={w} (one context, 8 random a0): (C0 roots over all v, distinct a1):",
              sharing(w))
