"""Width-parameterised SHA-256 analogue + the submask-family attack at R=19/20.

Mirrors publish/code/submask_family.py exactly, with w-bit words.  Rotation and
shift amounts are the FIPS ones scaled by w/32 and rounded; K and IV are the
FIPS constants truncated to w bits.  At w = 32 the model reproduces the real
attack bit for bit (checked in check32()).
"""
import numpy as np

K32 = [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
       0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
       0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
       0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da]
IV32 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]


def _scale(c, w):
    r = int(c * w / 32 + 0.5)
    return max(1, min(w - 1, r))


class W:
    """A width-w instance of the algebra."""

    def __init__(self, w):
        self.w = w
        self.M = (1 << w) - 1
        if w == 32:
            self.S0r = (2, 13, 22); self.S1r = (6, 11, 25)
            self.s0r = (7, 18, 3);  self.s1r = (17, 19, 10)
        else:
            self.S0r = tuple(_scale(c, w) for c in (2, 13, 22))
            self.S1r = tuple(_scale(c, w) for c in (6, 11, 25))
            self.s0r = tuple(_scale(c, w) for c in (7, 18, 3))
            self.s1r = tuple(_scale(c, w) for c in (17, 19, 10))
        self.K = [c & self.M for c in K32]
        self.IV = [c & self.M for c in IV32]
        self.dt = np.uint64

    # ---- primitives (work on python ints and on uint64 numpy arrays) ----
    def rotr(self, x, n):
        w, M = self.w, self.M
        return ((x >> np.uint64(n)) | (x << np.uint64(w - n))) & M if isinstance(x, np.ndarray) \
            else ((x >> n) | (x << (w - n))) & M

    def shr(self, x, n):
        return (x >> np.uint64(n)) if isinstance(x, np.ndarray) else (x >> n)

    def S0(self, x):
        a, b, c = self.S0r
        return self.rotr(x, a) ^ self.rotr(x, b) ^ self.rotr(x, c)

    def S1(self, x):
        a, b, c = self.S1r
        return self.rotr(x, a) ^ self.rotr(x, b) ^ self.rotr(x, c)

    def s0(self, x):
        a, b, c = self.s0r
        return self.rotr(x, a) ^ self.rotr(x, b) ^ self.shr(x, c)

    def s1(self, x):
        a, b, c = self.s1r
        return self.rotr(x, a) ^ self.rotr(x, b) ^ self.shr(x, c)

    def Ch(self, e, f, g):
        return ((e & f) ^ (~e & g)) & self.M

    def Maj(self, a, b, c):
        return (a & b) ^ (a & c) ^ (b & c)

    def T2(self, a, b, c):
        return (self.S0(a) + self.Maj(a, b, c)) & self.M

    # ---- compression ----
    def forward(self, Wm, R):
        M = self.M
        a = {-1: self.IV[0], -2: self.IV[1], -3: self.IV[2], -4: self.IV[3]}
        e = {-1: self.IV[4], -2: self.IV[5], -3: self.IV[6], -4: self.IV[7]}
        Wf = list(Wm)
        for t in range(16, R):
            Wf.append((self.s1(Wf[t - 2]) + Wf[t - 7] + self.s0(Wf[t - 15]) + Wf[t - 16]) & M)
        for r in range(R):
            T1 = (e[r - 4] + self.S1(e[r - 1]) + self.Ch(e[r - 1], e[r - 2], e[r - 3])
                  + self.K[r] + Wf[r]) & M
            a[r] = (T1 + self.T2(a[r - 1], a[r - 2], a[r - 3])) & M
            e[r] = (a[r - 4] + T1) & M
        return a, e, Wf

    def digest(self, Wm, R):
        a, e, _ = self.forward(Wm, R)
        s = [a[R - 1], a[R - 2], a[R - 3], a[R - 4], e[R - 1], e[R - 2], e[R - 3], e[R - 4]]
        return tuple((x + y) & self.M for x, y in zip(s, self.IV))

    def recover_W(self, a, e, r):
        return (a[r] - self.T2(a[r - 1], a[r - 2], a[r - 3]) - e[r - 4] - self.S1(e[r - 1])
                - self.Ch(e[r - 1], e[r - 2], e[r - 3]) - self.K[r]) & self.M

    def backward_chain(self, h, R):
        M = self.M
        s = [(h[i] - self.IV[i]) & M for i in range(8)]
        a = {R - 1: s[0], R - 2: s[1], R - 3: s[2], R - 4: s[3]}
        e = {R - 1: s[4], R - 2: s[5], R - 3: s[6], R - 4: s[7]}
        for r in (R - 1, R - 2, R - 3, R - 4):
            T1 = (a[r] - self.T2(a[r - 1], a[r - 2], a[r - 3])) & M
            a[r - 4] = (e[r] - T1) & M
        return a, e

    # ---- the sigma0 table ----
    def build_table(self, roots="largest"):
        M = self.M
        u = np.arange(M + 1, dtype=np.uint64)
        t = (self.s0(u) - u) & M
        tbl = np.full(M + 1, M + 1, dtype=np.uint64)   # M+1 = MISS
        order = np.arange(M + 1) if roots == "largest" else np.arange(M, -1, -1)
        tbl[t[order]] = order                          # last write wins
        return tbl

    # ---- the family ----
    def context(self, v, a6, a7, a10, a11=None):
        M = self.M
        c = {4: v, 5: v, 6: a6, 7: a7}
        c[8] = (M - v + self.S0(a7) + self.Maj(a7, a6, v)) & M
        c[9] = (M - v + self.S0(c[8]) + self.Maj(c[8], a7, a6)) & M
        c[10] = a10
        if a11 is not None:
            c[11] = a11
        return c

    def signature(self, h, ctx, R):
        """The five constants the whole per-context computation depends on."""
        M = self.M
        ab, eb = self.backward_chain(h, R)
        a = dict(ab); a.update(ctx)
        a.update({-1: self.IV[0], -2: self.IV[1], -3: self.IV[2], -4: self.IV[3]})
        e = {-1: self.IV[4], -2: self.IV[5], -3: self.IV[6], -4: self.IV[7]}
        for r in range(8, R):
            e[r] = (a[r - 4] + a[r] - self.T2(a[r - 1], a[r - 2], a[r - 3])) & M
        a4, a5, a6, a7, a8, a9, a10 = (a[i] for i in range(4, 11))
        e8, e9, e10 = e[8], e[9], e[10]
        assert e8 == M and e9 == M and a4 == a5
        T1_7 = (a7 - self.T2(a6, a5, a4)) & M
        c6 = (a6 - self.S0(a5)) & M
        W9base = (a9 - self.T2(a8, a7, a6) - self.K[9]) & M
        W10base = (a10 - self.T2(a9, a8, a7) - self.S1(e9) - self.K[10]) & M
        W11base = (a[11] - self.T2(a10, a9, a8) - self.S1(e10) - self.K[11]) & M
        Wr = {r: self.recover_W(a, e, r) for r in range(12, R)}
        K0p = (Wr[16] - self.s1(Wr[14])) & M
        K1p = (Wr[17] - self.s1(Wr[15])) & M
        K2p = (Wr[18] - self.s1(Wr[16])) & M
        K3p = ((Wr[19] - self.s1(Wr[17]) - Wr[12]) & M) if R >= 20 else None
        W9hat = (W9base - (a5 - self.S0(a4) - self.Maj(a4, 0, 0)) - self.S1(e8)
                 - self.Ch(e8, T1_7, (c6 - self.Maj(a5, a4, 0)) & M)) & M
        D = ((a6 - self.S0(a5) - self.Maj(a5, a4, 0)) + self.Ch(e9, e8, T1_7)) & M
        KC0 = (K0p - W9hat) & M
        KC1 = (K1p - W10base + D) & M
        KC2 = (K2p - W11base + T1_7 + self.Ch(e10, e9, e8)) & M
        return (KC0, KC1, KC2, a4, K3p)

    # ---- the sweep, driven by the signature alone ----
    def sweep(self, tbl, sig, R, A0=None):
        """Return (a0,a1,a2,a3) that pass C0,C1,C2, the collapse and (R=20) C3."""
        M = self.M
        KC0, KC1, KC2, v, K3p = sig
        MISS = M + 1
        am1, am2, am3, am4 = self.IV[0], self.IV[1], self.IV[2], self.IV[3]
        em1, em2, em3, em4 = self.IV[4], self.IV[5], self.IV[6], self.IV[7]
        T2iv = self.T2(am1, am2, am3)
        C0c = (-T2iv - em4 - self.S1(em1) - self.Ch(em1, em2, em3) - self.K[0]) & M
        Ce0 = (am4 - T2iv) & M
        if A0 is None:
            A0 = np.arange(M + 1, dtype=np.uint64)
        E0 = (A0 + Ce0) & M
        W0 = (A0 + C0c) & M
        G = (-(self.S0(A0) + self.Maj(A0, am1, am2)) - em3 - self.S1(E0)
             - self.Ch(E0, em1, em2) - self.K[1]) & M
        W1 = tbl[(KC0 - W0 - G) & M]
        keep = W1 != MISS
        A0, E0, W0, G, W1 = (x[keep] for x in (A0, E0, W0, G, W1))
        n_surv = A0.size
        A1 = (W1 - G) & M
        E1 = (am3 + A1 - (self.S0(A0) + self.Maj(A0, am1, am2))) & M
        F12 = (-(self.S0(A1) + self.Maj(A1, A0, am1)) - em2 - self.S1(E1)
               - self.Ch(E1, E0, em1) - self.K[2]) & M
        W2 = tbl[(KC1 - W1 - F12) & M]
        keep = W2 != MISS
        A0, A1, E0, E1, W1, F12, W2 = (x[keep] for x in (A0, A1, E0, E1, W1, F12, W2))
        A2 = (W2 - F12) & M
        e2 = (am2 + A2 - (self.S0(A1) + self.Maj(A1, A0, am1))) & M
        F23 = (-(self.S0(A2) + self.Maj(A2, A1, A0)) - em1 - self.S1(e2)
               - self.Ch(e2, E1, E0) - self.K[3]) & M
        W3 = tbl[(KC2 - W2 - F23) & M]
        keep = W3 != MISS
        A0, A1, A2, E0, E1, e2, F23, W3 = (x[keep] for x in
                                           (A0, A1, A2, E0, E1, e2, F23, W3))
        A3 = (W3 - F23) & M
        n_sol = A0.size
        hit = self.Maj(v, A3, A2) == A3
        idx = np.nonzero(hit)[0]
        out = dict(surv=n_surv, sol=n_sol, eps0=idx.size)
        if idx.size == 0:
            out.update(c3=np.zeros(0, np.uint64), a0=np.zeros(0, np.uint64),
                       a=(np.zeros(0, np.uint64),) * 4, kstar=np.zeros(0, np.uint64))
            return out
        e3 = (am1 + A3[idx] - (self.S0(A2[idx]) + self.Maj(A2[idx], A1[idx], A0[idx]))) & M
        W4 = (v - (self.S0(A3[idx]) + self.Maj(A3[idx], A2[idx], A1[idx])) - E0[idx]
              - self.S1(e3) - self.Ch(e3, e2[idx], E1[idx]) - self.K[4]) & M
        kstar = (self.s0(W4) + W3[idx]) & M          # the kappa3 that would solve C3
        out.update(a0=A0[idx], a=(A0[idx], A1[idx], A2[idx], A3[idx]), kstar=kstar,
                   c3=((kstar - (K3p if K3p is not None else 0)) & M))
        return out
