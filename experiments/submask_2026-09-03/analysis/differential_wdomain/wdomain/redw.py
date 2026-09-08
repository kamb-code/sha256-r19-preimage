#!/usr/bin/env python3
"""Reduced-width, EXACT absorption test in the message-word domain.

The criterion the real attack meets.  A constraint R_j absorbs an unknown u
by one lookup in a GLOBAL table iff

        R_j(u) = F(u + t) + s        for all u,

with F a fixed one-input function (the table; the same for every context and
every target) and t, s quantities already known when the lookup is made.
This is exactly the shape of C_j -> a_{j+1}:  R = sig0(W_{j+1}) - W_{j+1} + s
with W_{j+1} = a_{j+1} + t.

At width w the test is EXHAUSTIVE and exact: tabulate R_j over all 2^w values
of the unknown for two independent settings of everything else, and ask
whether the two tables are related by a translation of the argument plus a
constant offset.  If yes for every pair of settings, one universal F exists
(F is the common translate class).  If no for even one pair, no global table
can invert that constraint on that unknown.

Positive control: the a-domain absorbers C_j <- a_{j+1}, which must PASS.
Negative control: the known heavy edge a_4 -> C_0 at R = 20, which must FAIL.
"""
import numpy as np, random, sys

sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import K as K32                      # noqa: E402  (all 64)
IV32 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]


class Model:
    """w-bit SHA-256 analogue; rotation amounts scaled by w/32, round half up."""

    def __init__(self, w):
        self.w = w
        self.M = (1 << w) - 1
        r = lambda x: int(np.floor(x * w / 32 + 0.5))
        self.S0r = (r(2), r(13), r(22))
        self.S1r = (r(6), r(11), r(25))
        self.s0r = (r(7), r(18), max(r(3), 1))
        self.s1r = (r(17), r(19), max(r(10), 1))
        self.K = [k & self.M for k in K32]
        self.IV = [x & self.M for x in IV32]

    def rotr(self, x, n):
        n %= self.w
        if n == 0:
            return x & self.M
        return ((x >> n) | (x << (self.w - n))) & self.M

    def _x3(self, x, t, shift=False):
        a, b, c = t
        y = self.rotr(x, a) ^ self.rotr(x, b)
        return y ^ (((x & self.M) >> c) if shift else self.rotr(x, c))

    def S0(self, x): return self._x3(x, self.S0r)
    def S1(self, x): return self._x3(x, self.S1r)
    def s0(self, x): return self._x3(x, self.s0r, True)
    def s1(self, x): return self._x3(x, self.s1r, True)

    def Ch(self, e, f, g): return ((e & f) ^ (~e & g)) & self.M
    def Maj(self, a, b, c): return (a & b) ^ (a & c) ^ (b & c)
    def T2(self, a, b, c): return (self.S0(a) + self.Maj(a, b, c)) & self.M

    def expand(self, W, R):
        Wf = list(W)
        for t in range(16, R):
            Wf.append((self.s1(Wf[t-2]) + Wf[t-7] + self.s0(Wf[t-15]) + Wf[t-16]) & self.M)
        return Wf

    def forward(self, W, R):
        a = {-1: self.IV[0], -2: self.IV[1], -3: self.IV[2], -4: self.IV[3]}
        e = {-1: self.IV[4], -2: self.IV[5], -3: self.IV[6], -4: self.IV[7]}
        Wf = self.expand(W, R)
        for r in range(R):
            T1 = (e[r-4] + self.S1(e[r-1]) + self.Ch(e[r-1], e[r-2], e[r-3])
                  + self.K[r] + Wf[r]) & self.M
            a[r] = (T1 + self.T2(a[r-1], a[r-2], a[r-3])) & self.M
            e[r] = (a[r-4] + T1) & self.M
        return a, e, Wf

    def dig(self, W, R):
        a, e, _ = self.forward(W, R)
        return [a[R-1], a[R-2], a[R-3], a[R-4], e[R-1], e[R-2], e[R-3], e[R-4]]

    # ---- a-domain side -------------------------------------------------
    def recover_W(self, a, e, r):
        return (a[r] - self.T2(a[r-1], a[r-2], a[r-3]) - e[r-4] - self.S1(e[r-1])
                - self.Ch(e[r-1], e[r-2], e[r-3]) - self.K[r]) & self.M

    def backward_chain(self, s, R):
        a = {R-1: s[0], R-2: s[1], R-3: s[2], R-4: s[3]}
        e = {R-1: s[4], R-2: s[5], R-3: s[6], R-4: s[7]}
        for r in (R-1, R-2, R-3, R-4):
            T1 = (a[r] - self.T2(a[r-1], a[r-2], a[r-3])) & self.M
            a[r-4] = (e[r] - T1) & self.M
        return a, e

    def C_resid(self, free, ab, R):
        a = dict(ab); a.update(free)
        a.update({-1: self.IV[0], -2: self.IV[1], -3: self.IV[2], -4: self.IV[3]})
        e = {-1: self.IV[4], -2: self.IV[5], -3: self.IV[6], -4: self.IV[7]}
        for r in range(R):
            e[r] = (a[r-4] + a[r] - self.T2(a[r-1], a[r-2], a[r-3])) & self.M
        Wf = {r: self.recover_W(a, e, r) for r in range(R)}
        return [(Wf[16+j] - self.s1(Wf[14+j]) - Wf[9+j] - self.s0(Wf[1+j]) - Wf[j]) & self.M
                for j in range(R - 16)]


def translate_equiv(fA, fB, M1):
    """Is fB(x) = fA(x+tau) + sigma for some tau, sigma?  Exhaustive in tau."""
    n = len(fA)
    for tau in range(n):
        sig = (fB[0] - fA[tau]) & M1
        ok = True
        for x in range(1, n):
            if fB[x] != ((fA[(x + tau) % n] + sig) & M1):
                ok = False
                break
        if ok:
            return tau, sig
    return None


def selftest32():
    import hashlib, struct
    m = Model(32)
    msg = b"abc"
    pad = msg + b"\x80" + b"\x00" * (56 - 1 - 3) + struct.pack(">Q", 24)
    W = [struct.unpack(">I", pad[4*i:4*i+4])[0] for i in range(16)]
    d = m.dig(W, 64)
    out = b"".join(struct.pack(">I", (x + y) & 0xFFFFFFFF) for x, y in zip(d, m.IV))
    return out == hashlib.sha256(msg).digest()


if __name__ == "__main__":
    print("w=32 model == SHA-256 on 'abc':", "OK" if selftest32() else "FAIL")
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    NPAIR = 4
    m = Model(w)
    M1 = m.M
    N = 1 << w
    rng = random.Random(4242)
    RWd = lambda: rng.randrange(N)
    print(f"\nwidth w={w} ({N} values per word), R={R}, {NPAIR} context pairs per cell")

    print("\n--- POSITIVE / NEGATIVE CONTROL: a-domain, unknown a_k vs C_j ---")
    tgt = m.dig([RWd() for _ in range(16)], R)
    ab, _ = m.backward_chain(tgt, R)
    nfree = R - 8
    print("    a_k  " + " ".join(f"  C{j}" for j in range(R - 16)))
    for k in range(1, min(6, nfree)):
        verdict = []
        for j in range(R - 16):
            ok = True
            for _ in range(NPAIR):
                A = {i: RWd() for i in range(nfree)}
                B = {i: RWd() for i in range(nfree)}
                fA, fB = [], []
                for x in range(N):
                    A[k] = x; B[k] = x
                    fA.append(m.C_resid(A, ab, R)[j])
                    fB.append(m.C_resid(B, ab, R)[j])
                if translate_equiv(fA, fB, M1) is None:
                    ok = False; break
            verdict.append("ABS" if ok else " . ")
        print(f"    a_{k}   " + " ".join(verdict))

    print("\n--- W-DOMAIN: unknown W_i vs each of the 8 digest residuals ---")
    tgtW = m.dig([RWd() for _ in range(16)], R)
    names = ["a-1", "a-2", "a-3", "a-4", "e-1", "e-2", "e-3", "e-4"]
    print("    W_i  " + " ".join(f"{n:>4}" for n in names))
    nabs = 0
    for i in range(16):
        verdict = []
        for j in range(8):
            ok = True
            for _ in range(NPAIR):
                A = [RWd() for _ in range(16)]
                B = [RWd() for _ in range(16)]
                fA, fB = [], []
                for x in range(N):
                    A[i] = x; B[i] = x
                    fA.append((m.dig(A, R)[j] - tgtW[j]) & M1)
                    fB.append((m.dig(B, R)[j] - tgtW[j]) & M1)
                if translate_equiv(fA, fB, M1) is None:
                    ok = False; break
            if ok:
                nabs += 1
            verdict.append(" ABS" if ok else "   .")
        print(f"    W_{i:<2} " + " ".join(verdict))
    print(f"    absorbable (W_i, residual) cells: {nabs} of 128")

    # ------------------------------------------------------------------
    # HYBRID FRAME: digest peeled by the backward chain (constraints are
    # C_0..C_{R-17}), but the 12 free words parameterised as MESSAGE words
    # W_0..W_{R-9}.  a_r for r <= R-9 is a triangular function of W_0..W_r,
    # so this is a legitimate reparameterisation of the same frame.
    # ------------------------------------------------------------------
    print("\n--- HYBRID: backward chain + MESSAGE-word unknowns, vs C_0..C_%d ---"
          % (R - 17))
    nfreeW = R - 8
    tgtH = m.dig([RWd() for _ in range(16)], R)
    abH, _ = m.backward_chain(tgtH, R)

    def C_from_W(Wfree):
        """Wfree = list of W_0..W_{R-9}; forward to a_0..a_{R-9}, then C_j."""
        a = {-1: m.IV[0], -2: m.IV[1], -3: m.IV[2], -4: m.IV[3]}
        e = {-1: m.IV[4], -2: m.IV[5], -3: m.IV[6], -4: m.IV[7]}
        for r in range(nfreeW):
            T1 = (e[r-4] + m.S1(e[r-1]) + m.Ch(e[r-1], e[r-2], e[r-3])
                  + m.K[r] + Wfree[r]) & M1
            a[r] = (T1 + m.T2(a[r-1], a[r-2], a[r-3])) & M1
            e[r] = (a[r-4] + T1) & M1
        return m.C_resid({r: a[r] for r in range(nfreeW)}, abH, R)

    print("    W_i  " + " ".join(f"  C{j}" for j in range(R - 16)))
    ncells = 0
    for i in range(nfreeW):
        verdict = []
        for j in range(R - 16):
            ok = True
            for _ in range(NPAIR):
                A = [RWd() for _ in range(nfreeW)]
                B = [RWd() for _ in range(nfreeW)]
                fA, fB = [], []
                for x in range(N):
                    A[i] = x; B[i] = x
                    fA.append(C_from_W(A)[j])
                    fB.append(C_from_W(B)[j])
                if translate_equiv(fA, fB, M1) is None:
                    ok = False; break
            if ok:
                ncells += 1
            verdict.append("ABS" if ok else " . ")
        print(f"    W_{i:<2}  " + " ".join(verdict))
    print(f"    absorbable (W_i, C_j) cells: {ncells}")

    # ------------------------------------------------------------------
    # CONTROL for the hybrid: unknowns W_0..W_3 (<-> a_0..a_3, triangular)
    # but the CONTEXT a_4..a_{R-9} held fixed in a-coordinates.  Here the
    # published absorbers must reappear -- if they do, the zero above is
    # caused by context drift, not by a bug in the test.
    # ------------------------------------------------------------------
    print("\n--- CONTROL: unknowns W_0..W_3, context a_4..a_%d fixed in a-coords ---"
          % (R - 9))
    ctx = {i: RWd() for i in range(4, R - 8)}

    def C_from_W4(W4):
        a = {-1: m.IV[0], -2: m.IV[1], -3: m.IV[2], -4: m.IV[3]}
        e = {-1: m.IV[4], -2: m.IV[5], -3: m.IV[6], -4: m.IV[7]}
        for r in range(4):
            T1 = (e[r-4] + m.S1(e[r-1]) + m.Ch(e[r-1], e[r-2], e[r-3])
                  + m.K[r] + W4[r]) & M1
            a[r] = (T1 + m.T2(a[r-1], a[r-2], a[r-3])) & M1
            e[r] = (a[r-4] + T1) & M1
        free = {r: a[r] for r in range(4)}
        free.update(ctx)
        return m.C_resid(free, abH, R)

    print("    W_i  " + " ".join(f"  C{j}" for j in range(R - 16)))
    for i in range(4):
        verdict = []
        for j in range(R - 16):
            ok = True
            for _ in range(NPAIR):
                A = [RWd() for _ in range(4)]
                B = [RWd() for _ in range(4)]
                fA, fB = [], []
                for x in range(N):
                    A[i] = x; B[i] = x
                    fA.append(C_from_W4(A)[j])
                    fB.append(C_from_W4(B)[j])
                if translate_equiv(fA, fB, M1) is None:
                    ok = False; break
            verdict.append("ABS" if ok else " . ")
        print(f"    W_{i:<2}  " + " ".join(verdict))
