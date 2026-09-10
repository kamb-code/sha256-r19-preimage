#!/usr/bin/env python3
"""w-bit SHA-256 analogue, any R. Rotation amounts scaled w/32 (round half up),
exactly the convention of experiments/.../redwidth_a5/wmodel.py. At w = 32 this
IS SHA-256, checked against code/verify_r19.py in selftest().

Provides the pieces the R21 joint-tail audit needs:
  * forward compression and the message-schedule recovery W_r(a, e)
  * the backward chain that fixes a_{R-8}..a_{R-1} from a digest
  * sigma0 inverse (sigma0 is a GF(2) bijection at every width tested)
"""
from __future__ import annotations
import numpy as np

def _K64():
    """The 64 SHA-256 round constants: fractional parts of cube roots of primes."""
    ks, n = [], 2
    while len(ks) < 64:
        if all(n % d for d in range(2, int(n ** 0.5) + 1)):
            ks.append(int((n ** (1 / 3) % 1) * (1 << 32)))
        n += 1
    return ks


K32 = _K64()
IV32 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]


def scaled(w):
    r = lambda x: int(np.floor(x * w / 32 + 0.5))
    return ((r(2), r(13), r(22)), (r(6), r(11), r(25)),
            (r(7), r(18), max(r(3), 1)), (r(17), r(19), max(r(10), 1)))


class Model:
    def __init__(self, w, R=21):
        self.w, self.R, self.M = w, R, (1 << w) - 1
        self.S0r, self.S1r, self.s0r, self.s1r = scaled(w)
        self.K = [k & self.M for k in K32]
        self.IV = [x & self.M for x in IV32]
        self._inv = None

    def rotr(self, x, n):
        return x & self.M if n % self.w == 0 else \
            ((x >> (n % self.w)) | (x << (self.w - n % self.w))) & self.M

    def S0(self, x):
        a, b, c = self.S0r; return self.rotr(x, a) ^ self.rotr(x, b) ^ self.rotr(x, c)

    def S1(self, x):
        a, b, c = self.S1r; return self.rotr(x, a) ^ self.rotr(x, b) ^ self.rotr(x, c)

    def s0(self, x):
        a, b, c = self.s0r; return self.rotr(x, a) ^ self.rotr(x, b) ^ ((x & self.M) >> c)

    def s1(self, x):
        a, b, c = self.s1r; return self.rotr(x, a) ^ self.rotr(x, b) ^ ((x & self.M) >> c)

    def Ch(self, e, f, g): return ((e & f) ^ (~e & g)) & self.M
    def Maj(self, a, b, c): return ((a & b) ^ (a & c) ^ (b & c)) & self.M
    def T2(self, a, b, c): return (self.S0(a) + self.Maj(a, b, c)) & self.M

    # ---- sigma0 as a GF(2) matrix, and its inverse -----------------------
    def s0_matrix(self):
        return [self.s0(1 << i) for i in range(self.w)]

    def s0_is_bijective(self):
        basis, rank = [], 0
        for c in self.s0_matrix():
            v = c
            for b in basis:
                v = min(v, v ^ b)
            if v:
                basis.append(v); basis.sort(reverse=True); rank += 1
        return rank == self.w

    def build_s0_inverse(self):
        """Gauss-Jordan over GF(2): returns columns of the inverse map."""
        n = self.w
        aug = [(self.s0(1 << i), 1 << i) for i in range(n)]
        piv = {}
        for col, (v, t) in enumerate(aug):
            pass
        rows = [[ (self.s0(1 << i) >> j) & 1 for i in range(n)] for j in range(n)]
        inv = [[1 if i == j else 0 for i in range(n)] for j in range(n)]
        r = 0
        for c in range(n):
            p = next((k for k in range(r, n) if rows[k][c]), None)
            if p is None:
                return None
            rows[r], rows[p] = rows[p], rows[r]
            inv[r], inv[p] = inv[p], inv[r]
            for k in range(n):
                if k != r and rows[k][c]:
                    rows[k] = [x ^ y for x, y in zip(rows[k], rows[r])]
                    inv[k] = [x ^ y for x, y in zip(inv[k], inv[r])]
            r += 1
        cols = []
        for i in range(n):
            v = 0
            for j in range(n):
                if inv[j][i]:
                    v |= 1 << j
            cols.append(v)
        self._inv = cols
        return cols

    def s0_inv(self, y):
        """Inverse of sigma0, vectorised over int64 arrays."""
        if self._inv is None:
            self.build_s0_inverse()
        y = np.asarray(y, dtype=np.int64) & self.M
        out = np.zeros_like(y)
        for i, col in enumerate(self._inv):
            out ^= np.where(((y >> i) & 1).astype(bool), col, 0)
        return out & self.M

    # ---- forward compression --------------------------------------------
    def expand(self, W):
        Wf = list(W)
        for t in range(16, self.R):
            Wf.append((self.s1(Wf[t-2]) + Wf[t-7] + self.s0(Wf[t-15]) + Wf[t-16]) & self.M)
        return Wf

    def forward(self, W):
        Wf = self.expand(W)
        a = {-1: self.IV[0], -2: self.IV[1], -3: self.IV[2], -4: self.IV[3]}
        e = {-1: self.IV[4], -2: self.IV[5], -3: self.IV[6], -4: self.IV[7]}
        for r in range(self.R):
            T1 = (e[r-4] + self.S1(e[r-1]) + self.Ch(e[r-1], e[r-2], e[r-3])
                  + self.K[r] + Wf[r]) & self.M
            a[r] = (T1 + self.T2(a[r-1], a[r-2], a[r-3])) & self.M
            e[r] = (a[r-4] + T1) & self.M
        return a, e, Wf

    def digest(self, W):
        a, e, _ = self.forward(W)
        R = self.R
        return [(a[R-1] + self.IV[0]) & self.M, (a[R-2] + self.IV[1]) & self.M,
                (a[R-3] + self.IV[2]) & self.M, (a[R-4] + self.IV[3]) & self.M,
                (e[R-1] + self.IV[4]) & self.M, (e[R-2] + self.IV[5]) & self.M,
                (e[R-3] + self.IV[6]) & self.M, (e[R-4] + self.IV[7]) & self.M]

    def recoverW(self, a, e, r):
        """W_r from the state (eq. recW)."""
        return (a[r] - self.T2(a[r-1], a[r-2], a[r-3]) - e[r-4]
                - self.S1(e[r-1]) - self.Ch(e[r-1], e[r-2], e[r-3]) - self.K[r]) & self.M

    def backward_chain(self, H):
        """a_{R-8}..a_{R-1} and e_{R-8}..e_{R-1} from the digest."""
        R = self.R
        a = {R-1: (H[0]-self.IV[0]) & self.M, R-2: (H[1]-self.IV[1]) & self.M,
             R-3: (H[2]-self.IV[2]) & self.M, R-4: (H[3]-self.IV[3]) & self.M}
        e = {R-1: (H[4]-self.IV[4]) & self.M, R-2: (H[5]-self.IV[5]) & self.M,
             R-3: (H[6]-self.IV[6]) & self.M, R-4: (H[7]-self.IV[7]) & self.M}
        for r in range(R-1, R-5, -1):          # e_r = a_{r-4} + a_r - T2(...)
            a[r-4] = (e[r] - a[r] + self.T2(a[r-1], a[r-2], a[r-3])) & self.M
        return a, e


def selftest():
    import hashlib, struct, sys
    m = Model(32, R=64)
    msg = b"abc"
    pad = msg + b"\x80" + b"\x00" * (55 - len(msg)) + struct.pack(">Q", len(msg)*8)
    W = [struct.unpack(">I", pad[4*i:4*i+4])[0] for i in range(16)]
    got = "".join(f"{x:08x}" for x in m.digest(W))
    exp = hashlib.sha256(msg).hexdigest()
    ok = got == exp
    print(f"  w=32 R=64 vs hashlib('abc'): {'OK' if ok else 'FAIL'}")
    if not ok:
        print("   got", got, "\n   exp", exp); sys.exit(1)
    for w in (5, 6, 8, 10, 12, 16, 32):
        mm = Model(w)
        bij = mm.s0_is_bijective()
        inv = mm.build_s0_inverse() is not None
        rng = np.random.default_rng(1)
        x = rng.integers(0, 1 << w, 4096, dtype=np.int64)
        rt = bool(np.all(mm.s0_inv(mm.s0(x)) == (x & mm.M))) if inv else False
        print(f"  w={w:2d}: sigma0 bijective={bij}  inverse built={inv}  round-trip={rt}")
    return True


if __name__ == "__main__":
    selftest()
