"""Bit-blast the R=20 submask attack's constraint system C0..C3 into a
Z-LINEAR system in 0/1 variables.

Every SHA-256 primitive is Z-linearly encodable over 0/1 variables:
  XOR/AND/MAJ:  a+b+c = x + 2m  forces x = a^b^c and m = Maj(a,b,c)  (m = a&b when c=0)
  CH:           g ^ (e & (f^g))  -- three such gadgets
  ADD mod 2^w:  per-bit carry chain
so there is NO encoding obstruction.  What the encoding costs is variables:
each gadget introduces 2w new 0/1 variables against w new equations, i.e. it
adds w to the dimension of the solution lattice for free.
"""
from __future__ import annotations
import numpy as np
from wsha import mk, PARAMS


class Blast:
    def __init__(self, w):
        self.w = w
        self.M = (1 << w) - 1
        self.n = 0
        self.rows = []      # list of (dict var->coeff, rhs)
        self.truth = {}     # var -> true bit, for validation
        self.f = mk(w)
        self.gadgets = 0

    def newvar(self, bit):
        i = self.n; self.n += 1
        self.truth[i] = int(bit)
        return ('v', i)

    def newword(self, val):
        return [self.newvar((val >> i) & 1) for i in range(self.w)]

    def const(self, val):
        return [('c', (val >> i) & 1) for i in range(self.w)]

    def val(self, word):
        v = 0
        for i, t in enumerate(word):
            b = t[1] if t[0] == 'c' else self.truth[t[1]]
            v |= b << i
        return v

    def eq(self, terms, rhs):
        """terms: list of (coeff, bitentry). Constants folded into rhs."""
        d = {}
        for co, t in terms:
            if t[0] == 'c':
                rhs -= co * t[1]
            else:
                d[t[1]] = d.get(t[1], 0) + co
        d = {k: v for k, v in d.items() if v}
        # validate against the truth assignment
        s = sum(c * self.truth[k] for k, c in d.items())
        assert s == rhs, f"gadget equation violated: {s} != {rhs}"
        self.rows.append((d, rhs))

    # ---- bit-level gadgets -------------------------------------------------
    def carry_gadget(self, ins):
        """ins: list of 1..3 bit entries.  Returns (xor_bit, carry_bit)."""
        sv = sum(t[1] if t[0] == 'c' else self.truth[t[1]] for t in ins)
        x = self.newvar(sv & 1)
        m = self.newvar(sv >> 1)
        self.eq([(1, t) for t in ins] + [(-1, x), (-2, m)], 0)
        return x, m

    def xorw(self, words):
        """XOR of 2 or 3 words, bitwise."""
        self.gadgets += 1
        out = []
        for i in range(self.w):
            x, _ = self.carry_gadget([wd[i] for wd in words])
            out.append(x)
        return out

    def majw(self, a, b, c):
        self.gadgets += 1
        out = []
        for i in range(self.w):
            _, m = self.carry_gadget([a[i], b[i], c[i]])
            out.append(m)
        return out

    def andw(self, a, b):
        self.gadgets += 1
        out = []
        for i in range(self.w):
            _, m = self.carry_gadget([a[i], b[i]])
            out.append(m)
        return out

    def chw(self, e, f, g):
        x = self.xorw([f, g])
        y = self.andw(e, x)
        return self.xorw([g, y])

    def rot(self, a, n):
        n %= self.w
        return [a[(i + n) % self.w] for i in range(self.w)]

    def shr(self, a, n):
        return [a[i + n] if i + n < self.w else ('c', 0) for i in range(self.w)]

    def s0(self, a):
        p = PARAMS[self.w]['s0']
        return self.xorw([self.rot(a, p[0]), self.rot(a, p[1]), self.shr(a, p[2])])

    def s1(self, a):
        p = PARAMS[self.w]['s1']
        return self.xorw([self.rot(a, p[0]), self.rot(a, p[1]), self.shr(a, p[2])])

    def S0(self, a):
        p = PARAMS[self.w]['S0']
        return self.xorw([self.rot(a, p[0]), self.rot(a, p[1]), self.rot(a, p[2])])

    def S1(self, a):
        p = PARAMS[self.w]['S1']
        return self.xorw([self.rot(a, p[0]), self.rot(a, p[1]), self.rot(a, p[2])])

    def addsub(self, terms):
        """terms: list of (sign, word).  Returns word = sum sign*word mod 2^w.
        Carry chain, all coefficients small."""
        self.gadgets += 1
        true = sum(s * self.val(wd) for s, wd in terms) & self.M
        out = self.newword(true)
        npos = sum(1 for s, _ in terms if s > 0)
        nneg = len(terms) - npos
        carry = 0
        for i in range(self.w):
            sv = sum(s * (wd[i][1] if wd[i][0] == 'c' else self.truth[wd[i][1]])
                     for s, wd in terms) + carry
            zi = (true >> i) & 1
            nc = (sv - zi) // 2
            assert (sv - zi) % 2 == 0
            # carry variable range: [-nneg, npos]
            lo, hi = -nneg, npos
            nb = max(1, (hi - lo).bit_length())
            cbits = [self.newvar(((nc - lo) >> k) & 1) for k in range(nb)]
            assert lo + sum((1 << k) * self.truth[cb[1]] for k, cb in enumerate(cbits)) == nc
            tms = [(s, wd[i]) for s, wd in terms]
            tms += [(-1, out[i])]
            tms += [(-2 * (1 << k), cb) for k, cb in enumerate(cbits)]
            self.eq(tms, 2 * lo - carry)
            carry = nc
        return out

    def force_zero(self, terms):
        """Assert sum sign*word == 0 mod 2^w, without allocating an output word."""
        self.gadgets += 1
        npos = sum(1 for s, _ in terms if s > 0)
        nneg = len(terms) - npos
        carry = 0
        for i in range(self.w):
            sv = sum(s * (wd[i][1] if wd[i][0] == 'c' else self.truth[wd[i][1]])
                     for s, wd in terms) + carry
            nc = sv // 2
            assert sv % 2 == 0, "constraint is not satisfied by the planted solution"
            lo, hi = -nneg, npos
            nb = max(1, (hi - lo).bit_length())
            cbits = [self.newvar(((nc - lo) >> k) & 1) for k in range(nb)]
            tms = [(s, wd[i]) for s, wd in terms]
            tms += [(-2 * (1 << k), cb) for k, cb in enumerate(cbits)]
            self.eq(tms, 2 * lo - carry)
            carry = nc

    def matrix(self):
        N = self.n
        A = [[0] * N for _ in self.rows]
        b = []
        for r, (d, rhs) in enumerate(self.rows):
            for k, c in d.items():
                A[r][k] = c
            b.append(rhs)
        return A, b


def build_R20(w, seed=0, subset='all'):
    """Bit-blast C0..C3 with a planted solution a0..a3.  subset in
    {'all','c3','c0'} selects which constraints to impose."""
    rng = np.random.default_rng(seed)
    f = mk(w); M = f['M']
    R = lambda: int(rng.integers(0, 1 << w))
    a0, a1, a2, a3 = R(), R(), R(), R()
    am1, am2, am3, am4 = R(), R(), R(), R()
    em1, em2, em3 = R(), R(), R()
    K1, K2, K3, K4 = R(), R(), R(), R()
    Ce0, C0c, a4 = R(), R(), R()
    S0, S1, s0, s1 = f['S0'], f['S1'], f['s0'], f['s1']
    Maj = lambda a, b, c: (a & b) ^ (a & c) ^ (b & c)
    Ch = lambda e, ff, g: ((e & ff) ^ (~e & g)) & M

    E0 = (a0 + Ce0) & M
    W0 = (a0 + C0c) & M
    T2a0 = (S0(a0) + Maj(a0, am1, am2)) & M
    G = (-T2a0 - em3 - S1(E0) - Ch(E0, em1, em2) - K1) & M
    W1 = (a1 + G) & M
    KC0 = (s0(W1) - W1 + W0 + G) & M
    E1 = (am3 + a1 - T2a0) & M
    T2a1 = (S0(a1) + Maj(a1, a0, am1)) & M
    F12 = (-T2a1 - em2 - S1(E1) - Ch(E1, E0, em1) - K2) & M
    W2 = (a2 + F12) & M
    KC1 = (s0(W2) - W2 + W1 + F12) & M
    e2 = (am2 + a2 - T2a1) & M
    T2a2 = (S0(a2) + Maj(a2, a1, a0)) & M
    F23 = (-T2a2 - em1 - S1(e2) - Ch(e2, E1, E0) - K3) & M
    W3 = (a3 + F23) & M
    KC2 = (s0(W3) - W3 + W2 + F23) & M
    e3 = (am1 + a3 - T2a2) & M
    T2a3 = (S0(a3) + Maj(a3, a2, a1)) & M
    W4 = (a4 - T2a3 - E0 - S1(e3) - Ch(e3, e2, E1) - K4) & M
    K3p = (s0(W4) + W3) & M

    B = Blast(w)
    A0 = B.newword(a0); A1 = B.newword(a1); A2 = B.newword(a2); A3 = B.newword(a3)
    C = B.const
    bE0 = B.addsub([(1, A0), (1, C(Ce0))])
    bW0 = B.addsub([(1, A0), (1, C(C0c))])
    bT2a0 = B.addsub([(1, B.S0(A0)), (1, B.majw(A0, C(am1), C(am2)))])
    bG = B.addsub([(-1, bT2a0), (-1, C((em3 + K1) & M)), (-1, B.S1(bE0)),
                   (-1, B.chw(bE0, C(em1), C(em2)))])
    bW1 = B.addsub([(1, A1), (1, bG)])
    bE1 = B.addsub([(1, C(am3)), (1, A1), (-1, bT2a0)])
    bT2a1 = B.addsub([(1, B.S0(A1)), (1, B.majw(A1, A0, C(am1)))])
    bF12 = B.addsub([(-1, bT2a1), (-1, C((em2 + K2) & M)), (-1, B.S1(bE1)),
                     (-1, B.chw(bE1, bE0, C(em1)))])
    bW2 = B.addsub([(1, A2), (1, bF12)])
    be2 = B.addsub([(1, C(am2)), (1, A2), (-1, bT2a1)])
    bT2a2 = B.addsub([(1, B.S0(A2)), (1, B.majw(A2, A1, A0))])
    bF23 = B.addsub([(-1, bT2a2), (-1, C((em1 + K3) & M)), (-1, B.S1(be2)),
                     (-1, B.chw(be2, bE1, bE0))])
    bW3 = B.addsub([(1, A3), (1, bF23)])
    be3 = B.addsub([(1, C(am1)), (1, A3), (-1, bT2a2)])
    bT2a3 = B.addsub([(1, B.S0(A3)), (1, B.majw(A3, A2, A1))])
    bW4 = B.addsub([(1, C(a4)), (-1, bT2a3), (-1, bE0), (-1, B.S1(be3)),
                    (-1, B.chw(be3, be2, bE1)), (-1, C(K4))])

    n_before = B.n
    if subset in ('all', 'c0'):
        B.force_zero([(1, B.s0(bW1)), (-1, bW1), (1, bW0), (1, bG), (-1, C(KC0))])
    if subset == 'all':
        B.force_zero([(1, B.s0(bW2)), (-1, bW2), (1, bW1), (1, bF12), (-1, C(KC1))])
        B.force_zero([(1, B.s0(bW3)), (-1, bW3), (1, bW2), (1, bF23), (-1, C(KC2))])
    if subset in ('all', 'c3'):
        B.force_zero([(1, B.s0(bW4)), (1, bW3), (-1, C(K3p))])

    A, b = B.matrix()
    truth = [B.truth[i] for i in range(B.n)]
    meta = dict(w=w, N=B.n, M=len(B.rows), gadgets=B.gadgets,
                a=(a0, a1, a2, a3),
                abits=[[A0, A1, A2, A3][k][i][1] for k in range(4) for i in range(w)])
    return A, b, truth, meta


if __name__ == "__main__":
    import json, math
    for w in (8, 12, 16, 32):
        for subset in ('c3', 'all'):
            A, b, truth, meta = build_R20(w, seed=1, subset=subset)
            N, M = meta['N'], meta['M']
            rk = np.linalg.matrix_rank(np.array(A, dtype=float)) if N < 3000 else M
            print(json.dumps(dict(w=w, subset=subset, vars=N, eqs=M, rank=int(rk),
                                  gadgets=meta['gadgets'],
                                  dim_L0=N + 1 - int(rk),
                                  target_norm=round(math.sqrt(N + 1), 2))))
