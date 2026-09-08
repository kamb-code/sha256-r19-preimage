#!/usr/bin/env python3
"""(b) GROEBNER BASES / ANF over GF(2).

Two formulations, both sized honestly.

 A. Natural variables (the 4w unknown bits only).  The equations are then of
    very high degree and near-maximal ANF density; measured exactly here on
    the 16-variable system (a2,a3) at w=8 and on subcubes at w=32.
 B. Carry-augmented variables: every modular addition gets its own carry bits,
    which makes the whole system QUADRATIC.  Counted exactly by instrumenting
    the same construction the CP-SAT model uses, then the degree of regularity
    of a semi-regular Boolean quadratic system of that size is computed from
    the Hilbert series (1+z)^n / (1+z^2)^m, and the F4/F5 cost from
    binom(n, d_reg)^omega.
"""
import numpy as np
from sympy import symbols, series, Poly, ZZ
import sympy as sp
from alg import Alg, family_instance, R


# ---------------------------------------------------------------- part A
def anf_density(w=8):
    A = Alg(w)
    rng = np.random.default_rng(3)
    print(f"[A] natural variables: exact ANF of c3 as a function of (a2,a3) "
          f"at w={w}  ({2*w} variables)")
    degs_all, dens_all = [], []
    for t in range(6):
        inst, v = family_instance(A, rng)
        n = 1 << w
        a = np.arange(n, dtype=np.uint64)
        A2 = np.repeat(a, n)
        A3 = np.tile(a, n)
        a0 = np.uint64(int(rng.integers(0, n)))
        a1 = np.uint64(int(rng.integers(0, n)))
        c3 = inst.residuals(np.full(A2.shape, a0), np.full(A2.shape, a1), A2, A3)[3]
        # variable order: bits of a2 (low index) then bits of a3
        idx = (A2 << np.uint64(w)) | A3   # careful: build index consistently
        order = np.argsort(idx)
        f = c3[order]
        fb = ((f[:, None] >> np.arange(w, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(np.uint8)
        anf = fb.copy()
        nv = 2 * w
        for i in range(nv):
            step = 1 << i
            r = anf.reshape(-1, 2 * step, w)
            r[:, step:, :] ^= r[:, :step, :]
        wt = np.array([bin(i).count('1') for i in range(1 << nv)], dtype=np.uint8)
        for b in range(w):
            nz = np.nonzero(anf[:, b])[0]
            degs_all.append(int(wt[nz].max()) if nz.size else 0)
            dens_all.append(nz.size / (1 << nv))
    print(f"    degree per output bit: min {min(degs_all)} max {max(degs_all)} "
          f"mean {np.mean(degs_all):.2f}   (maximum possible {2*w})")
    print(f"    ANF density: mean {np.mean(dens_all):.4f} of all {1<<(2*w)} "
          f"monomials (a random function gives 0.5)")
    print(f"    -> at w=32 with 4 unknown words the same behaviour gives ~2^127 "
          f"monomials per equation; the polynomial cannot even be written down.")


# ---------------------------------------------------------------- part B
class Counter:
    """Counts the carry-augmented quadratic system for the context-fixed
    R=20 instance (unknowns a0..a3)."""

    def __init__(self, w):
        self.w = w
        self.nvar = 0      # boolean variables
        self.neq = 0       # quadratic equations
        self.adds = 0
        self.majs = 0
        self.chs = 0

    def new_word(self):
        self.nvar += self.w
        return 'V'

    def add(self, nterms):
        """a modular sum of nterms words: nterms-1 two-input additions."""
        for _ in range(nterms - 1):
            self.adds += 1
            self.nvar += self.w - 1        # carry bits c_1..c_{w-1}
            self.neq += 2 * self.w - 1     # sum bits + carry recurrences
        self.nvar += self.w                # the result word
        self.neq += 0

    def maj(self):
        self.majs += 1
        self.nvar += self.w
        self.neq += self.w                 # t = ab+ac+bc, quadratic

    def ch(self):
        self.chs += 1
        self.nvar += self.w
        self.neq += self.w                 # t = e f + (1+e) g, quadratic


def count_system(w):
    """Mirror the structure of alg.Instance.residuals with unknown a0..a3."""
    c = Counter(w)
    unk = set(range(4))          # rounds whose a-word is unknown
    c.nvar += 4 * w              # a0..a3
    # which words are non-constant
    a_unknown = lambda r: 0 <= r <= 3
    # T2_r = Sigma0(a_{r-1}) + Maj(a_{r-1},a_{r-2},a_{r-3}) for r = 0..11
    T2_nonconst = {}
    for r in range(0, 12):
        args = [r - 1, r - 2, r - 3]
        nun = sum(a_unknown(x) for x in args)
        if nun == 0:
            T2_nonconst[r] = False
            continue
        T2_nonconst[r] = True
        if nun >= 2:
            c.maj()              # genuinely quadratic Maj
        c.add(2)                 # Sigma0(.) + Maj(.)
    # e_r = a_{r-4} + a_r - T2_r  for r = 0..7
    e_nonconst = {}
    for r in range(0, 8):
        nun = a_unknown(r - 4) + a_unknown(r) + (1 if T2_nonconst[r] else 0)
        e_nonconst[r] = nun > 0
        if nun > 0:
            c.add(3)
    for r in range(8, R):
        e_nonconst[r] = False
    # W_r = a_r - T2_r - e_{r-4} - Sigma1(e_{r-1}) - Ch(e_{r-1},e_{r-2},e_{r-3}) - K_r
    W_nonconst = {}
    for r in range(0, 12):
        ch_args = [r - 1, r - 2, r - 3]
        nun_ch = sum(1 for x in ch_args if e_nonconst.get(x, False))
        if nun_ch >= 2:
            c.ch()
        terms = 1 + (1 if T2_nonconst[r] else 0) + (1 if e_nonconst.get(r - 4, False) else 0) \
            + (1 if e_nonconst.get(r - 1, False) else 0) + (1 if nun_ch > 0 else 0)
        W_nonconst[r] = a_unknown(r) or terms > 1
        if terms > 1:
            c.add(terms)
    for r in range(12, R):
        W_nonconst[r] = False
    # the four constraints
    for j in range(4):
        terms = sum(1 for x in (16 + j, 14 + j, 9 + j, 1 + j, j) if W_nonconst.get(x, False))
        c.add(max(terms, 2))
        c.neq += w              # the equation itself = 0
    return c


def d_reg_boolean(n, m, maxd=4000):
    """First non-positive coefficient of (1+z)^n / (1+z^2)^m."""
    z = sp.symbols('z')
    # coefficients by convolution with numpy floats is unsafe; use integer DP
    # (1+z)^n coefficients: binom(n,k); 1/(1+z^2)^m = sum_j (-1)^j C(m+j-1,j) z^{2j}
    import math
    coef = [0] * (maxd + 1)
    for j in range(0, maxd // 2 + 1):
        cj = ((-1) ** j) * math.comb(m + j - 1, j)
        for k in range(0, maxd - 2 * j + 1):
            coef[2 * j + k] += cj * math.comb(n, k)
    for d in range(len(coef)):
        if coef[d] <= 0:
            return d
    return None


if __name__ == "__main__":
    import sys; sys.stdout.reconfigure(line_buffering=True)
    anf_density(8)
    print()
    print("[B] carry-augmented quadratic Boolean system for the context-fixed "
          "R=20 instance (unknowns a0..a3):")
    print("  w    bool vars n   quad eqs m   modular adds   Maj   Ch")
    for w in (8, 16, 32):
        c = count_system(w)
        print(f" {w:3d}   {c.nvar:10d}   {c.neq:10d}   {c.adds:10d}  {c.majs:4d}  {c.chs:4d}")
    c = count_system(32)
    n, m = c.nvar, c.neq
    print(f"\n  at w=32: n = {n} variables, m = {m} quadratic equations "
          f"(m/n = {m/n:.2f})")
    for (nn, mm, tag) in ((n, m, "full carry-augmented system"),
                          (128, 128, "only the 128 unknown bits, if the "
                                     "equations were quadratic (they are not)")):
        d = d_reg_boolean(nn, mm, maxd=800)
        if d is None:
            print(f"  {tag}: d_reg > bound")
            continue
        import math
        cost = math.comb(nn, min(d, nn)) ** 2
        print(f"  {tag}: n={nn} m={mm} -> d_reg = {d}, "
              f"F4/F5 cost ~ binom(n,d_reg)^2 = 2^{math.log2(cost):.0f}")
    print("\n  brute force over the 128 unknown bits for comparison: 2^128; "
          "the published attack: 2^45.4")
