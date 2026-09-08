"""LLL / BKZ-lite over exact integer bases, float64 GSO.

Validated against sympy's exact DomainMatrix.lll in selftest().
"""
from __future__ import annotations
import numpy as np
from fractions import Fraction


def _gso(B):
    """B: list of lists of python ints (rows). Returns mu (float), norms2 (float)."""
    d = len(B)
    Bf = np.array([[float(x) for x in r] for r in B], dtype=np.float64)
    n = Bf.shape[1]
    mu = np.zeros((d, d))
    Bs = np.zeros((d, n))
    nrm = np.zeros(d)
    for i in range(d):
        Bs[i] = Bf[i]
        for j in range(i):
            if nrm[j] > 0:
                mu[i, j] = np.dot(Bf[i], Bs[j]) / nrm[j]
                Bs[i] -= mu[i, j] * Bs[j]
        nrm[i] = np.dot(Bs[i], Bs[i])
    return mu, nrm


def lll(B, delta=0.99, max_iter=None):
    """Integer LLL, Cohen 2.6.3 with incremental GSO updates (float64 mu/B)."""
    B = [list(map(int, r)) for r in B]
    d = len(B)
    if d <= 1:
        return B
    if max_iter is None:
        max_iter = 400 * d * d + 20000
    mu, nrm = _gso(B)
    k = 1
    it = 0
    since_refresh = 0
    while k < d and it < max_iter:
        it += 1
        # size reduce row k against j<k
        for j in range(k - 1, -1, -1):
            q = int(round(mu[k, j]))
            if q != 0:
                Bj = B[j]
                B[k] = [a - q * b for a, b in zip(B[k], Bj)]
                mu[k, :j] -= q * mu[j, :j]
                mu[k, j] -= q
        if nrm[k] >= (delta - mu[k, k - 1] ** 2) * nrm[k - 1]:
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            m_ = mu[k, k - 1]
            Bn = nrm[k] + m_ * m_ * nrm[k - 1]
            if Bn <= 0 or not np.isfinite(Bn):
                mu, nrm = _gso(B); k = max(k - 1, 1); continue
            mu[k, k - 1] = m_ * nrm[k - 1] / Bn
            nrm[k] = nrm[k - 1] * nrm[k] / Bn
            nrm[k - 1] = Bn
            if k - 1 > 0:
                tmp = mu[k - 1, :k - 1].copy()
                mu[k - 1, :k - 1] = mu[k, :k - 1]
                mu[k, :k - 1] = tmp
            if k + 1 < d:
                t = mu[k + 1:, k].copy()
                mu[k + 1:, k] = mu[k + 1:, k - 1] - m_ * t
                mu[k + 1:, k - 1] = t + mu[k, k - 1] * mu[k + 1:, k]
            k = max(k - 1, 1)
            since_refresh += 1
            if since_refresh >= 50 * d:
                mu, nrm = _gso(B); since_refresh = 0
    # final exact-ish refresh + one size-reduction pass
    mu, nrm = _gso(B)
    for kk in range(1, d):
        for j in range(kk - 1, -1, -1):
            q = int(round(mu[kk, j]))
            if q != 0:
                B[kk] = [a - q * b for a, b in zip(B[kk], B[j])]
                mu[kk, :j] -= q * mu[j, :j]
                mu[kk, j] -= q
    return B


def lll_exact(B):
    """sympy's exact integer LLL -- required when entries exceed ~2^25,
    where the float64 Gram-Schmidt in lll() loses all precision."""
    from sympy.polys.matrices import DomainMatrix
    from sympy import ZZ
    rows = [[int(x) for x in r] for r in B]
    d, n = len(rows), len(rows[0])
    M = DomainMatrix([[ZZ(x) for x in r] for r in rows], (d, n), ZZ)
    return [[int(x) for x in r] for r in M.lll().to_list()]


def lll_auto(B, thresh=1 << 25):
    mx = max(abs(x) for r in B for x in r)
    return lll_exact(B) if mx > thresh else lll(B)


def enum_shortest(B, mu, nrm, lo, hi, bound2):
    """Schnorr-Euchner enumeration for shortest vector in projected block [lo,hi).
    Returns coefficient vector (ints, length hi-lo) or None."""
    n = hi - lo
    if n <= 1:
        return None
    R = np.zeros((n, n))
    for i in range(n):
        R[i, i] = 1.0
        for j in range(i):
            R[i, j] = mu[lo + i, lo + j]
    c = nrm[lo:hi].copy()
    best = None
    best2 = bound2
    x = np.zeros(n, dtype=np.int64)
    ctr = np.zeros(n)          # center
    partial = np.zeros(n + 1)
    x[n - 1] = 0
    k = n - 1
    last_nonzero = 0
    x[0] = 1
    ctr[:] = 0.0
    step = np.zeros(n, dtype=np.int64)
    step[:] = 1
    # simple recursive enumeration instead (clearer, fine for small blocks)
    res = []

    def rec(k, partial_norm, coefs):
        nonlocal best, best2
        if partial_norm >= best2:
            return
        if k < 0:
            if any(coefs):
                if partial_norm < best2 - 1e-9:
                    best2 = partial_norm
                    best = list(coefs)
            return
        center = -sum(R[i, k] * coefs[i - 0] for i in range(k + 1, n) if coefs[i] != 0)
        rem = best2 - partial_norm
        if rem <= 0 or c[k] <= 0:
            return
        rad = (rem / c[k]) ** 0.5
        loi = int(np.ceil(center - rad - 1e-12))
        hii = int(np.floor(center + rad + 1e-12))
        if hii - loi > 4000:
            return
        order = sorted(range(loi, hii + 1), key=lambda v: abs(v - center))
        for v in order:
            coefs[k] = v
            rec(k - 1, partial_norm + c[k] * (v - center) ** 2, coefs)
        coefs[k] = 0

    rec(n - 1, 0.0, [0] * n)
    return best


def bkz(B, block=20, delta=0.99, tours=4):
    """Very simple BKZ: LLL + block enumeration with insertion."""
    B = lll(B, delta)
    d = len(B)
    for _ in range(tours):
        改 = False
        mu, nrm = _gso(B)
        for lo in range(d - 1):
            hi = min(lo + block, d)
            if hi - lo < 2:
                continue
            co = enum_shortest(B, mu, nrm, lo, hi, nrm[lo] * 0.999)
            if co is None:
                continue
            # build new vector
            new = [0] * len(B[0])
            for i, ci in enumerate(co):
                if ci:
                    new = [a + ci * b for a, b in zip(new, B[lo + i])]
            if not any(new):
                continue
            B = B[:lo] + [new] + B[lo:]
            B = lll(B, delta)
            B = [r for r in B if any(r)]
            改 = True
            mu, nrm = _gso(B)
        if not 改:
            break
    return B


def selftest():
    import random
    from sympy.polys.matrices import DomainMatrix
    from sympy import ZZ
    random.seed(1)
    for trial in range(5):
        d = 12
        B = [[random.randint(-2 ** 20, 2 ** 20) for _ in range(d)] for _ in range(d)]
        mine = lll([r[:] for r in B])
        M = DomainMatrix([[ZZ(x) for x in r] for r in B], (d, d), ZZ)
        ref = M.lll().to_list()
        ref = [[int(x) for x in r] for r in ref]
        n_mine = min(sum(x * x for x in r) for r in mine)
        n_ref = min(sum(x * x for x in r) for r in ref)
        print(f"trial {trial}: mine shortest^2={n_mine} sympy shortest^2={n_ref} "
              f"{'OK' if n_mine <= n_ref else 'WORSE'}")


if __name__ == "__main__":
    selftest()
