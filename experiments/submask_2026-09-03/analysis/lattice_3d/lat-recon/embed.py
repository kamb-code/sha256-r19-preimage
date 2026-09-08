"""Standard lattice embedding for: find z in {0,1}^N with A z = b (over Z).

Centred:  z = (1+x)/2,  x in {-1,1}^N,  A x = 2b - A*1 =: b'
Lattice rows (dim N+1, ambient N+1+M):
    j = 0..N-1 : [ e_j | 0 | K*A[:,j] ]
    j = N      : [ 0   | 1 | -K*b'    ]
Target lattice point: [ x | 1 | 0 ], squared norm N+1.
"""
from __future__ import annotations
import math
import numpy as np
from lat import lll, lll_exact, lll_auto, bkz, _gso


def build(A, b, K=None):
    A = [[int(x) for x in r] for r in A]
    M = len(A)
    N = len(A[0])
    b = [int(x) for x in b]
    ones = [sum(A[i][j] for j in range(N)) for i in range(M)]
    bp = [2 * b[i] - ones[i] for i in range(M)]
    if K is None:
        K = int(4 * math.isqrt(N + 1) + 8)
    rows = []
    for j in range(N):
        r = [0] * (N + 1) + [K * A[i][j] for i in range(M)]
        r[j] = 1
        rows.append(r)
    r = [0] * (N + 1) + [-K * bp[i] for i in range(M)]
    r[N] = 1
    rows.append(r)
    return rows, N, M, K


def decode(v, N, M):
    """A lattice vector -> the 0/1 solution it encodes, or None."""
    if any(x != 0 for x in v[N + 1:]):
        return None
    sgn = v[N]
    if sgn not in (1, -1):
        return None
    x = [sgn * v[j] for j in range(N)]
    if any(xx not in (1, -1) for xx in x):
        return None
    return [(1 + xx) // 2 for xx in x]


def zero_eq_sublattice(red, N, M):
    """Rows of an LLL-reduced basis whose equation block vanishes."""
    return [r for r in red if all(x == 0 for x in r[N + 1:])]


def gram_det(vecs):
    if not vecs:
        return None
    V = np.array([[float(x) for x in r] for r in vecs])
    G = V @ V.T
    sign, logdet = np.linalg.slogdet(G)
    if sign <= 0:
        return None
    return math.exp(0.5 * logdet)


def diagnostics(red, N, M):
    L0 = zero_eq_sublattice(red, N, M)
    d0 = len(L0)
    det0 = gram_det(L0)
    tgt = math.sqrt(N + 1)
    out = dict(dim_L=N + 1, dim_L0=d0, target_norm=tgt)
    if det0 and d0 > 0:
        gh = math.sqrt(d0 / (2 * math.pi * math.e)) * det0 ** (1.0 / d0)
        out['det_L0'] = det0
        out['gh_L0'] = gh
        out['ratio'] = tgt / gh
        # expected number of lattice points of L0 within the target norm
        out['log2_count'] = d0 * math.log2(tgt / gh) if gh > 0 else float('nan')
    norms = sorted(math.sqrt(sum(x * x for x in r)) for r in L0) if L0 else []
    out['L0_shortest'] = norms[0] if norms else None
    return out


def solve(A, b, use_bkz=False, block=20, K=None):
    rows, N, M, Kv = build(A, b, K)
    red = lll_auto(rows)
    if use_bkz:
        red = bkz(red, block=block, tours=2)
    sols = []
    for r in red:
        for cand in (r, [-x for x in r]):
            z = decode(cand, N, M)
            if z is not None:
                sols.append(z)
    return sols, red, N, M
