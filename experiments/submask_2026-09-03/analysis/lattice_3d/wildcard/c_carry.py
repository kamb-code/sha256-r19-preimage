#!/usr/bin/env python3
"""(c) CARRY STRUCTURE.

Three questions, each answered exhaustively at reduced width and confirmed at
full width.

Q1  On the candidate manifold (c0=c1=c2=0) what is the residual of a general
    Z-linear combination L = sum_j lam_j c_j?   [prediction: lam_3 * c3]
Q2  Zero-set size of every low-weight combination over a free pair/triple of
    unknowns, exhaustively.  The known collapse (eps = Maj(v,a3,a2)-a3) has a
    zero set of 3^w; a structureless 32-bit filter has 2^(kw-w).  This is the
    direct measurement of "does the condition decompose per bit".
Q3  Bit-locality: for a product-set (per-bit) condition, flipping bit b of an
    unknown may only move bit b of the residual.  Measure the off-diagonal
    mass for eps (positive control) and for every c_j.
"""
import itertools
import sys
import numpy as np
from alg import Alg, Instance, family_instance, plant, R


def grid_pair(w):
    n = 1 << w
    a = np.arange(n, dtype=np.uint64)
    X = np.repeat(a, n)
    Y = np.tile(a, n)
    return X, Y


def q1_candidate_manifold(w=8, n_inst=40, seed=1):
    """Exhaustive at width w: enumerate every (a0,a1,a2,a3) with c0=c1=c2=0 and
    check that every Z-combination equals lam3*c3 there."""
    A = Alg(w)
    M = A.M
    rng = np.random.default_rng(seed)
    n = 1 << w
    tot = 0
    bad = 0
    c3vals = []
    for _ in range(n_inst):
        inst, v = family_instance(A, rng)
        # brute force over (a0,a1,a2,a3) is 2^(4w); instead fix a0 and sweep
        # (a1,a2,a3) = 2^(3w).  w=8 -> 16.7M, fine.
        for a0 in rng.integers(0, n, 2, dtype=np.uint64):
            A1 = np.repeat(np.arange(n, dtype=np.uint64), n * n)
            A2 = np.tile(np.repeat(np.arange(n, dtype=np.uint64), n), n)
            A3 = np.tile(np.arange(n, dtype=np.uint64), n * n)
            A0 = np.full(A1.shape, a0, dtype=np.uint64)
            c = inst.residuals(A0, A1, A2, A3)
            keep = (c[0] == 0) & (c[1] == 0) & (c[2] == 0)
            idx = np.nonzero(keep)[0]
            tot += idx.size
            if idx.size == 0:
                continue
            cc = [x[idx] for x in c]
            c3vals.append(cc[3])
            for lam in itertools.product((-2, -1, 0, 1, 2), repeat=4):
                if all(l == 0 for l in lam):
                    continue
                L = np.zeros(idx.size, dtype=np.uint64)
                for j in range(4):
                    L = (L + np.uint64(lam[j] % (1 << w)) * cc[j]) & M
                want = (np.uint64(lam[3] % (1 << w)) * cc[3]) & M
                if not np.array_equal(L, want):
                    bad += 1
            del A1, A2, A3, A0, c
    c3all = np.concatenate(c3vals) if c3vals else np.array([], dtype=np.uint64)
    print(f"[Q1] w={w}: {tot} points on the candidate manifold from {n_inst} "
          f"instances; combination-identity violations: {bad}")
    if c3all.size:
        z = int((c3all == 0).sum())
        print(f"     c3 on the manifold: {z} zeros of {c3all.size} "
              f"(uniform predicts {c3all.size/ (1<<w):.1f}); "
              f"distinct values {len(np.unique(c3all))}/{1<<w}")
    return tot, bad


def zero_set_pair(inst, A, free, fixed_vals, w):
    """Zero-set sizes of all +-1 combinations over a free PAIR of unknowns."""
    M = A.M
    n = 1 << w
    X, Y = grid_pair(w)
    args = []
    it = 0
    for k in range(4):
        if k in free:
            args.append(X if k == free[0] else Y)
        else:
            args.append(np.full(X.shape, fixed_vals[k], dtype=np.uint64))
    c = inst.residuals(*args)
    out = {}
    for lam in itertools.product((-1, 0, 1), repeat=4):
        if all(l == 0 for l in lam):
            continue
        L = np.zeros(X.shape, dtype=np.uint64)
        for j in range(4):
            if lam[j] == 0:
                continue
            L = (L + np.uint64(lam[j] % (1 << w)) * c[j]) & M
        out[lam] = int((L == 0).sum())
    return out, c, X, Y


def is_product_set(zx, zy, w):
    """Given the zero set as (x,y) pairs, test whether it is a product set
    across bit positions: Z == prod_i Z_i with Z_i the projection to bit i."""
    n = zx.size
    if n == 0:
        return False, 0
    allowed = []
    for i in range(w):
        bx = ((zx >> np.uint64(i)) & np.uint64(1)).astype(np.uint8)
        by = ((zy >> np.uint64(i)) & np.uint64(1)).astype(np.uint8)
        pats = set(zip(bx.tolist(), by.tolist()))
        allowed.append(len(pats))
    prod = 1
    for a in allowed:
        prod *= a
    return prod == n, prod


def q2_zero_sets(w=8, n_inst=12, seed=2):
    A = Alg(w)
    rng = np.random.default_rng(seed)
    n = 1 << w
    print(f"\n[Q2] w={w}: exhaustive zero sets over free pairs "
          f"(uniform prediction {n} of {n*n})")
    print(f"     control: |{{eps=0}}| over (a2,a3) = 3^{w} = {3**w}")
    agg = {}
    epsz = []
    for t in range(n_inst):
        inst, v = family_instance(A, rng)
        fixed = {k: int(rng.integers(0, n)) for k in range(4)}
        for free in [(2, 3), (1, 2), (1, 3), (0, 3)]:
            out, c, X, Y = zero_set_pair(inst, A, free, fixed, w)
            for lam, cnt in out.items():
                agg.setdefault((free, lam), []).append(cnt)
            if free == (2, 3):
                eps = (A.Maj(np.uint64(v), Y, X) - X) & A.M  # X=a2, Y=a3
                epsz.append(int((eps == 0).sum()))
                # product-set test on the c3 zero set
                z = np.nonzero(c[3] == 0)[0]
                ok, prod = is_product_set(X[z], Y[z], w)
                if t == 0:
                    print(f"     inst0: |{{c3=0}}| over (a2,a3) = {z.size}; "
                          f"product-set: {ok} (product bound {prod})")
                    ze = np.nonzero(eps == 0)[0]
                    oke, prode = is_product_set(X[ze], Y[ze], w)
                    print(f"     inst0: |{{eps=0}}| = {ze.size}; "
                          f"product-set: {oke} (product bound {prode})")
    print(f"     eps zero-set mean over {len(epsz)} instances: "
          f"{np.mean(epsz):.1f} (3^{w} = {3**w})")
    rows = []
    for (free, lam), v in agg.items():
        rows.append((np.mean(v), free, lam, np.max(v)))
    rows.sort(reverse=True)
    print("     top 12 combinations by mean zero-set size:")
    for m, free, lam, mx in rows[:12]:
        print(f"       free={free} lam={lam}: mean {m:9.1f} max {mx:7d} "
              f"(x{m/n:.3f} of uniform)")
    print("     bottom 3:")
    for m, free, lam, mx in rows[-3:]:
        print(f"       free={free} lam={lam}: mean {m:9.1f} max {mx:7d}")
    return rows


def q3_bit_locality(w=32, n_pts=4000, seed=3):
    """Off-diagonal mass of the bit-flip response matrix."""
    A = Alg(w)
    rng = np.random.default_rng(seed)
    inst, v = family_instance(A, rng)
    n = n_pts
    base = [rng.integers(0, 1 << w, n, dtype=np.uint64) for _ in range(4)]
    cb = inst.residuals(*base)
    eps_b = (A.Maj(np.uint64(v), base[3], base[2]) - base[3]) & A.M
    print(f"\n[Q3] w={w}: mean number of residual bits moved by one input bit "
          f"flip ({n} random points)")
    print("     name        a0     a1     a2     a3    (a bitwise/product-set "
          "residual gives 1.00 on its own variables)")
    names = ['c0', 'c1', 'c2', 'c3', 'eps']
    for nm in names:
        row = []
        for k in range(4):
            spread = 0.0
            for b in range(w):
                x = [z.copy() for z in base]
                x[k] = x[k] ^ np.uint64(1 << b)
                if nm == 'eps':
                    f = (A.Maj(np.uint64(v), x[3], x[2]) - x[3]) & A.M
                    d = f ^ eps_b
                else:
                    f = inst.residuals(*x)[names.index(nm)]
                    d = f ^ cb[names.index(nm)]
                spread += np.mean([bin(int(y)).count('1') for y in d[:400]])
            row.append(spread / w)
        print(f"     {nm:8s} " + "  ".join(f"{x:5.2f}" for x in row))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('all', 'q1'):
        q1_candidate_manifold(w=8, n_inst=int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    if what in ('all', 'q2'):
        q2_zero_sets(w=8)
    if what in ('all', 'q3'):
        q3_bit_locality(w=32)
