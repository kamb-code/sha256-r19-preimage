#!/usr/bin/env python3
"""Part (d): GF(2) eigen-structure of the sigma/Sigma maps and of the schedule,
and whether a table could exploit it differently from sigma0(u)-u.

  1. sigma0, sigma1, Sigma0, Sigma1 as 32x32 GF(2) matrices: ranks of f+I, orders,
     characteristic polynomials factored over GF(2), primary invariant subspaces.
  2. On each invariant subspace V of sigma0: is the modular map u -> sigma0(u)-u
     equal to the XOR map u -> sigma0(u)^u (borrow-free)?  Fraction measured
     exhaustively (dim <= 22) or by sampling.
  3. Borrow vector b(u) = (sigma0(u)-u) ^ (sigma0(u)^u): fraction zero, number of
     distinct values in 2^24 samples (lower bound on its entropy), i.e. the cost
     of 'enumerate borrow patterns, then solve linearly'.
  4. XOR-inversion as a first approximation: solve sigma0(u)^u = c linearly and
     measure the Hamming distance of sigma0(u)-u from c.
  5. The a5 blocker kernel g(x) = Sigma1(x+c9) - Sigma0(x): best single-bit linear
     approximation bias and best rank-1 GF(2) correlation over 2^20 samples, for
     c9 in {0, 2^31, random}; the same for Sigma1(x) ^ Sigma0(x) (rank 31) as control.
  6. The linearised schedule: rank of the 128x512 map (W0..W15) -> (W16..W19).
  7. Rotation-symmetric words and how each map treats them.
"""
import sys, time
import numpy as np
import sympy
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import M, S0, S1, s0, s1, rotr, U32

np.seterr(over="ignore")


def mat(f):
    """32x32 GF(2) matrix of a linear map on bit-vectors (column i = f(e_i))."""
    A = np.zeros((32, 32), np.uint8)
    for i in range(32):
        y = f(1 << i)
        for j in range(32):
            A[j, i] = (y >> j) & 1
    return A


def gf2_rank(A):
    A = A.copy() % 2; r = 0; rows, cols = A.shape
    for ccol in range(cols):
        piv = None
        for i in range(r, rows):
            if A[i, ccol]:
                piv = i; break
        if piv is None: continue
        A[[r, piv]] = A[[piv, r]]
        for i in range(rows):
            if i != r and A[i, ccol]:
                A[i] ^= A[r]
        r += 1
        if r == rows: break
    return r


def gf2_nullspace(A):
    """basis of {x : A x = 0} over GF(2), as ints."""
    A = A.copy() % 2; rows, cols = A.shape
    pivcols = []; r = 0
    for ccol in range(cols):
        piv = None
        for i in range(r, rows):
            if A[i, ccol]: piv = i; break
        if piv is None: continue
        A[[r, piv]] = A[[piv, r]]
        for i in range(rows):
            if i != r and A[i, ccol]: A[i] ^= A[r]
        pivcols.append(ccol); r += 1
        if r == rows: break
    free = [j for j in range(cols) if j not in pivcols]
    basis = []
    for fcol in free:
        x = np.zeros(cols, np.uint8); x[fcol] = 1
        for i, pc in enumerate(pivcols):
            if A[i, fcol]: x[pc] = 1
        basis.append(int(sum(int(b) << j for j, b in enumerate(x))))
    return basis


def apply(A, u):
    """apply GF(2) matrix to int u."""
    y = 0
    for i in range(32):
        if (u >> i) & 1:
            col = int(sum(int(A[j, i]) << j for j in range(32)))
            y ^= col
    return y


def matmul(A, B): return (A.astype(np.int64) @ B.astype(np.int64)) % 2


def matpow_poly(A, poly_coeffs):
    """evaluate polynomial (list of GF(2) coeffs, low degree first) at matrix A."""
    n = A.shape[0]; P = np.zeros((n, n), np.uint8); Ak = np.eye(n, dtype=np.uint8)
    for cf in poly_coeffs:
        if cf: P ^= Ak
        Ak = matmul(Ak, A).astype(np.uint8)
    return P


def order(A, maxn=200000):
    I = np.eye(32, dtype=np.uint8); P = A.copy()
    for n in range(1, maxn + 1):
        if np.array_equal(P, I): return n
        P = matmul(P, A).astype(np.uint8)
    return None


def span(basis):
    out = [0]
    for b in basis:
        out = out + [x ^ b for x in out]
    return out


def popcnt_arr(x):
    x = x.astype(np.uint64); cnt = np.zeros_like(x)
    for _ in range(32):
        cnt += x & 1; x >>= 1
    return cnt


def main():
    rng = np.random.default_rng(20260906)
    x = sympy.symbols('x')
    maps = {"sigma0": s0, "sigma1": s1, "Sigma0": S0, "Sigma1": S1}
    mats = {k: mat(f) for k, f in maps.items()}
    print("=== 1. GF(2) structure of the four rotate-xor maps")
    I = np.eye(32, dtype=np.uint8)
    for k, A in mats.items():
        cp = sympy.Matrix(A.astype(int)).charpoly(x)
        fac = sympy.factor_list(sympy.Poly(cp.as_expr(), x, modulus=2))
        facs = ", ".join(f"({sympy.sstr(f.as_expr())})^{m}" for f, m in fac[1])
        print(f"  {k}: rank {gf2_rank(A)}, rank({k}+I) = {gf2_rank(A ^ I)}, order {order(A)}")
        print(f"     charpoly mod 2 = {facs}")
    print(f"  rank(Sigma0 + Sigma1) = {gf2_rank(mats['Sigma0'] ^ mats['Sigma1'])},  "
          f"rank(sigma0 + sigma1) = {gf2_rank(mats['sigma0'] ^ mats['sigma1'])}")

    print("\n=== 2. primary invariant subspaces of sigma0, and borrow-freeness of sigma0(u)-u on them")
    A = mats["sigma0"]
    cp = sympy.Poly(sympy.Matrix(A.astype(int)).charpoly(x).as_expr(), x, modulus=2)
    for f, m in sympy.factor_list(cp)[1]:
        fm = sympy.Poly(f ** m, x, modulus=2)
        coeffs = [int(cc) % 2 for cc in reversed(fm.all_coeffs())]
        P = matpow_poly(A, coeffs)
        basis = gf2_nullspace(P)
        dim = len(basis)
        if dim <= 22:
            V = np.array(span(basis), dtype=np.uint64).astype(U32)
        else:
            sel = rng.integers(0, 2, (1 << 20, dim))
            V = np.zeros(1 << 20, np.uint64)
            for i, b in enumerate(basis):
                V ^= (sel[:, i].astype(np.uint64) * np.uint64(b))
            V = V.astype(U32)
        Vv = V
        modv = (s0(Vv) - Vv); xorv = s0(Vv) ^ Vv
        bf = float((modv == xorv).mean())
        # sigma0 restricted to V is a bijection (invariant); is the modular difference
        # still random-map-like on V?  distinct fraction of sigma0(u)-u over V
        distinct = len(np.unique(modv)) / len(Vv)
        print(f"  factor ({sympy.sstr(f.as_expr())})^{m}: dim {dim}, "
              f"{'exhaustive' if dim <= 22 else '2^20 samples'}: borrow-free fraction {bf:.3e}, "
              f"distinct fraction of sigma0(u)-u on V: {distinct:.4f}")

    print("\n=== 3. borrow vector b(u) = (sigma0(u)-u) ^ (sigma0(u)^u) over 2^24 random u")
    n = 1 << 24
    u = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(U32)
    b = (s0(u) - u) ^ (s0(u) ^ u)
    nd = len(np.unique(b))
    print(f"  fraction with b = 0 (modular == XOR): {(b == 0).mean():.3e}  (2^-32-ish would be {2**-32:.1e}; "
          f"a borrow-free subtraction needs sigma0(u) >= u bitwise)")
    print(f"  distinct borrow vectors: {nd:,} of {n:,} samples ({nd/n:.4f}); mean weight {popcnt_arr(b).mean():.2f}")
    print(f"  -> enumerating borrow patterns to linearise costs >= 2^{np.log2(nd):.1f}; the table costs 1 lookup")

    print("\n=== 4. XOR-inversion as a first approximation to the modular equation")
    L = (A ^ I)
    # solve L u = c for random c in the image: pick u0 random, c = L u0 ^ (small?) -- we want
    # c = sigma0(u*)-u* for a random u*, then solve the XOR system L u = c and compare
    us = rng.integers(0, 1 << 32, 4096, dtype=np.uint64).astype(U32)
    cs = (s0(us) - us)
    # solve L u = c by Gaussian elimination per c
    Lm = L.astype(np.uint8)
    solvable = 0; dist = []
    for cval in cs:
        cval = int(cval)
        Aug = np.concatenate([Lm, np.array([[(cval >> j) & 1] for j in range(32)], np.uint8)], axis=1)
        if gf2_rank(Aug) > gf2_rank(Lm):
            continue
        solvable += 1
        # particular solution
        Aug2 = Aug.copy(); rows, cols = Aug2.shape; pivcols = []; r = 0
        for ccol in range(32):
            piv = None
            for i in range(r, rows):
                if Aug2[i, ccol]: piv = i; break
            if piv is None: continue
            Aug2[[r, piv]] = Aug2[[piv, r]]
            for i in range(rows):
                if i != r and Aug2[i, ccol]: Aug2[i] ^= Aug2[r]
            pivcols.append(ccol); r += 1
        sol = 0
        for i, pc in enumerate(pivcols):
            if Aug2[i, 32]: sol |= 1 << pc
        for ker in (0, 0x27f42515):
            uu = sol ^ ker
            dist.append(bin(((s0(uu) - uu) & M) ^ cval).count("1"))
    dist = np.array(dist)
    print(f"  c = sigma0(u*)-u* for 4096 random u*: XOR system solvable for {solvable}/4096 "
          f"(image of sigma0+I is 31-dim: 1/2 expected)")
    print(f"  Hamming distance of sigma0(u_xor)-u_xor from c: mean {dist.mean():.2f}, min {dist.min()}, "
          f"exact hits {(dist == 0).sum()} of {len(dist)}  (random 32-bit: mean 16)")

    print("\n=== 5. linear approximations of the a5 blocker kernel g(x) = Sigma1(x+c9) - Sigma0(x)")
    n = 1 << 20
    xs = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(U32)
    def best_bias(g):
        # single input bit -> single output bit correlations, plus 4096 random mask pairs
        G = g(xs)
        best = 0.0
        xb = np.stack([(xs >> U32(i)) & U32(1) for i in range(32)])   # 32 x n
        gb = np.stack([(G >> U32(j)) & U32(1) for j in range(32)])
        corr = np.abs((xb.astype(np.float64) @ (1 - 2 * gb.astype(np.float64)).T) / n)  # not centred; do properly
        xpm = 1 - 2 * xb.astype(np.float64); gpm = 1 - 2 * gb.astype(np.float64)
        corr = np.abs(xpm @ gpm.T) / n
        best = corr.max()
        # random masks
        am = rng.integers(0, 1 << 32, 2048, dtype=np.uint64).astype(U32)
        bm = rng.integers(0, 1 << 32, 2048, dtype=np.uint64).astype(U32)
        bestm = 0.0
        for a_, b_ in zip(am, bm):
            par = popcnt_arr((xs & a_) ^ (G & b_)) & 1
            bestm = max(bestm, abs(1 - 2 * par.mean()))
        return best, bestm
    for name, c9 in (("c9=0", 0), ("c9=2^31", 1 << 31), ("c9=random", int(rng.integers(0, 1 << 32)))):
        g = lambda v, c9=c9: (S1(v + U32(c9)) - S0(v))
        b1, bm = best_bias(g)
        print(f"  {name}: best |corr| single-bit {b1:.4f}, best of 2048 random mask pairs {bm:.4f} "
              f"(noise floor ~ {3/np.sqrt(n):.4f})")
    b1, bm = best_bias(lambda v: S1(v) ^ S0(v))
    print(f"  control Sigma1(x)^Sigma0(x) (GF(2)-linear, rank 31): best single-bit |corr| {b1:.4f}, masks {bm:.4f}")
    b1, bm = best_bias(lambda v: S1(v) - S0(v))
    print(f"  Sigma1(x)-Sigma0(x) (modular): best single-bit |corr| {b1:.4f}, masks {bm:.4f}")

    print("\n=== 6. linearised schedule (W0..W15) -> (W16..W19), 128 x 512 over GF(2)")
    def sched_lin(Wbits):
        W = list(Wbits)
        for t in range(16, 20):
            W.append(s1(W[t - 2]) ^ W[t - 7] ^ s0(W[t - 15]) ^ W[t - 16])
        return W[16:20]
    Sm = np.zeros((128, 512), np.uint8)
    for i in range(512):
        Wb = [0] * 16; Wb[i // 32] = 1 << (i % 32)
        out = sched_lin(Wb)
        for t in range(4):
            for j in range(32):
                Sm[32 * t + j, i] = (out[t] >> j) & 1
    print(f"  rank = {gf2_rank(Sm)} (full rank 128 means the four XOR-linearised constraints are independent)")
    # which message words does each constraint touch, and with which map
    print("  C_j (XOR form) = W_{16+j} ^ s1(W_{14+j}) ^ W_{9+j} ^ s0(W_{1+j}) ^ W_j: only W_{1+j} sits under s0;")
    print("  the modular version replaces ^ by +, and the difference is the carry vector of a 5-operand sum.")

    print("\n=== 7. rotation-symmetric words")
    for v in (0, M, 0x55555555, 0xAAAAAAAA, 0x0F0F0F0F):
        print(f"  v={v:08x}: S0={S0(v):08x} S1={S1(v):08x} s0={s0(v):08x} s1={s1(v):08x} "
              f"rot-invariant under: {[r for r in range(1,32) if rotr(v, r) == v]}")
    print("  The Sigma maps commute with rotation; the sigma maps do not (the shifts), the K_r break")
    print("  translation symmetry across rounds, and modular addition breaks rotation symmetry by carries.")


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"\n[{time.time()-t0:.0f}s]")
