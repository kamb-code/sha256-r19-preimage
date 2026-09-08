#!/usr/bin/env python3
"""(d) LINEARISATION over GF(2).

Treat the four residuals c0..c3 as GF(2) vectorial Boolean functions of the
128 unknown bits (a0,a1,a2,a3).

 1. Exact ANF linear part  L[i][j] = f_i(e_j) xor f_i(0)   (Moebius, degree 1).
 2. GF(2) rank of L, per residual and for the stacked 128x128 system.
 3. Exactness: which output bits are affine?  (f(x^y) = f(x)^f(y)^f(0))
 4. Correlation of each output bit with its own affine part, over random x.
 5. Algebraic degree profile of c3's output bits, exact on a 16-variable
    subcube (Moebius transform over 2^16).
"""
import numpy as np
from alg import Alg, Instance, family_instance

W = 32
A = Alg(W)
M = A.M


def bits_matrix(vals, w=W):
    """vals: (n,) uint64 -> (n,w) uint8 bit matrix, LSB first."""
    v = np.asarray(vals, dtype=np.uint64)
    return ((v[:, None] >> np.arange(w, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(np.uint8)


def gf2_rank(Mx):
    Mx = Mx.copy().astype(np.uint8)
    rows, cols = Mx.shape
    r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if Mx[i, c]:
                piv = i
                break
        if piv is None:
            continue
        Mx[[r, piv]] = Mx[[piv, r]]
        sel = np.nonzero(Mx[r + 1:, c])[0] + r + 1
        Mx[sel] ^= Mx[r]
        r += 1
        if r == rows:
            break
    return r


def linear_part(inst):
    """Return (const, L) with L of shape (4, 32, 128)."""
    zero = [np.uint64(0)] * 4
    c0 = inst.residuals(*[np.array([0], dtype=np.uint64)] * 4)
    const = np.array([int(x[0]) for x in c0], dtype=np.uint64)
    args = [np.zeros(128, dtype=np.uint64) for _ in range(4)]
    for k in range(4):
        for b in range(W):
            args[k][k * W + b] = np.uint64(1) << np.uint64(b)
    cs = inst.residuals(*args)
    L = np.zeros((4, W, 128), dtype=np.uint8)
    for j in range(4):
        d = cs[j] ^ const[j]
        L[j] = bits_matrix(d).T          # (32 out bits, 128 in bits)
    return const, L


def main():
    rng = np.random.default_rng(17)
    ranks = {j: [] for j in range(4)}
    stack_ranks = []
    affine_bits = {j: np.zeros(W) for j in range(4)}
    corr = {j: np.zeros(W) for j in range(4)}
    n_inst = 20
    n_test = 20000
    for t in range(n_inst):
        inst, v = family_instance(A, rng)
        const, L = linear_part(inst)
        for j in range(4):
            ranks[j].append(gf2_rank(L[j]))
        stack_ranks.append(gf2_rank(L.reshape(4 * W, 128)))
        # affinity / correlation test
        X = [rng.integers(0, 1 << W, n_test, dtype=np.uint64) for _ in range(4)]
        f = inst.residuals(*X)
        xb = np.concatenate([bits_matrix(X[k]) for k in range(4)], axis=1)  # (n,128)
        for j in range(4):
            pred = (xb @ L[j].T) & 1                      # (n,32)
            pred ^= bits_matrix(np.full(n_test, const[j], dtype=np.uint64))
            got = bits_matrix(f[j])
            agree = (pred == got).mean(axis=0)
            corr[j] += agree
            affine_bits[j] += (agree == 1.0)
    print("GF(2) ANF linear part of the residuals, %d random family instances" % n_inst)
    print(" residual   rank(L) of the 32x128 linear part   output bits that are")
    print("            (max 32)                            exactly affine (of 32)")
    for j in range(4):
        print(f"   c{j}       {np.mean(ranks[j]):5.2f}  (min {min(ranks[j])}, max {max(ranks[j])})"
              f"              {affine_bits[j].sum()/n_inst:6.2f}")
    print(f" stacked 128x128 system rank: mean {np.mean(stack_ranks):.2f}, "
          f"min {min(stack_ranks)}, max {max(stack_ranks)} (max possible 128)")
    print("\n agreement of each output bit with its own affine part "
          "(0.5 = no linear structure), c3:")
    a3 = corr[3] / n_inst
    print("   bit :  " + " ".join(f"{i:5d}" for i in range(0, 32, 4)))
    print("   agr :  " + " ".join(f"{a3[i]:.3f}" for i in range(0, 32, 4)))
    print(f"   min {a3.min():.4f}  max {a3.max():.4f}  mean {a3.mean():.4f}")
    print("   bits with |2*agr-1| > 0.01 :",
          int((np.abs(2 * a3 - 1) > 0.01).sum()))
    for j in range(3):
        aj = corr[j] / n_inst
        print(f"   c{j}: min {aj.min():.4f} max {aj.max():.4f} "
              f"#bits with |2a-1|>0.01 = {int((np.abs(2*aj-1) > 0.01).sum())}")

    # ---- algebraic degree on a 16-variable subcube -----------------------
    print("\n algebraic degree of c3's output bits, exact on a 16-variable "
          "subcube (bits 0..15 of a3, rest fixed):")
    inst, v = family_instance(A, rng)
    base = [int(rng.integers(0, 1 << W)) for _ in range(4)]
    n = 1 << 16
    idx = np.arange(n, dtype=np.uint64)
    A3v = (np.uint64(base[3] & 0xFFFF0000) | idx)
    A0v = np.full(n, base[0], dtype=np.uint64)
    A1v = np.full(n, base[1], dtype=np.uint64)
    A2v = np.full(n, base[2], dtype=np.uint64)
    f = inst.residuals(A0v, A1v, A2v, A3v)[3]
    fb = bits_matrix(f)                                   # (2^16, 32)
    # Moebius transform in place over 16 variables
    anf = fb.copy()
    for i in range(16):
        step = 1 << i
        r = anf.reshape(-1, 2 * step, 32)
        r[:, step:, :] ^= r[:, :step, :]
    wt = np.array([bin(i).count('1') for i in range(n)], dtype=np.uint8)
    degs = []
    for b in range(W):
        nz = np.nonzero(anf[:, b])[0]
        degs.append(int(wt[nz].max()) if nz.size else 0)
    print("   per-output-bit degree:", degs)
    print(f"   min {min(degs)}, max {max(degs)}, mean {np.mean(degs):.1f} "
          f"(16 = maximal on this subcube)")
    nmon = [int(anf[:, b].sum()) for b in range(W)]
    print(f"   ANF monomial count per bit: min {min(nmon)}, max {max(nmon)}, "
          f"mean {np.mean(nmon):.0f} (dense would be {n/2:.0f})")


if __name__ == "__main__":
    main()
