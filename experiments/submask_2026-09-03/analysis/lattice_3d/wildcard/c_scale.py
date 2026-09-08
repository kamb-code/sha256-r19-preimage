#!/usr/bin/env python3
"""Width scaling of the zero-set excess: is any excess per-bit (exponential in
w, i.e. worth bits) or a constant?  Focus on combinations that contain c3."""
import itertools
import numpy as np
from alg import Alg, Instance, family_instance

LAMS = [l for l in itertools.product((-1, 0, 1), repeat=4) if l[3] != 0]


def run(w, n_inst, seed=11):
    A = Alg(w)
    M = A.M
    rng = np.random.default_rng(seed)
    n = 1 << w
    a = np.arange(n, dtype=np.uint64)
    X = np.repeat(a, n)
    Y = np.tile(a, n)
    agg = {}
    eps_tot = []
    for _ in range(n_inst):
        inst, v = family_instance(A, rng)
        fixed = {k: int(rng.integers(0, n)) for k in range(4)}
        for free in [(2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1)]:
            args = []
            for k in range(4):
                if k == free[0]:
                    args.append(X)
                elif k == free[1]:
                    args.append(Y)
                else:
                    args.append(np.full(X.shape, fixed[k], dtype=np.uint64))
            c = inst.residuals(*args)
            for lam in LAMS:
                L = np.zeros(X.shape, dtype=np.uint64)
                for j in range(4):
                    if lam[j]:
                        L = (L + np.uint64(lam[j] % (1 << w)) * c[j]) & M
                agg.setdefault((free, lam), []).append(int((L == 0).sum()))
        eps = (A.Maj(np.uint64(v), np.tile(a, n), np.repeat(a, n)) - np.tile(a, n)) & M
        eps_tot.append(int((eps == 0).sum()))
    return agg, np.mean(eps_tot), n


if __name__ == "__main__":
    print("free pair sweep, ratio = |zero set| / uniform prediction 2^w")
    print("control eps ratio should be (3/2)^w  (a genuine per-bit collapse)")
    for w in (6, 8, 10):
        n_inst = {6: 60, 8: 24, 10: 6}[w]
        agg, epsm, n = run(w, n_inst)
        best = max(((np.mean(v), k) for k, v in agg.items()))
        worst = min(((np.mean(v), k) for k, v in agg.items()))
        allr = np.array([np.mean(v) for v in agg.values()]) / n
        print(f"  w={w:2d} ({n_inst} instances, {len(agg)} (pair,lambda) cells with lam3!=0): "
              f"eps ratio {epsm/n:8.2f} [(3/2)^w = {1.5**w:8.2f}]   "
              f"best c3-cell ratio {best[0]/n:.3f} {best[1]}   "
              f"mean {allr.mean():.3f}  max {allr.max():.3f}  min {allr.min():.3f}")
