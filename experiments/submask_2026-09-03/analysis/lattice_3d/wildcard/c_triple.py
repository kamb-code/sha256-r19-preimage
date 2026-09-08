#!/usr/bin/env python3
"""Zero-set scan over free TRIPLES of unknowns (the known collapse used a pair,
so a three-variable product-set condition must be excluded too).  Exhaustive at
w=8: 2^24 assignments per (instance, triple)."""
import itertools
import numpy as np
from alg import Alg, family_instance

W = 8
A = Alg(W)
M = A.M
LAMS = [l for l in itertools.product((-1, 0, 1), repeat=4) if l[3] != 0]
TRIPLES = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
n = 1 << W
rng = np.random.default_rng(808)

a = np.arange(n, dtype=np.uint64)
X = np.repeat(a, n * n)
Y = np.tile(np.repeat(a, n), n)
Z = np.tile(a, n * n)

agg = {}
epsr = []
NI = 5
for t in range(NI):
    inst, v = family_instance(A, rng)
    fixed = {k: int(rng.integers(0, n)) for k in range(4)}
    for tri in TRIPLES:
        args = []
        for k in range(4):
            if k == tri[0]:
                args.append(X)
            elif k == tri[1]:
                args.append(Y)
            elif k == tri[2]:
                args.append(Z)
            else:
                args.append(np.full(X.shape, fixed[k], dtype=np.uint64))
        c = inst.residuals(*args)
        for lam in LAMS:
            L = np.zeros(X.shape, dtype=np.uint64)
            for j in range(4):
                if lam[j]:
                    L = (L + np.uint64(lam[j] % (1 << W)) * c[j]) & M
            agg.setdefault((tri, lam), []).append(int((L == 0).sum()))
        del c
    # positive control on a triple: eps = Maj(v,a3,a2) - a3 (free of a1)
    eps = (A.Maj(np.uint64(v), Z, Y) - Z) & M
    epsr.append(int((eps == 0).sum()))

pred = n * n            # 2^(3w) / 2^w
print(f"w={W}, {NI} instances, exhaustive over 2^{3*W} assignments per cell")
print(f"uniform prediction for a 1-word condition on 3 free words: {pred}")
print(f"positive control eps (a product-set condition on (a2,a3), free a1): "
      f"mean {np.mean(epsr):.0f} = {np.mean(epsr)/pred:.3f} x uniform "
      f"[(3/2)^{W} x 1 = {1.5**W:.2f}]")
rows = sorted(((np.mean(x) / pred, k) for k, x in agg.items()), reverse=True)
print(f"{len(agg)} (triple, lambda) cells with lam3 != 0:")
print(f"  best  {rows[0][0]:.4f} x uniform  {rows[0][1]}")
print(f"  worst {rows[-1][0]:.4f} x uniform {rows[-1][1]}")
r = np.array([x[0] for x in rows])
print(f"  mean {r.mean():.4f}  sd {r.std():.4f}  "
      f"(a per-bit collapse would show >= 1.5^8 = 25.6)")
