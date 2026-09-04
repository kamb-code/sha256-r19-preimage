#!/usr/bin/env python3
"""EXLENS 02 -- how much steering room is there in the R=20 fourth constraint?

c3 = X - K3p  with  X = s0(W4)+W3  (candidate side)  and K3p a per-context const.
So EVERY additive context lever is worth exactly  2^32 * max_k P(X=k)  and no more.
This script measures what the corpus can see of the law of X, states the power,
and computes the collision-entropy estimator (which lower-bounds the lever).
"""
import sys
import numpy as np
from math import log2, sqrt, erfc

sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import s0, s1, S0, S1

SP = "/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/"
d = np.load(SP + "corpus_r20.npz")
W3 = d["W3"].astype(np.uint32); W4 = d["W4"].astype(np.uint32)
c3 = d["c3"].astype(np.uint32); ctx = d["ctx"].astype(np.int64)
a0 = d["a0"].astype(np.uint32); v = d["v"].astype(np.uint32)
N = W3.size
X = (s0(W4) + W3).astype(np.uint32)
rng = np.random.default_rng(7)
U = rng.integers(0, 1 << 32, size=N, dtype=np.uint64).astype(np.uint32)   # control


def bitbias(arr, name, ntests):
    """per-bit P(bit=1)-0.5 with z scores; returns max |z|"""
    z = []
    for b in range(32):
        p = ((arr >> np.uint32(b)) & np.uint32(1)).mean()
        z.append((p - 0.5) / (0.5 / sqrt(arr.size)))
    z = np.array(z)
    thr = sqrt(2) * abs(np.log(0.05 / ntests)) ** 0.5 * 0  # placeholder
    k = int(np.argmax(np.abs(z)))
    print(f"  {name:16s} max|z| = {abs(z[k]):5.2f} at bit {k:2d}   "
          f"(bias {((arr>>np.uint32(k))&np.uint32(1)).mean()-0.5:+.5f})")
    return z


def chi2_window(arr, w, name):
    """chi-square uniformity over every w-bit window; returns worst (offset, z)"""
    best = (None, -1e9)
    nb = 1 << w
    for off in range(32 - w + 1):
        h = np.bincount(((arr >> np.uint32(off)) & np.uint32(nb - 1)).astype(np.int64),
                        minlength=nb)
        e = arr.size / nb
        chi = ((h - e) ** 2 / e).sum()
        dof = nb - 1
        zz = (chi - dof) / sqrt(2 * dof)
        if zz > best[1]:
            best = (off, zz)
    print(f"  {name:16s} {w}-bit windows: worst z = {best[1]:6.2f} at offset {best[0]}"
          f"   (33-w={33-w} tests)")
    return best


print("=" * 78)
print("N = %d candidates, 40 contexts.  Detection thresholds:" % N)
print("  per-bit bias   : 1 sigma = %.5f in probability; 3.5 sigma = %.5f" %
      (0.5 / sqrt(N), 3.5 * 0.5 / sqrt(N)))
print("  Bonferroni for ~200 tests at 5%% -> |z| > %.2f" % 3.9)
print("=" * 78)

print("\n[A] per-bit bias of the candidate-side quantity X = s0(W4)+W3, "
      "and of its ingredients")
for arr, nm in ((X, "X=s0(W4)+W3"), (W3, "W3 (max root)"), (W4, "W4"),
                (s0(W4).astype(np.uint32), "s0(W4)"), (c3, "c3 residual"),
                (U, "uniform control")):
    bitbias(arr, nm, 32 * 6)

print("\n[B] chi-square uniformity of X and c3 in sliding bit windows")
for arr, nm in ((X, "X"), (c3, "c3"), (W3, "W3"), (U, "control")):
    chi2_window(arr, 8, nm)
    chi2_window(arr, 12, nm)

print("\n[C] COLLISION ENTROPY of X -- the estimator that bounds the lever")
print("    (max_k P(X=k) >= CP, so 2^32*CP is a guaranteed-attainable speedup")
print("     if K3p can be steered to the mode)")


def ncoll(arr):
    _, cnt = np.unique(arr, return_counts=True)
    return int((cnt * (cnt - 1) // 2).sum())


npairs = N * (N - 1) / 2
for arr, nm in ((X, "X"), (c3, "c3"), (U, "control")):
    k = ncoll(arr)
    exp = npairs / 2 ** 32
    print(f"  {nm:8s} exact collisions = {k:4d}   expected if uniform = {exp:.2f}"
          f"   -> CP_hat = 2^{log2(max(k,1e-9)/npairs) if k else float('-inf'):.2f}")
print("  Poisson 95%% upper bound on CP from these counts:")
for arr, nm in ((X, "X"), (c3, "c3")):
    k = ncoll(arr)
    # one-sided 95% upper Poisson bound for observed k
    ub = {0: 3.00, 1: 4.74, 2: 6.30, 3: 7.75, 4: 9.15, 5: 10.51,
          6: 11.84, 7: 13.15, 8: 14.43, 9: 15.71, 10: 16.96}.get(k, k + 1.65 * sqrt(k) + 1)
    print(f"    {nm}: k={k} -> CP <= {ub:.2f}/{npairs:.3g} = 2^{log2(ub/npairs):.2f}"
          f"  i.e. steering gain <= 2^{log2(ub/npairs)+32:.2f}  ... but see note")

print("\n[D] Does the LAW of X depend on the context?  (40 x 256 table on low 8 bits)")
tab = np.zeros((40, 256))
for i in range(N):
    tab[ctx[i], X[i] & 0xFF] += 1
row = tab.sum(1, keepdims=True); col = tab.sum(0, keepdims=True)
e = row * col / tab.sum()
chi = ((tab - e) ** 2 / e).sum(); dof = 39 * 255
print(f"  independence chi2 = {chi:.0f}, dof = {dof}, z = {(chi-dof)/sqrt(2*dof):+.2f}")
tabU = np.zeros((40, 256))
for i in range(N):
    tabU[ctx[i], U[i] & 0xFF] += 1
e = tabU.sum(1, keepdims=True) * tabU.sum(0, keepdims=True) / tabU.sum()
chiU = ((tabU - e) ** 2 / e).sum()
print(f"  control          chi2 = {chiU:.0f}, z = {(chiU-dof)/sqrt(2*dof):+.2f}")

print("\n[E] Can c3 be predicted from a0 ALONE (the only pre-lookup quantity)?")
for nb in (6, 8, 10):
    m = (1 << nb) - 1
    t = np.zeros((1 << nb, 1 << nb))
    np.add.at(t, ((a0 & np.uint32(m)).astype(np.int64),
                  (c3 & np.uint32(m)).astype(np.int64)), 1)
    e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
    ok = e > 0
    chi = (((t - e) ** 2 / np.where(ok, e, 1))[ok]).sum()
    dof = ((1 << nb) - 1) ** 2
    # mutual information estimate, bias-corrected
    p = t / t.sum()
    px = p.sum(1, keepdims=True); py = p.sum(0, keepdims=True)
    nz = p > 0
    mi = (p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum()
    mi_bias = dof / (2 * N * np.log(2))
    print(f"  low {nb:2d} bits of a0 vs c3: chi2 z = {(chi-dof)/sqrt(2*dof):+6.2f}  "
          f"MI_hat = {mi:.4f} bits, plug-in bias = {mi_bias:.4f} bits "
          f"-> MI - bias = {mi-mi_bias:+.4f}")

print("\n[F] Can c3 be predicted from the CONTEXT alone (v, K3p)?  "
      "40 contexts, ~1717 rows each")
print("    proxy: number of candidates with low 8 bits of c3 == 0, per context")
h = np.array([((c3[ctx == c] & 0xFF) == 0).sum() for c in range(40)])
print("    counts:", h)
print(f"    total {h.sum()}, expected {N/256:.1f}; chi2 over contexts = "
      f"{((h-h.mean())**2/h.mean()).sum():.1f}, dof 39, "
      f"z = {(((h-h.mean())**2/h.mean()).sum()-39)/sqrt(2*39):+.2f}")
print("    -> a context effect of factor r on P(c3=0) would show here only if")
print("       r-1 >> sqrt(1/6.7) = 0.39, i.e. only factors above ~1.4 are visible.")
