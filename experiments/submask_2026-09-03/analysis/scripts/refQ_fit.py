"""refQ_fit.py
(1) Is the 'root-multiplicity distribution' an independent finding, or is it one
    number (the image fraction) re-expressed?  Fit a ONE-parameter mean-1
    under-dispersed family (Binomial(m,1/m), i.e. balls-in-bins with m balls per
    bin's worth of independence) to the EXACT full histogram of f=sigma0(u)-u.
(2) Same question empirically: over the class {invertible GF(2)-linear L} the map
    L(u)-u at n=24; do (>=1,>=2,>=3) deviations lie on a 1-D curve?
(3) Cost bookkeeping for the 'all roots' consequence.
"""
import numpy as np
from math import comb, log, exp

N = 2.0 ** 32
# EXACT histogram from refQ_img mode 0 (my own full-domain scan)
hist = {0: 1573363668, 1: 1586605819, 2: 793515119, 3: 262185142, 4: 64407366,
        5: 12552261, 6: 2022487, 7: 277772, 8: 33662, 9: 3624, 10: 338, 11: 33,
        12: 3, 13: 1, 14: 1}
tot = sum(hist.values())
assert tot == 2 ** 32, tot
assert sum(k * v for k, v in hist.items()) == 2 ** 32   # mean multiplicity exactly 1

obs = np.array([hist.get(k, 0) for k in range(15)], float) / N
print("exact P(k) for f(u)=sigma0(u)-u  vs Poisson(1)  vs best 1-parameter fit")

pois = np.array([exp(-1) / np.math.factorial(k) if hasattr(np.math, 'factorial') else 0 for k in range(15)]) \
    if False else np.array([exp(-1) / np.prod(np.arange(1, k + 1)) for k in range(15)])


def binom_mean1(m, kmax=15):
    """Binomial(m, 1/m): mean 1, under-dispersed, -> Poisson(1) as m->inf."""
    return np.array([comb(m, k) * (1.0 / m) ** k * (1 - 1.0 / m) ** (m - k) for k in range(kmax)])


# match P(0) exactly
best = None
for m in range(2, 100001):
    p0 = (1 - 1.0 / m) ** m
    if best is None or abs(p0 - obs[0]) < abs(best[1] - obs[0]):
        best = (m, p0)
m = best[0]
fit = binom_mean1(m)
print(f"  one-parameter fit: Binomial(m={m}, 1/m)  [m=inf is Poisson(1)]")
print(f"  {'k':>3} {'exact':>14} {'Poisson(1)':>14} {'Binom fit':>14} {'exact-Pois':>12} {'exact-fit':>12}")
for k in range(9):
    print(f"  {k:3d} {obs[k]:14.9f} {pois[k]:14.9f} {fit[k]:14.9f} {obs[k]-pois[k]:+12.2e} {obs[k]-fit[k]:+12.2e}")
ge = lambda a, j: a[j:].sum()
for j in (1, 2, 3):
    print(f"  P(>={j}): exact {ge(obs,j):.9f}  Poisson {ge(pois,j):.9f}  1-par fit {ge(fit,j):.9f}"
          f"   (exact-fit {ge(obs,j)-ge(fit,j):+.2e},  exact-Poisson {ge(obs,j)-ge(pois,j):+.2e})")

print("\n(2) class experiment at n=24: are dev(>=2), dev(>=3) determined by dev(>=1)?")
n = 24
Nn = 1 << n
u = np.arange(Nn, dtype=np.uint32)


def image_stats(v):
    c = np.bincount(v, minlength=Nn)
    h = np.bincount(c, minlength=6).astype(float)
    return np.array([(Nn - h[:j].sum()) / Nn for j in (1, 2, 3)])


def apply_lin(cols, x):
    half = n // 2
    lo = np.zeros(1 << half, np.uint32)
    hi = np.zeros(1 << (n - half), np.uint32)
    idx = np.arange(1 << half, dtype=np.uint32)
    for i in range(half):
        lo[((idx >> i) & 1) == 1] ^= cols[i]
    idx2 = np.arange(1 << (n - half), dtype=np.uint32)
    for i in range(n - half):
        hi[((idx2 >> i) & 1) == 1] ^= cols[half + i]
    return lo[x & ((1 << half) - 1)] ^ hi[x >> half]


rng = np.random.default_rng(7)
rows = []
for t in range(40):
    cols = rng.integers(0, Nn, size=n, dtype=np.uint64).astype(np.uint32)
    v = ((apply_lin(cols, u) - u) & (Nn - 1)).astype(np.uint32)
    rows.append(image_stats(v))
R = np.array(rows)
P = np.array([1 - exp(-1), 1 - 2 * exp(-1), 1 - 2.5 * exp(-1)])
D = R - P
print(f"  n=24, 40 random invertible linear L.  random-map sd of each stat ~ 7.6e-5")
for j, nm in enumerate([">=1", ">=2", ">=3"]):
    print(f"   dev({nm}): mean {D[:,j].mean():+.6f} sd {D[:,j].std(ddof=1):.6f}")
for j, nm in enumerate([">=2", ">=3"]):
    sl, ic = np.polyfit(D[:, 0], D[:, j + 1], 1)
    r = np.corrcoef(D[:, 0], D[:, j + 1])[0, 1]
    resid = D[:, j + 1] - (sl * D[:, 0] + ic)
    print(f"   dev({nm}) = {sl:+.4f}*dev(>=1) {ic:+.2e}   r={r:+.4f}  resid sd={resid.std(ddof=1):.2e}")
print("   sigma0 at n=32 for comparison: dev(>=1)=+1.5522e-03  dev(>=2)=+2.115e-05  dev(>=3)=-7.938e-04")
print(f"   -> class slopes predict dev(>=2)={np.polyfit(D[:,0],D[:,1],1)[0]*1.5522e-3:+.2e}, "
      f"dev(>=3)={np.polyfit(D[:,0],D[:,2],1)[0]*1.5522e-3:+.2e}")

print("\n(3) cost bookkeeping")
p_exact = 2721603628 / 2 ** 32
p_rand = 1 - exp(-1)
for nm, p in (("exact", p_exact), ("random-map", p_rand)):
    lk_max = 1 + p + p * p           # lookups per swept a0, max-representative (abort on miss)
    cand_max = p ** 3
    lk_all = 3.0                     # all-roots: E[n]=1 each level -> 1 + 1 + 1 lookups
    cand_all = 1.0
    print(f"  {nm:10s} p={p:.9f}  p^3={p**3:.6f}")
    print(f"     max-rep : {cand_max:.6f} cand / swept a0, {lk_max:.4f} lookups / swept a0"
          f"  => {lk_max/cand_max:.3f} lookups per candidate")
    print(f"     all-root: {cand_all:.6f} cand / swept a0, {lk_all:.4f} lookups / swept a0"
          f"  => {lk_all/cand_all:.3f} lookups per candidate")
    print(f"     gain: candidates/swept-a0 x{cand_all/cand_max:.5f}   lookups/candidate x{(lk_max/cand_max)/(lk_all/cand_all):.5f}")
print(f"  effect of the anomaly itself on the all-roots gain: {1/p_exact**3:.5f} vs {1/p_rand**3:.5f}"
      f"  = {(1/p_exact**3)/(1/p_rand**3)-1:+.4%}")
print(f"  effect of the anomaly on 20-round cost 2^45.4: {log(p_exact**3/p_rand**3,2):+.5f} bits")
