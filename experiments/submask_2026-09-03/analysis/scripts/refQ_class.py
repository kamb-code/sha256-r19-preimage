"""refQ_class.py -- is image(L(u)-u) > 1-1/e a property of sigma0 specifically,
or a generic property of the class {GF(2)-linear map} minus identity on Z/2^n?

Exhaustive at n=24 (16.7M) so the random-map sampling noise is sigma=7.6e-5 on the
image fraction -- far below the 1.55e-3 effect claimed at n=32.
"""
import numpy as np, time

n = 24
N = 1 << n
MASK = N - 1
u = np.arange(N, dtype=np.uint32)


def image_stats(v):
    c = np.bincount(v, minlength=N)
    h = np.bincount(c, minlength=5)
    return (N - h[0]) / N, (N - h[0] - h[1]) / N, (N - h[0] - h[1] - h[2]) / N


def apply_lin(cols, x):
    """cols: array of n uint32 column images; apply via two 12-bit lookup tables."""
    half = n // 2
    lo = np.zeros(1 << half, np.uint32)
    hi = np.zeros(1 << (n - half), np.uint32)
    for i in range(half):
        idx = np.arange(1 << half, dtype=np.uint32)
        lo[(idx >> i) & 1 == 1] ^= cols[i]
    for i in range(n - half):
        idx = np.arange(1 << (n - half), dtype=np.uint32)
        hi[(idx >> i) & 1 == 1] ^= cols[half + i]
    return lo[x & ((1 << half) - 1)] ^ hi[x >> half]


def cols_of(fn):
    return np.array([fn(np.uint32(1) << np.uint32(i)) for i in range(n)], dtype=np.uint32)


def rotr(x, r, w=n):
    x = np.uint32(x)
    return ((x >> np.uint32(r)) | (x << np.uint32(w - r))) & np.uint32(MASK)


# ---- 1. the sigma0 analogue at n=24: shifts scaled 7,18,3 -> 5,13,2 (7/32*24=5.25 etc)
def s0_analogue(a, b, c):
    return lambda x: (rotr(x, a) ^ rotr(x, b) ^ (np.uint32(x) >> np.uint32(c))) & np.uint32(MASK)


print(f"n={n}, N={N}, 1-1/e = {1-1/np.e:.9f}, random-map sd of image frac = {np.sqrt(0.097208/N):.2e}")

print("\n-- true random maps (control) --")
rng = np.random.default_rng(1)
for t in range(5):
    v = rng.integers(0, N, size=N, dtype=np.uint64).astype(np.uint32)
    g1, g2, g3 = image_stats(v)
    print(f"  random map {t}: >=1 {g1:.6f}  >=2 {g2:.6f}  >=3 {g3:.6f}")

print("\n-- sigma0-shaped maps  L(u)-u  with L = rotr(a)^rotr(b)^(>>c) --")
res = []
for (a, b, c) in [(5, 13, 2), (7, 18, 3), (2, 13, 3), (5, 14, 2), (3, 11, 4), (6, 17, 2), (1, 8, 7), (9, 19, 5)]:
    if max(a, b) >= n:
        continue
    L = s0_analogue(a, b, c)
    v = (L(u) - u) & MASK
    g1, g2, g3 = image_stats(v.astype(np.uint32))
    res.append(g1)
    print(f"  (a,b,c)=({a:2d},{b:2d},{c}): >=1 {g1:.6f}  >=2 {g2:.6f}  >=3 {g3:.6f}   dev(>=1)={g1-(1-1/np.e):+.6f}")

print("\n-- random invertible GF(2)-linear L, map L(u)-u --")
devs = []
rng2 = np.random.default_rng(7)
t0 = time.time()
trials = 60
for t in range(trials):
    while True:
        cols = rng2.integers(0, N, size=n, dtype=np.uint64).astype(np.uint32)
        # rank check over GF(2)
        rows = list(int(x) for x in cols)
        piv, r = [], 0
        tmp = rows[:]
        for i in range(n):
            p = None
            for j in range(len(tmp)):
                if (tmp[j] >> i) & 1:
                    p = j
                    break
            if p is None:
                continue
            pv = tmp.pop(p)
            piv.append(pv)
            tmp = [x ^ pv if (x >> i) & 1 else x for x in tmp]
            r += 1
        if r == n:
            break
    v = (apply_lin(cols, u) - u) & MASK
    g1, g2, g3 = image_stats(v.astype(np.uint32))
    devs.append(g1 - (1 - 1 / np.e))
    if t < 8:
        print(f"  rand-lin {t}: >=1 {g1:.6f}  >=2 {g2:.6f}  >=3 {g3:.6f}   dev={devs[-1]:+.6f}")
devs = np.array(devs)
print(f"  ... {trials} random invertible linear maps: mean dev of image frac = {devs.mean():+.6f}"
      f"  sd = {devs.std(ddof=1):.6f}  min {devs.min():+.6f} max {devs.max():+.6f}  [{time.time()-t0:.0f}s]")
print(f"  fraction of random-linear maps with dev > 0: {(devs>0).mean():.3f}")
