"""refQ_scale.py -- how does the class spread of the image fraction of L(u)-u
scale with word size n?  Needed to say whether sigma0's +1.55e-3 at n=32 is a
typical member of the class or an outlier.
"""
import numpy as np, time
from math import exp

P1 = 1 - exp(-1)


def run(n, trials, seed=11):
    Nn = 1 << n
    u = np.arange(Nn, dtype=np.uint32)
    half = n // 2
    idx = np.arange(1 << half, dtype=np.uint32)
    idx2 = np.arange(1 << (n - half), dtype=np.uint32)
    rng = np.random.default_rng(seed)
    devs = []
    for t in range(trials):
        cols = rng.integers(0, Nn, size=n, dtype=np.uint64).astype(np.uint32)
        lo = np.zeros(1 << half, np.uint32)
        hi = np.zeros(1 << (n - half), np.uint32)
        for i in range(half):
            lo[((idx >> i) & 1) == 1] ^= cols[i]
        for i in range(n - half):
            hi[((idx2 >> i) & 1) == 1] ^= cols[half + i]
        v = ((lo[u & ((1 << half) - 1)] ^ hi[u >> half]) - u) & (Nn - 1)
        c = np.bincount(v.astype(np.uint32), minlength=Nn)
        img = Nn - int((c == 0).sum())
        devs.append(img / Nn - P1)
    d = np.array(devs)
    print(f"  n={n:2d} ({trials} random invertible-ish L): mean dev {d.mean():+.6f}  sd {d.std(ddof=1):.6f}"
          f"  |dev| median {np.median(np.abs(d)):.6f}   sampling sd of a true random map {np.sqrt(0.097208/Nn):.2e}")
    return d


t0 = time.time()
for n, tr in ((18, 60), (20, 60), (22, 40), (24, 40), (26, 20), (28, 8)):
    run(n, tr)
    print(f"    [{time.time()-t0:.0f}s]")
print("\nsigma0 (n=32) actual dev = +0.001552165")
