"""Is the 3w-bit fibre condition searchable, or flat?
Count contexts matching the low k bits of a prescribed (KC0,KC1,KC2) for k=0..3w.
A searchable landscape would show an excess at intermediate k; a flat one follows
2^(4w-k) exactly, and then brute force at 2^(3w) per fibre member is optimal.
"""
import sys, numpy as np
sys.path.insert(0, '/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/lattice/mdtable')
from wmodel import W
from vecsig import make_vecsig

WD = 8
w = W(WD); M = w.M; R = 20
rng = np.random.default_rng(77)
msg = [int(x) for x in rng.integers(0, M + 1, 16, dtype=np.uint64)]
h = w.digest(msg, R)
vs = make_vecsig(w, h, R)
v = int(rng.integers(0, M + 1, dtype=np.uint64))
N = 1 << 24
A = [rng.integers(0, M + 1, N, dtype=np.uint64) for _ in range(4)]
K0, K1, K2, _ = vs(v, *A)
key = (K0 | (K1 << np.uint64(WD)) | (K2 << np.uint64(2 * WD))).astype(np.int64)
tgt = int(key[0])
print(f"w={WD}: {N:,} sampled contexts, target signature {tgt:#08x}")
print(f"  {'k bits matched':>15} {'observed':>10} {'uniform':>12}")
for k in range(0, 3 * WD + 1, 2):
    m = (1 << k) - 1
    obs = int(((key ^ tgt) & m == 0).sum())
    print(f"  {k:15d} {obs:10,} {N / 2**k:12.1f}")
print("  a flat (unsearchable) landscape tracks the uniform column at every k")
