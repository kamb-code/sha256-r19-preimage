"""EXHAUSTIVE amortisation test at w=6 (and w=8 spot check).

Fix the digest and v.  Enumerate ALL 2^(4w) contexts (a6,a7,a10,a11).
  brute force : one sweep of 2^w a0 per context          -> 2^(4w) sweeps
  amortised   : one sweep per distinct (KC0,KC1,KC2),
                then every context is answered by testing whether its kappa3
                lies in that sweep's list L                -> 2^(3w) sweeps
Both must return exactly the same set of 20-round preimages.
The saving is the fibre size, 2^w -- and it is bought by enumerating the
2^(4w) contexts, which is what costs 2^128 at full width.
"""
import sys, time, numpy as np
sys.path.insert(0, '/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/lattice/mdtable')
from wmodel import W
from vecsig import make_vecsig
from t1_validate import rebuild_msg

WD = int(sys.argv[1]) if len(sys.argv) > 1 else 6
w = W(WD); M = w.M; R = 20
rng = np.random.default_rng(2026)
tbl = w.build_table()
msg = [int(x) for x in rng.integers(0, M + 1, 16, dtype=np.uint64)]
h = w.digest(msg, R)
vs = make_vecsig(w, h, R)
v = int(rng.integers(0, M + 1, dtype=np.uint64))
n = M + 1

g = np.arange(n, dtype=np.uint64)
A6, A7, A10, A11 = (x.ravel() for x in np.meshgrid(g, g, g, g, indexing='ij'))
t0 = time.time()
K0, K1, K2, K3 = vs(v, A6, A7, A10, A11)
key = (K0 | (K1 << np.uint64(WD)) | (K2 << np.uint64(2 * WD))).astype(np.int64)
print(f"w={WD}: {A6.size:,} contexts, signature map computed in {time.time()-t0:.1f}s")

cnt = np.bincount(key, minlength=1 << (3 * WD))
occ = cnt[cnt > 0]
print(f"  distinct (KC0,KC1,KC2) reached: {occ.size:,} of {1<<(3*WD):,} "
      f"({occ.size/(1<<(3*WD)):.4f}; random-map prediction 1-1/e = 0.632 at load 1... "
      f"load here = {A6.size/(1<<(3*WD)):.1f})")
print(f"  fibre sizes: mean {occ.mean():.2f} (expected 2^w = {n}), "
      f"max {occ.max()}, min {occ.min()}; Poisson(2^w) sd {np.sqrt(n):.1f}, observed sd {occ.std():.1f}")

order = np.argsort(key, kind='stable')
ks, starts = np.unique(key[order], return_index=True)
ends = np.append(starts[1:], key.size)

t0 = time.time()
amort_hits = []          # (ctx index, a0 index within the shared sweep)
nsweeps = 0
for kk, s, e in zip(ks, starts, ends):
    KC0 = int(kk) & M; KC1 = (int(kk) >> WD) & M; KC2 = (int(kk) >> (2 * WD)) & M
    out = w.sweep(tbl, (KC0, KC1, KC2, v, 0), R)      # K3p = 0 -> kstar is what we need
    nsweeps += 1
    if out['kstar'].size == 0:
        continue
    L = {}
    for i, x in enumerate(out['kstar']):
        L.setdefault(int(x), []).append(i)
    for j in order[s:e]:
        k3 = int(K3[j])
        if k3 in L:
            for i in L[k3]:
                amort_hits.append((int(j), tuple(int(out['a'][t][i]) for t in range(4))))
print(f"  amortised pass: {nsweeps:,} sweeps ({nsweeps*n:,} swept a0) in {time.time()-t0:.1f}s"
      f"  -> {len(amort_hits)} preimages")

t0 = time.time()
brute = []
for j in range(A6.size):
    ctx = w.context(v, int(A6[j]), int(A7[j]), int(A10[j]), int(A11[j]))
    sig = (int(K0[j]), int(K1[j]), int(K2[j]), v, int(K3[j]))
    out = w.sweep(tbl, sig, R)
    z = np.nonzero(out['c3'] == 0)[0]
    for i in z:
        brute.append((j, tuple(int(out['a'][t][i]) for t in range(4))))
print(f"  brute-force pass: {A6.size:,} sweeps ({A6.size*n:,} swept a0) in {time.time()-t0:.1f}s"
      f"  -> {len(brute)} preimages")
print(f"  identical solution sets: {sorted(brute) == sorted(amort_hits)}")
print(f"  swept-a0 saving from amortisation: {A6.size/nsweeps:.1f}x  (= fibre size 2^w = {n})")

ok = 0
for j, a0123 in brute[:300]:
    ctx = w.context(v, int(A6[j]), int(A7[j]), int(A10[j]), int(A11[j]))
    Wm = rebuild_msg(w, h, ctx, R, a0123)
    ok += (w.digest(Wm, R) == h)
print(f"  verified by forward hashing: {ok}/{min(300,len(brute))}")
