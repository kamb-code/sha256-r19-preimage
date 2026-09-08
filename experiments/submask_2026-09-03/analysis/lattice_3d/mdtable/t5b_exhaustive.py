"""Exhaustive amortisation test at w=6, with an independent cross-check.

All 2^24 contexts (v fixed).  The amortised route does ONE sweep per distinct
(KC0,KC1,KC2) -- 2^18 sweeps -- and answers every context by testing whether its
kappa3 is in that sweep's list.  Brute force does one sweep per context, 2^24.
Cross-checked on a random sample of contexts, and every sampled preimage is
verified by hashing the message forward.
"""
import sys, time, numpy as np
sys.path.insert(0, '/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/lattice/mdtable')
from wmodel import W
from vecsig import make_vecsig
from t1_validate import rebuild_msg

WD = 6
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
K0, K1, K2, K3 = vs(v, A6, A7, A10, A11)
key = (K0 | (K1 << np.uint64(WD)) | (K2 << np.uint64(2 * WD))).astype(np.int64)
order = np.argsort(key, kind='stable')
ks, starts = np.unique(key[order], return_index=True)
ends = np.append(starts[1:], key.size)

t0 = time.time()
hits = {}                       # context index -> list of (a0,a1,a2,a3)
for kk, s, e in zip(ks, starts, ends):
    KC0 = int(kk) & M; KC1 = (int(kk) >> WD) & M; KC2 = (int(kk) >> (2 * WD)) & M
    out = w.sweep(tbl, (KC0, KC1, KC2, v, 0), R)
    if out['kstar'].size == 0:
        continue
    L = {}
    for i, x in enumerate(out['kstar']):
        L.setdefault(int(x), []).append(i)
    for j in order[s:e]:
        for i in L.get(int(K3[j]), ()):
            hits.setdefault(int(j), []).append(tuple(int(out['a'][t][i]) for t in range(4)))
nh = sum(len(x) for x in hits.values())
print(f"amortised: {ks.size:,} sweeps ({ks.size*n:,} swept a0) in {time.time()-t0:.0f}s "
      f"-> {nh:,} preimages in {len(hits):,} of {A6.size:,} contexts")
print(f"  predicted: 2^24 * c^3 * (3/4)^w = {A6.size * (tbl!=M+1).mean()**3 * 0.75**WD:,.0f}")
print(f"  brute force would need {A6.size:,} sweeps: saving {A6.size/ks.size:.0f}x = fibre size 2^w")

# --- independent cross-check on a random sample of contexts ---
rs = np.random.default_rng(99)
sample = rs.choice(A6.size, 40000, replace=False)
bad = 0; nb = 0
for j in sample:
    j = int(j)
    sig = (int(K0[j]), int(K1[j]), int(K2[j]), v, int(K3[j]))
    out = w.sweep(tbl, sig, R)
    z = np.nonzero(out['c3'] == 0)[0]
    got = sorted(tuple(int(out['a'][t][i]) for t in range(4)) for i in z)
    nb += len(got)
    if got != sorted(hits.get(j, [])):
        bad += 1
print(f"cross-check on 40,000 contexts swept individually: {nb} preimages, "
      f"{40000-bad}/40000 contexts agree with the amortised route "
      f"{'OK' if bad == 0 else 'FAIL'}")

# --- verify by forward hashing ---
ok = 0; tot = 0
for j in sample[:3000]:
    j = int(j)
    for a0123 in hits.get(int(j), []):
        ctx = w.context(v, int(A6[j]), int(A7[j]), int(A10[j]), int(A11[j]))
        Wm = rebuild_msg(w, h, ctx, R, a0123)
        tot += 1; ok += (w.digest(Wm, R) == h)
print(f"forward-hash verification of the sampled preimages: {ok}/{tot} "
      f"{'OK' if ok == tot else 'FAIL'}")
