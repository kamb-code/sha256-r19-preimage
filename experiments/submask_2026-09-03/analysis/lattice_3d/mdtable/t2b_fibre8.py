"""Second width (w=8): sample contexts, bucket by (KC0,KC1,KC2), and show that
the biggest bucket shares one candidate list, so one sweep answers all of them.
"""
import sys, time, numpy as np
sys.path.insert(0, '/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/lattice/mdtable')
from wmodel import W
from vecsig import make_vecsig
from t1_validate import rebuild_msg

WD = 8
w = W(WD); M = w.M; R = 20
rng = np.random.default_rng(4242)
tbl = w.build_table()
msg = [int(x) for x in rng.integers(0, M + 1, 16, dtype=np.uint64)]
h = w.digest(msg, R)
vs = make_vecsig(w, h, R)
v = int(rng.integers(0, M + 1, dtype=np.uint64))

N = 1 << 26
t0 = time.time()
A = [rng.integers(0, M + 1, N, dtype=np.uint64) for _ in range(4)]
K0, K1, K2, K3 = vs(v, *A)
key = (K0 | (K1 << np.uint64(WD)) | (K2 << np.uint64(2 * WD))).astype(np.int64)
cnt = np.bincount(key, minlength=1 << (3 * WD))
print(f"w=8: {N:,} random contexts -> load {N/(1<<(3*WD)):.1f} per signature, "
      f"max bucket {cnt.max()}, in {time.time()-t0:.0f}s")
print(f"  cost of one extra fibre member by brute-force sampling: 2^{3*WD} context "
      f"evaluations; one sweep costs 2^{WD} -- the deficit is 2^{2*WD}")

top = int(np.argmax(cnt))
idx = np.nonzero(key == top)[0]
KC0 = top & M; KC1 = (top >> WD) & M; KC2 = (top >> (2 * WD)) & M
base = w.sweep(tbl, (KC0, KC1, KC2, v, 0), R)
print(f"  largest bucket: {idx.size} contexts sharing (KC0,KC1,KC2) = "
      f"({KC0:#04x},{KC1:#04x},{KC2:#04x})")
same = 0
for j in idx:
    s = w.signature(h, w.context(v, int(A[0][j]), int(A[1][j]), int(A[2][j]), int(A[3][j])), R)
    o = w.sweep(tbl, (s[0], s[1], s[2], s[3], 0), R)
    same += (np.array_equal(o['a0'], base['a0']) and np.array_equal(o['kstar'], base['kstar']))
print(f"  identical candidate list across the bucket: {same}/{idx.size}")
L = {}
for i, x in enumerate(base['kstar']):
    L.setdefault(int(x), []).append(i)
print(f"  one sweep of 2^{WD} a0 -> {base['eps0']} collapse survivors, "
      f"{len(L)} solvable kappa3 of {M+1} ({len(L)/(M+1):.3f})")
found = 0
for j in idx:
    for i in L.get(int(K3[j]), ()):
        ctx = w.context(v, int(A[0][j]), int(A[1][j]), int(A[2][j]), int(A[3][j]))
        Wm = rebuild_msg(w, h, ctx, R, [base['a'][t][i] for t in range(4)])
        found += (w.digest(Wm, R) == h)
print(f"  verified 20-round preimages read off that ONE sweep: {found} "
      f"(expected {idx.size*len(L)/(M+1):.2f})")
