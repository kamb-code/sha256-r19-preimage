"""Fibre structure of the context -> (KC0,KC1,KC2) map, and the amortisation
upper bound it would give if fibres were free.  Exhaustive at w = 8.

  free context words (v fixed): a6,a7,a10,a11 -> 4w bits
  signature coordinates fixed:  KC0,KC1,KC2   -> 3w bits
  fibre size (expected):        2^w
"""
import sys, time, numpy as np
sys.path.insert(0, '/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/lattice/mdtable')
from wmodel import W
from vecsig import make_vecsig
from t1_validate import rebuild_msg

WD = 8
w = W(WD); M = w.M
rng = np.random.default_rng(20260908)
tbl = w.build_table()
R = 20

msg = [int(x) for x in rng.integers(0, M + 1, 16, dtype=np.uint64)]
h = w.digest(msg, R)
vs = make_vecsig(w, h, R)

# --- cross-check the vectorised map against the scalar reference ---
bad = 0
for _ in range(200):
    v, a6, a7, a10, a11 = (int(x) for x in rng.integers(0, M + 1, 5, dtype=np.uint64))
    ref = w.signature(h, w.context(v, a6, a7, a10, a11), R)
    got = vs(v, *(np.array([x], dtype=np.uint64) for x in (a6, a7, a10, a11)))
    if (ref[0], ref[1], ref[2], ref[4]) != tuple(int(g[0]) for g in got):
        bad += 1
print(f"vectorised signature vs scalar reference: {200-bad}/200 {'OK' if bad==0 else 'FAIL'}")

# --- pick a target fibre and enumerate it exhaustively ---
v = int(rng.integers(0, M + 1, dtype=np.uint64))
a6t, a7t, a10t, a11t = (int(x) for x in rng.integers(0, M + 1, 4, dtype=np.uint64))
tgt = w.signature(h, w.context(v, a6t, a7t, a10t, a11t), R)
print(f"\nw={WD}, v=0x{v:02x}; target signature (KC0,KC1,KC2) = "
      f"({tgt[0]:#04x},{tgt[1]:#04x},{tgt[2]:#04x}), its kappa3 = {tgt[4]:#04x}")

g = np.arange(M + 1, dtype=np.uint64)
A6, A7, A10 = np.meshgrid(g, g, g, indexing='ij')
A6 = A6.ravel(); A7 = A7.ravel(); A10 = A10.ravel()
t0 = time.time()
members = []
for a11 in range(M + 1):
    A11 = np.full(A6.size, a11, dtype=np.uint64)
    K0, K1, K2, K3 = vs(v, A6, A7, A10, A11)
    hit = (K0 == tgt[0]) & (K1 == tgt[1]) & (K2 == tgt[2])
    idx = np.nonzero(hit)[0]
    for j in idx:
        members.append((int(A6[j]), int(A7[j]), int(A10[j]), a11, int(K3[j])))
scanned = (M + 1) ** 4
print(f"exhaustive scan of {scanned:,} contexts in {time.time()-t0:.1f}s: "
      f"fibre size {len(members)} (expected 2^w = {M+1})")
print(f"  brute-force cost per fibre member: 2^{np.log2(scanned/max(len(members),1)):.1f} "
      f"context evaluations (generic prediction 2^{3*WD})")

k3s = sorted(set(m[4] for m in members))
print(f"  distinct kappa3 over the fibre: {len(k3s)} of {M+1} possible")

# --- the exact structural claim: same signature => identical candidate list ---
base = w.sweep(tbl, tgt, R)
same = 0
for m in members[:40]:
    s = w.signature(h, w.context(v, m[0], m[1], m[2], m[3]), R)
    assert (s[0], s[1], s[2], s[3]) == (tgt[0], tgt[1], tgt[2], tgt[3])
    o = w.sweep(tbl, s, R)
    if (np.array_equal(o['a0'], base['a0']) and np.array_equal(o['kstar'], base['kstar'])
            and o['sol'] == base['sol']):
        same += 1
print(f"  candidate list identical across fibre members: {same}/{min(40,len(members))}")

# --- the amortised attack: one sweep, then read off every fibre member that works ---
L = {}                                   # kappa3 -> list of candidate indices
for i, ks in enumerate(base['kstar']):
    L.setdefault(int(ks), []).append(i)
print(f"\namortisation: one sweep of 2^{WD} a0 gives {base['eps0']} collapse survivors, "
      f"{len(L)} distinct solvable kappa3 values ({len(L)/(M+1):.3f} of the {M+1} possible)")
found = 0
for m in members:
    if m[4] in L:
        for i in L[m[4]]:
            a0123 = [base['a'][t][i] for t in range(4)]
            ctx = w.context(v, m[0], m[1], m[2], m[3])
            Wm = rebuild_msg(w, h, ctx, R, a0123)
            if w.digest(Wm, R) == h:
                found += 1
print(f"  20-round preimages obtained from that ONE sweep, over the fibre: {found}")
print(f"  (unamortised cost would be {len(members)} sweeps x 2^{WD} = 2^"
      f"{np.log2(len(members)*(M+1)):.1f} swept a0)")
