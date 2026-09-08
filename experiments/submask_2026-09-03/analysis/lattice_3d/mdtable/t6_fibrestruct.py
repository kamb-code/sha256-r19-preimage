"""Is the fibre of the context -> (KC0,KC1,KC2) map STRUCTURED?

If fibre members were related by a fixed shift (XOR or additive) on the context
words, or formed an affine GF(2) subspace, one could walk a fibre for free and
the amortisation would break the barrier.  Tested exhaustively at w=6 over all
2^24 contexts, and by sampling at w=8.

Null model: the map is a random map, fibres are unstructured sets of expected
size 2^w.
"""
import sys, numpy as np, collections
sys.path.insert(0, '/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/lattice/mdtable')
from wmodel import W
from vecsig import make_vecsig

WD = int(sys.argv[1]) if len(sys.argv) > 1 else 6
w = W(WD); M = w.M; R = 20
rng = np.random.default_rng(5)
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
sizes = ends - starts
print(f"w={WD}: {A6.size:,} contexts -> {ks.size:,} occupied fibres "
      f"(of {1<<(3*WD):,}); sizes mean {sizes.mean():.2f} max {sizes.max()} min {sizes.min()}")
# Poisson goodness of fit on fibre sizes
lam = A6.size / (1 << (3 * WD))
obs = np.bincount(sizes, minlength=int(sizes.max()) + 1).astype(float)
from math import lgamma, log, exp
kk = np.arange(obs.size)
exp_p = np.exp(kk * np.log(lam) - lam - np.array([lgamma(x + 1) for x in kk]))
expc = exp_p * (1 << (3 * WD))
m = (expc > 20) & (kk > 0)
chi2 = float(((obs[m] - expc[m]) ** 2 / expc[m]).sum())
print(f"  fibre-size histogram vs Poisson({lam:.1f}): chi2 = {chi2:.1f} on {int(m.sum())-1} df")

# --- recurring differences: a fixed shift that maps fibre to fibre ---
big = np.argsort(-sizes)[:400]
xor_ct = collections.Counter(); add_ct = collections.Counter()
pairs = 0
for b in big:
    idx = order[starts[b]:ends[b]]
    c = np.stack([A6[idx], A7[idx], A10[idx], A11[idx]]).T.astype(np.int64)
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            d = c[i] ^ c[j]
            xor_ct[tuple(d)] += 1
            add_ct[tuple((c[i] - c[j]) & M)] += 1
            pairs += 1
tot_shift = (n ** 4) - 1
print(f"  {pairs:,} within-fibre pairs over the {len(big)} largest fibres")
for name, ct in (("XOR", xor_ct), ("additive", add_ct)):
    top, mult = ct.most_common(1)[0]
    lam2 = pairs / tot_shift
    print(f"  most frequent {name} difference {top}: multiplicity {mult}"
          f"   (expected max under the null ~{max(1, int(3*lam2)) if lam2>1 else 1}, "
          f"mean per shift {lam2:.2e})")

# --- affine test: is any large fibre closed under XOR of differences? ---
closed = 0
for b in big[:50]:
    idx = order[starts[b]:ends[b]]
    S = set(zip(A6[idx].tolist(), A7[idx].tolist(), A10[idx].tolist(), A11[idx].tolist()))
    L = list(S)
    if len(L) < 3:
        continue
    p0 = L[0]
    hit = 0; tries = 0
    for i in range(1, min(len(L), 12)):
        for j in range(i + 1, min(len(L), 12)):
            q = tuple(p0[t] ^ L[i][t] ^ L[j][t] for t in range(4))
            tries += 1; hit += q in S
    if tries and hit == tries:
        closed += 1
print(f"  fibres closed under the affine XOR rule p0^pi^pj: {closed}/50 "
      f"(a GF(2)-affine fibre would give 50/50)")
