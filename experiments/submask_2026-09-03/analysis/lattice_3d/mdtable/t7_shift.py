"""Falsify the recurring within-fibre shifts: does any fixed shift of the
context words preserve (KC0,KC1,KC2) more often than chance, on FRESH contexts?
A signature-preserving shift would make fibres walkable and break the barrier.
"""
import sys, numpy as np, collections
sys.path.insert(0, '/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/lattice/mdtable')
from wmodel import W
from vecsig import make_vecsig

WD = 6
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
big = np.argsort(-sizes)[:400]
add_ct = collections.Counter(); xor_ct = collections.Counter()
for b in big:
    idx = order[starts[b]:ends[b]]
    c = np.stack([A6[idx], A7[idx], A10[idx], A11[idx]]).T.astype(np.int64)
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            add_ct[tuple((c[i] - c[j]) & M)] += 1
            xor_ct[tuple(c[i] ^ c[j])] += 1

print("top recurring within-fibre shifts, retested on ALL 2^24 contexts:")
print(f"  chance rate for a random shift to preserve (KC0,KC1,KC2) = 2^-{3*WD} = {2.0**(-3*WD):.2e}")
for name, ct, op in (("additive", add_ct, 'add'), ("XOR", xor_ct, 'xor')):
    for shift, mult in ct.most_common(4):
        d = [np.uint64(x) for x in shift]
        if op == 'add':
            B = [(A + x) & M for A, x in zip((A6, A7, A10, A11), d)]
        else:
            B = [A ^ x for A, x in zip((A6, A7, A10, A11), d)]
        L0, L1, L2, _ = vs(v, *B)
        keep = (L0 == K0) & (L1 == K1) & (L2 == K2)
        r = float(keep.mean())
        print(f"  {name} {shift}: fibre multiplicity {mult}, "
              f"preserves signature on {int(keep.sum()):,}/{A6.size:,} contexts "
              f"= {r:.3e}  ({r/2.0**(-3*WD):.2f}x chance)")
