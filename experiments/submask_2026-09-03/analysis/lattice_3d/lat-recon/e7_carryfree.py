"""E7: the only remaining lattice-shaped hope -- restrict the unknown to the set
on which sigma0 is EXACTLY Z-linear, so a word-level lattice would apply.

sigma0(x) = ROTR_a(x) ^ ROTR_b(x) ^ SHR_c(x).  If at every bit position at most
one of the three source bits is set, the XOR equals the integer sum and sigma0
restricted to that set is Z-linear (with the two rotations split at the wrap).
Question: how much entropy survives?  A word-level lattice needs the surviving
entropy on a0..a3 to exceed the 32 bits that C3 costs.
"""
import json, math
import numpy as np
from wsha import mk, PARAMS


def carryfree_count(w):
    a, b, c = PARAMS[w]['s0']
    x = np.arange(1 << w, dtype=np.int64)
    def rotr(v, n):
        n %= w
        return ((v >> n) | (v << (w - n))) & ((1 << w) - 1)
    A = rotr(x, a); B = rotr(x, b); C = x >> c
    ok = ((A & B) == 0) & ((A & C) == 0) & ((B & C) == 0)
    n = int(ok.sum())
    return dict(w=w, count=n, bits=round(math.log2(max(n, 1)), 3),
                bits_per_word_bit=round(math.log2(max(n, 1)) / w, 4),
                fraction=n / (1 << w))


if __name__ == "__main__":
    out = []
    for w in (8, 12, 16):
        r = carryfree_count(w); print(json.dumps(r), flush=True); out.append(r)
    # w = 32 by exhaustive count in chunks
    a, b, c = PARAMS[32]['s0']; Mw = 0xFFFFFFFF
    tot = 0
    CH = 1 << 24
    for base in range(0, 1 << 32, CH):
        x = (np.arange(base, base + CH, dtype=np.uint64) & Mw).astype(np.uint32)
        A = ((x >> np.uint32(a)) | (x << np.uint32(32 - a)))
        B = ((x >> np.uint32(b)) | (x << np.uint32(32 - b)))
        C = (x >> np.uint32(c))
        tot += int((((A & B) == 0) & ((A & C) == 0) & ((B & C) == 0)).sum())
    r = dict(w=32, count=tot, bits=round(math.log2(max(tot, 1)), 3),
             bits_per_word_bit=round(math.log2(max(tot, 1)) / 32, 4),
             fraction=tot / 2 ** 32)
    print(json.dumps(r), flush=True); out.append(r)
    json.dump(out, open("e7_carryfree.json", "w"), indent=1)
