"""refQ_corpus.py -- what, if anything, do the two corpora say about this claim?
(a) confirm the corpus W1,W2,W3 carry the max-representative signature (so the
    'quarter of solutions' constant is forced by the known table policy);
(b) confirm the measured per-sweep survival rate distinguishes p^3 from
    (1-1/e)^3 (from lin_K_seq_big.log), and size the effect;
(c) check whether the image anomaly can touch the fourth-constraint residual at
    all: is c3 (R=20) in any way tied to W1/W2/W3 root multiplicity class?
"""
import numpy as np, sys
from math import exp, log

D = "/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/"
r19 = np.load(D + "corpus_r19.npz")
r20 = np.load(D + "corpus_r20.npz")
M = np.uint32(0xFFFFFFFF)


def rotr(x, n):
    x = x.astype(np.uint32)
    return ((x >> np.uint32(n)) | (x << np.uint32(32 - n))).astype(np.uint32)


def s0(x):
    return (rotr(x, 7) ^ rotr(x, 18) ^ (x.astype(np.uint32) >> np.uint32(3))).astype(np.uint32)


print("(a) max-representative signature in the corpus message words")
print("    exact bit means of ALL stored max representatives (from lin_N_table.log, bits 24..31):")
print("      0.501086 0.501931 0.503720 0.507737 0.515409 0.530850 0.561988 0.622566")
for nm, C in (("r19", r19), ("r20", r20)):
    for w in ("W1", "W2", "W3"):
        x = C[w].astype(np.uint32)
        hi = [float(((x >> np.uint32(b)) & 1).mean()) for b in range(24, 32)]
        print(f"    {nm} {w}: bits24..31 mean " + " ".join(f"{h:.4f}" for h in hi) + f"   n={len(x)}")
    x = C["W0"].astype(np.uint32)   # W0 is not a table root; control
    hi = [float(((x >> np.uint32(b)) & 1).mean()) for b in range(24, 32)]
    print(f"    {nm} W0 (control, not a root): " + " ".join(f"{h:.4f}" for h in hi))

print("\n(b) does the measured sweep survival separate p^3 from (1-1/e)^3?")
p = 2721603628 / 2 ** 32
kept = [2134635, 2133084, 2134913, 2134095, 2136547, 2135535, 2135402, 2134168, 2132789, 2136894]
n = 8388608
tot_k, tot_n = sum(kept), n * len(kept)
ph = tot_k / tot_n
se = (ph * (1 - ph) / tot_n) ** 0.5
print(f"    lin_K_seq_big: {tot_k}/{tot_n} = {ph:.6f} +- {se:.6f}")
for nm, q in (("p^3 (exact image)", p ** 3), ("(1-1/e)^3 (random map)", (1 - exp(-1)) ** 3)):
    print(f"      {nm:24s} = {q:.6f}   z = {(ph-q)/se:+.2f}")
print("    -> the exact constant is the right one; the random-map constant is excluded.")
print(f"    size of the correction: {log(p**3/(1-exp(-1))**3,2):+.5f} bits out of the ~45.4 bits of R=20 cost")

print("\n(c) can the image anomaly touch the fourth constraint?")
c3 = r20["c3"].astype(np.uint32)
print(f"    r20 residual c3: n={len(c3)}, zeros={int((c3==0).sum())}, mean={c3.mean():.4g}")
# the image/multiplicity structure is a property of the TABLE ONLY: it is a single
# global constant, identical for every context, every a0 and every candidate.
# Test the only corpus-visible proxy: is c3 correlated with the low-order structure
# of the roots (which is what multiplicity classes are built from)?
for w in ("W1", "W2", "W3"):
    x = r20[w].astype(np.uint32)
    # bucket by top 4 bits (max-representative bias lives there) and look at the
    # residual's per-bit means; report the largest deviation seen
    b = (x >> np.uint32(28)).astype(int)
    worst = 0.0
    for g in range(16):
        m = b == g
        if m.sum() < 500:
            continue
        for bit in range(32):
            f = float(((c3[m] >> np.uint32(bit)) & 1).mean())
            worst = max(worst, abs(f - 0.5) / (0.5 / np.sqrt(m.sum())))
    print(f"    {w}: max |z| of a residual bit mean over 16 top-nibble buckets x 32 bits = {worst:.2f}"
          f"  (512 tests, Bonferroni 5% threshold z={3.9:.1f})")
