#!/usr/bin/env python3
"""relx: the strongest available test that the 4th constraint is a flat 2^-32.

The corpus has 68,680 values of c3 -- enough to bound a per-bit bias at ~6e-3.
But c3 = s0(W4) + W3 - K3p is defined for EVERY stage-3 survivor, not just the
Maj-passing ones, and those are ~1e4 times more numerous.  Evaluating c3 there
gives n ~ 1e8 and bounds a per-bit bias at ~1e-4.

That population is only a proxy for the attack's, so we ALSO test
    c3  _|_  k        (k = #agreeing bits of (a2,a3), the Maj-filter statistic)
If c3 is independent of k, the c3 distribution among Maj-passers is the same as
among all stage-3 survivors, and the 1e8-sample bound transfers.

Reported for the attack's real question:  P(c3 = 0) = density of the law of
s0(W4)+W3 at the context constant K3p, so a non-uniform law is exactly what
"some contexts are better than others" would mean.
"""
import sys, time
import numpy as np
from scipy.stats import chi2
from relx_lib import open_table, random_target, Ctx, run_a0, make_context

LOG = int(sys.argv[1]) if len(sys.argv) > 1 else 24
NCTX = int(sys.argv[2]) if len(sys.argv) > 2 else 12
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 606
R = 20
N = 1 << LOG
tbl = open_table("max")
rng = np.random.default_rng(SEED)
POP = np.array([bin(i).count("1") for i in range(1 << 16)], np.uint8)
t0 = time.time()

bitcnt = np.zeros(32, np.int64)
low16 = np.zeros(1 << 16, np.int64)
top16 = np.zeros(1 << 16, np.int64)
# joint of (k, c3 low 6 bits) for the independence test
kj = np.zeros((33, 64), np.int64)
tot = 0
maj_c3_lo = []
for ci in range(NCTX):
    h = random_target(rng, R)
    C = Ctx(h, make_context(rng, R), R)
    A0 = (np.arange(N, dtype=np.uint64) +
          int(rng.integers(0, 1 << 32))).astype(np.uint32)
    o = run_a0(tbl, C, A0, want_c3_all=True)
    c = o["_c3_all"]
    a2 = o["_a2_all"]; a3 = o["_a3_all"]
    ag = (~(a2 ^ a3)).astype(np.uint32)
    k = (POP[ag & 0xFFFF].astype(np.int64) +
         POP[(ag >> np.uint32(16))].astype(np.int64))
    for b in range(32):
        bitcnt[b] += int(((c >> np.uint32(b)) & 1).sum())
    low16 += np.bincount((c & np.uint32(0xFFFF)).astype(np.int64), minlength=1 << 16)
    top16 += np.bincount((c >> np.uint32(16)).astype(np.int64), minlength=1 << 16)
    kj += np.bincount(k * 64 + (c & np.uint32(63)).astype(np.int64),
                      minlength=33 * 64).reshape(33, 64)
    tot += len(c)
    maj_c3_lo.append(o["c3"])
    print(f"  ctx {ci+1}/{NCTX} tot={tot:,} [{time.time()-t0:.0f}s]", flush=True)

np.savez(f"relx11_c3deep_{LOG}_{NCTX}.npz", bitcnt=bitcnt, low16=low16,
         top16=top16, kj=kj, tot=tot, maj=np.concatenate(maj_c3_lo))

print(f"\n===== R=20, {NCTX} ctx x 2^{LOG} a0 -> {tot:,} stage-3 survivors"
      f"  [{time.time()-t0:.0f}s]")
p = bitcnt / tot
sd = 0.5 / np.sqrt(tot)
z = (p - 0.5) / sd
print(f"\n--- per-bit bias of c3 (n={tot:,}, sd={sd:.2e}, Bonferroni |z|>3.2)")
print("  z: " + " ".join(f"{x:+.1f}" for x in z))
print(f"  max|z| = {np.abs(z).max():.2f} at bit {int(np.argmax(np.abs(z)))};"
      f" chi2 {np.sum(z**2):.1f}/32 df p={chi2.sf(np.sum(z**2),32):.3f}")
print(f"  => every per-bit bias |p-1/2| < {3*sd:.2e} at 3 sigma"
      f" (corpus-only bound was 5.7e-03)")

for nm, hcnt in (("low 16 bits", low16), ("top 16 bits", top16)):
    e = tot / len(hcnt)
    X = ((hcnt - e) ** 2 / e).sum()
    df = len(hcnt) - 1
    zz = (X - df) / np.sqrt(2 * df)
    print(f"\n--- {nm}: chi2 = {X:.0f} on {df} df  (z = {zz:+.2f},"
          f" p = {chi2.sf(X, df):.4f});  cell 0 count {hcnt[0]}"
          f" exp {e:.1f} z={(hcnt[0]-e)/np.sqrt(e):+.2f}")
    phi = np.abs(np.fft.fft(hcnt.astype(float)) / tot)[1:]
    thr = np.sqrt(np.log(len(phi)) / tot)
    print(f"    max |characteristic function| = {phi.max():.6f};"
          f" null E[max] ~ {thr:.6f}; #>1.5*thr = {(phi>1.5*thr).sum()}")

print("\n--- is c3 independent of k (the Maj-filter statistic)?")
m = kj.sum(1) > 2000
sub = kj[m]
rows = sub.sum(1, keepdims=True); cols = sub.sum(0, keepdims=True); n2 = sub.sum()
e = rows * cols / n2
X = (((sub - e) ** 2) / e).sum()
df = (sub.shape[0] - 1) * (sub.shape[1] - 1)
print(f"  chi2 of independence (k rows with >2000 samples) = {X:.1f} on {df} df,"
      f" p = {chi2.sf(X, df):.4f}   [{sub.shape[0]} k-values x 64 c3 values]")
mi = 0.0
pj = sub / n2
px = pj.sum(1, keepdims=True); py = pj.sum(0, keepdims=True)
nz = pj > 0
mi = float((pj[nz] * np.log2(pj[nz] / (px @ py)[nz])).sum())
print(f"  MI(k ; c3 low 6 bits) = {mi:.7f} bits;"
      f" chance level {df/(2*n2*np.log(2)):.7f} bits")

maj = np.concatenate(maj_c3_lo)
print(f"\n--- sanity: the {len(maj):,} genuine Maj-passing candidates,"
      f" {int((maj==0).sum())} of them have c3 == 0"
      f" (expected {len(maj)/2**32:.2e})")
