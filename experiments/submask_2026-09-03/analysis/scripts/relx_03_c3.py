#!/usr/bin/env python3
"""relx: is the 4th-constraint residual c3 a flat 2^-32 filter?
Tests: per-bit bias, low-k-bit uniformity, per-context heterogeneity,
dependence on a0 / on context, and P(c3=0) implied by measured marginals."""
import numpy as np
from scipy.stats import chi2, norm, binomtest

SP = "/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/"
d = np.load(SP + "corpus_r20.npz")
c3 = d["c3"].astype(np.uint32)
ctx = d["ctx"]; a0 = d["a0"].astype(np.uint32)
W3 = d["W3"].astype(np.uint32); W4 = d["W4"].astype(np.uint32)
n = len(c3)
print(f"n = {n}, contexts = {ctx.max()+1}, zeros = {(c3==0).sum()}")

bits = ((c3[:, None] >> np.arange(32)[None, :]) & 1).astype(np.int8)

print("\n=== per-bit bias of c3  (sd = %.5f, Bonferroni 32 tests -> |z|>3.2)"
      % (0.5 / np.sqrt(n)))
p = bits.mean(0)
z = (p - 0.5) / (0.5 / np.sqrt(n))
worst = np.argsort(-np.abs(z))[:6]
for b in range(32):
    flag = " <<<" if abs(z[b]) > 3.2 else ""
    if abs(z[b]) > 2.0 or b in worst[:3]:
        print(f"  bit {b:2d}: p={p[b]:.5f}  z={z[b]:+6.2f}{flag}")
print(f"  max |z| = {np.abs(z).max():.2f} at bit {np.argmax(np.abs(z))}")
print(f"  chi2 over 32 bits = {np.sum(z**2):.1f} (df 32, p={chi2.sf(np.sum(z**2),32):.3f})")

print("\n=== pairwise bit correlations (496 tests, Bonferroni |z|>3.9)")
c = bits - 0.5
cov = (c.T @ c) / n
iu = np.triu_indices(32, 1)
r = cov[iu] / 0.25
zz = r * np.sqrt(n)
print(f"  max |z| = {np.abs(zz).max():.2f}; #|z|>3.9 = {(np.abs(zz)>3.9).sum()}"
      f" (expected {496*2*norm.sf(3.9):.2f})")
o = np.argsort(-np.abs(zz))[:5]
for k in o:
    print(f"    bits ({iu[0][k]:2d},{iu[1][k]:2d}) z={zz[k]:+.2f}")

print("\n=== low-k-bit uniformity of c3  (does P(c3 mod 2^k == 0) = 2^-k ?)")
for k in range(1, 14):
    m = 1 << k
    cnt = np.bincount(c3 & (m - 1), minlength=m)
    e = n / m
    X = ((cnt - e) ** 2 / e).sum()
    z0 = (cnt[0] - e) / np.sqrt(e)
    print(f"  k={k:2d}: chi2={X:9.1f} df={m-1:5d} p={chi2.sf(X,m-1):.4f}  "
          f"count[0]={cnt[0]:6d} exp={e:8.1f} z={z0:+.2f}")

print("\n=== high-bit / top-k uniformity (c3 >> (32-k))")
for k in range(1, 13):
    m = 1 << k
    cnt = np.bincount(c3 >> (32 - k), minlength=m)
    e = n / m
    X = ((cnt - e) ** 2 / e).sum()
    print(f"  k={k:2d}: chi2={X:9.1f} df={m-1:5d} p={chi2.sf(X,m-1):.4f}")

print("\n=== per-context heterogeneity: is any context's c3 distribution better?")
nc = ctx.max() + 1
# use low 8 bits: per-context chi2 vs uniform, and per-context mean of c3
rows = []
for c_ in range(nc):
    x = c3[ctx == c_]
    cnt = np.bincount(x & 255, minlength=256)
    e = len(x) / 256
    X = ((cnt - e) ** 2 / e).sum()
    rows.append((c_, len(x), X, chi2.sf(X, 255)))
ps = np.array([r[3] for r in rows])
print(f"  per-ctx chi2(low8) p-values: min={ps.min():.4f} (ctx {rows[int(np.argmin(ps))][0]}),"
      f" #p<0.05 = {(ps<0.05).sum()} (expected {0.05*nc:.1f})")
# KS-like: is the pooled distribution of p uniform?
print(f"  mean p = {ps.mean():.3f} (expect 0.5)")

# per-context bit bias: 40 ctx x 32 bits = 1280 tests
zc = np.zeros((nc, 32))
for c_ in range(nc):
    m = ctx == c_
    zc[c_] = (bits[m].mean(0) - 0.5) / (0.5 / np.sqrt(m.sum()))
print(f"  per-ctx per-bit: max|z| = {np.abs(zc).max():.2f}"
      f" (1280 tests, Bonferroni |z|>4.2); #>4.2 = {(np.abs(zc)>4.2).sum()}")

print("\n=== does c3 depend on a0 ?  (MI between a0 low/high bits and c3 low bits)")


def mi_bits(x, y, kx, ky):
    """MI in bits between low-kx bits of x and low-ky bits of y."""
    a = (x & ((1 << kx) - 1)).astype(np.int64)
    b = (y & ((1 << ky) - 1)).astype(np.int64)
    j = np.bincount(a * (1 << ky) + b, minlength=(1 << (kx + ky))).reshape(1 << kx, 1 << ky)
    tot = j.sum()
    pj = j / tot
    px = pj.sum(1, keepdims=True); py = pj.sum(0, keepdims=True)
    nzm = pj > 0
    return float((pj[nzm] * np.log2(pj[nzm] / (px @ py)[nzm])).sum())


# MI bias: for a KxL table with N samples, E[MI] ~ (K-1)(L-1)/(2 N ln2)
for kx, ky, lbl in [(4, 4, "a0 low4 vs c3 low4"), (6, 6, "a0 low6 vs c3 low6"),
                    (4, 4, "a0 HIGH4 vs c3 low4")]:
    xx = a0 if "HIGH" not in lbl else (a0 >> 28)
    m = mi_bits(xx, c3, kx, ky)
    e = (2**kx - 1) * (2**ky - 1) / (2 * n * np.log(2))
    print(f"  {lbl}: MI={m:.6f} bits, chance-level {e:.6f}")

print("\n=== c3 vs its algebraic parts (sanity: c3 = s0(W4)+W3-K3p)")
print("  per-bit bias of W3 (max representative -> expect strong bias):")
bw = ((W3[:, None] >> np.arange(32)[None, :]) & 1).mean(0)
print("   ", " ".join(f"{x:.3f}" for x in bw))
print("  per-bit bias of W4:")
bw4 = ((W4[:, None] >> np.arange(32)[None, :]) & 1).mean(0)
print("   ", " ".join(f"{x:.3f}" for x in bw4))
