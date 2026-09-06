#!/usr/bin/env python3
"""Exhaustive enumeration of context assignments for the a5 -> C1 edge in the
w-bit model, using the verified reduction T_1 = c' - G(a5; a4,a6,a7,c8,c9; a3).

Modes
  lit W      literal exhaustive over ALL (a4,a6,a7,c8,c9) in (2^W)^5, i.e. every
             context assignment with a8..a11 free (a8,a9 <-> c8,c9 bijective;
             a10,a11 only shift c').  W=5: all a3, 2 c'.  W=6: 2 (a3,c') states.
  famA W     a4=a6=a7=v: all (v,c8,c9) in (2^W)^3, S states.       (W=8: 2^24)
  famB W     a7=a6, a4 free: all (a4^a6 mask, c8', c9) with a6&a4 random per state.
  kern W     bare kernel H = S0(a5) - S1(a5+c9) - Ch(a5+c9, X, Y), all (c9,X,Y),
             no Maj term, no c': avalanche and image size of a5 -> H.
  rand W N   N random general contexts (a4,a6,a7,c8,c9), S states.
Per context: mean Hamming weight of dT_1 per flipped bit of a5 over all a5, all
bits and S states; image size of a5 -> T_1 (min over states); max fibre.
Reports min/mean/max, the histogram, the fraction below thresholds, and the
lowest-weight contexts in full.
"""
import sys, json, time
import numpy as np
from multiprocessing import Pool
from wmodel import Model

MODE = sys.argv[1]
W = int(sys.argv[2])
NRAND = int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 20
SEED = 20260906
m = Model(W)
MM = m.M
NW = 1 << W
PC = np.array([bin(i).count('1') for i in range(NW)], dtype=np.int64)
A5 = np.arange(NW, dtype=np.int64)[None, :]
CHUNK = {5: 4096, 6: 8192, 8: 2048, 12: 128}.get(W, 512)


U = np.uint16
S0T = m.S0(np.arange(NW, dtype=np.int64)).astype(U)
S1T = m.S1(np.arange(NW, dtype=np.int64)).astype(U)
PC16 = PC.astype(np.int64)
A5U = A5.astype(U)
A5B = [(A5 ^ (1 << b)).astype(U) for b in range(W)]      # a5 with bit b flipped
S0A = [S0T[A5U]] + [S0T[x] for x in A5B]                  # S0 rows, base and per bit
MU = U(MM)


def G(a5, a4, a6, a7, c8, c9, a3):
    """reference (int64) form, used only for cross-checks."""
    c7 = (a3 + a7 - m.S0(a6)) & MM
    y = (a5 + c9) & MM
    X = (c8 - m.Maj(a7, a6, a5)) & MM
    Y = (c7 - m.Maj(a6, a5, a4)) & MM
    return (m.S0(a5) + m.Maj(a5, a4, a3) - m.S1(y) - m.Ch(y, X, Y)) & MM


def Gfast(k, _unused, p43, m43, p8, m8, c8, c7p, m7, c9):
    """G with precomputed per-context constants; k = 0 base, k = b+1 bit b flipped.
    p43 = a4&a3, m43 = a4^a3, p8 = a7&a6, m8 = a7^a6, c7p = c7 - (a6&a4), m7 = a6^a4."""
    a5u = A5U if k == 0 else A5B[k - 1]
    y = (a5u + c9) & MU
    X = (c8 - p8 - (m8 & a5u)) & MU
    Y = (c7p - (m7 & a5u)) & MU
    ch = Y ^ (y & (X ^ Y))
    maj = p43 + (m43 & a5u)
    return (S0A[k] + maj - S1T[y] - ch) & MU


def H(a5, c9, X, Y):
    y = (a5 + c9) & MM
    return (m.S0(a5) - m.S1(y) - m.Ch(y, X, Y)) & MM


def Hfast(k, _unused, c9, X, Y):
    a5u = A5U if k == 0 else A5B[k - 1]
    y = (a5u + c9) & MU
    return (S0A[k] - S1T[y] - (Y ^ (y & (X ^ Y)))) & MU


def image_stats(G0):
    s = np.sort(G0, axis=1)
    distinct = 1 + (np.diff(s, axis=1) != 0).sum(axis=1)
    # max fibre: longest run of equal values in each sorted row
    eq = (np.diff(s, axis=1) == 0).astype(np.int64)
    # run lengths via cumulative trick
    maxrun = np.zeros(s.shape[0], dtype=np.int64)
    run = np.zeros(s.shape[0], dtype=np.int64)
    for k in range(eq.shape[1]):
        run = (run + 1) * eq[:, k]
        maxrun = np.maximum(maxrun, run)
    return distinct, maxrun + 1


def params_from_index(idx):
    """context index -> parameter arrays (C,1) for the mode."""
    idx = np.asarray(idx, dtype=np.int64)
    if MODE == 'lit':
        c9 = idx & MM; c8 = (idx >> W) & MM; a7 = (idx >> (2 * W)) & MM
        a6 = (idx >> (3 * W)) & MM; a4 = (idx >> (4 * W)) & MM
    elif MODE == 'famA':
        c9 = idx & MM; c8 = (idx >> W) & MM; a4 = a6 = a7 = (idx >> (2 * W)) & MM
    elif MODE == 'famB':
        c9 = idx & MM; c8 = (idx >> W) & MM; mask = (idx >> (2 * W)) & MM
        a4 = mask; a6 = np.zeros_like(idx); a7 = a6      # a6&a4 bits randomised in worker
        return c9, c8, a7, a6, a4, mask
    elif MODE == 'kern':
        c9 = idx & MM; X = (idx >> W) & MM; Y = (idx >> (2 * W)) & MM
        return c9, X, Y
    return c9, c8, a7, a6, a4


def worker(args):
    lo, hi, seed = args
    rng = np.random.default_rng(seed)
    idx = np.arange(lo, hi, dtype=np.int64)
    C = idx.size
    if MODE == 'rand':
        c9, c8, a7, a6, a4 = (rng.integers(0, NW, size=C, dtype=np.int64) for _ in range(5))
        mask = None
    elif MODE == 'kernrand':
        c9, X, Y = (rng.integers(0, NW, size=C, dtype=np.int64) for _ in range(3))
    elif MODE == 'famB':
        c9, c8, a7, a6, a4, mask = params_from_index(idx)
    elif MODE == 'kern':
        c9, X, Y = params_from_index(idx)
    else:
        c9, c8, a7, a6, a4 = params_from_index(idx)
        mask = None
    if MODE in ('kern', 'kernrand'):
        c9u, Xu, Yu = (x[:, None].astype(U) for x in (c9, X, Y))
        c9, X, Y = c9[:, None], X[:, None], Y[:, None]
        H0 = Hfast(0, A5U, c9u, Xu, Yu)
        tot = np.zeros(C, dtype=np.int64)
        for b in range(W):
            Hb = Hfast(b + 1, A5U, c9u, Xu, Yu)
            tot += PC16[Hb ^ H0].sum(axis=1)
        distinct, maxf = image_stats(H0.astype(np.int64))
        mean = tot / (NW * W)
        return idx, mean, distinct, maxf, np.stack([c9[:, 0], X[:, 0], Y[:, 0]], axis=1)
    c9, c8, a7, a6, a4 = (x[:, None] for x in (c9, c8, a7, a6, a4))
    if MODE == 'lit' and W == 5:
        states = [(np.full((C, 1), a3v, dtype=np.int64), rng.integers(0, NW, size=(C, 1), dtype=np.int64))
                  for a3v in range(NW) for _ in range(2)]
    else:
        S = {6: 2, 8: 8, 12: 4}.get(W, 4)
        states = [(rng.integers(0, NW, size=(C, 1), dtype=np.int64), rng.integers(0, NW, size=(C, 1), dtype=np.int64))
                  for _ in range(S)]
    tot = np.zeros(C, dtype=np.int64)
    distinct = np.full(C, NW, dtype=np.int64); maxf = np.ones(C, dtype=np.int64)
    for a3, cp in states:
        if MODE == 'famB':
            # a7 = a6 with a6 ^ a4 = mask; the common bits a6 & a4 random per state
            common = rng.integers(0, NW, size=(C, 1), dtype=np.int64) & ~mask[:, None]
            a6s = common; a4s = (common ^ mask[:, None]) & MM; a7s = a6s
        else:
            a6s, a4s, a7s = a6, a4, a7
        c7p = (a3 + a7s - m.S0(a6s) - (a6s & a4s)) & MM
        cst = [x.astype(U) for x in ((a4s & a3), (a4s ^ a3), (a7s & a6s), (a7s ^ a6s), c8, c7p, (a6s ^ a4s), c9)]
        cpu = cp.astype(U)
        G0 = Gfast(0, A5U, *cst)
        T0 = (cpu - G0) & MU
        for b in range(W):
            Gb = Gfast(b + 1, A5U, *cst)
            Tb = (cpu - Gb) & MU
            tot += PC16[Tb ^ T0].sum(axis=1)
        d, f = image_stats(G0.astype(np.int64))
        distinct = np.minimum(distinct, d); maxf = np.maximum(maxf, f)
    mean = tot / (NW * W * len(states))
    prm = np.stack([a4[:, 0], a6[:, 0], a7[:, 0], c8[:, 0], c9[:, 0]] if MODE != 'famB'
                   else [mask, a6[:, 0], a7[:, 0], c8[:, 0], c9[:, 0]], axis=1)
    return idx, mean, distinct, maxf, prm


def main():
    t0 = time.time()
    if MODE == 'lit':
        total = NW ** 5
    elif MODE in ('famA', 'famB', 'kern'):
        total = NW ** 3
    else:
        total = NRAND
    jobs = [(lo, min(lo + CHUNK, total), SEED + (lo // CHUNK)) for lo in range(0, total, CHUNK)]
    print(f"mode {MODE} w={W}: {total:,} contexts in {len(jobs)} chunks", flush=True)
    hist = np.zeros(int(4 * W / 0.05) + 2, dtype=np.int64)     # bins of 0.05 bit
    keep = []            # (mean, distinct, maxf, params) lowest weights
    keep_img = []        # smallest images
    n = 0; s = 0.0; mn = 1e9; mx = -1; ss = 0.0
    below = {0.5: 0, 1.0: 0, 1.5: 0, 2.0: 0, 2.5: 0, 3.0: 0}
    with Pool(26) as p:
        for k, (idx, mean, distinct, maxf, prm) in enumerate(p.imap_unordered(worker, jobs, chunksize=4)):
            n += mean.size; s += mean.sum(); ss += (mean ** 2).sum()
            mn = min(mn, mean.min()); mx = max(mx, mean.max())
            hist += np.bincount(np.minimum((mean / 0.05).astype(np.int64), hist.size - 1), minlength=hist.size)
            for th in below:
                below[th] += int((mean < th).sum())
            order = np.argsort(mean)[:16]
            for i in order:
                keep.append((float(mean[i]), int(distinct[i]), int(maxf[i]), prm[i].tolist()))
            order = np.argsort(distinct)[:8]
            for i in order:
                keep_img.append((int(distinct[i]), int(maxf[i]), float(mean[i]), prm[i].tolist()))
            if len(keep) > 4000:
                keep.sort(); keep = keep[:64]
                keep_img.sort(); keep_img = keep_img[:64]
            if k % 500 == 0:
                print(f"  chunk {k}/{len(jobs)}: n={n:,} min {mn:.3f} mean {s/n:.3f} max {mx:.3f} "
                      f"below2={below[2.0]} {time.time()-t0:.0f}s", flush=True)
    keep.sort(); keep = keep[:64]
    keep_img.sort(); keep_img = keep_img[:64]
    mean_all = s / n; sd = np.sqrt(max(ss / n - mean_all ** 2, 0))
    print(f"\nDONE mode {MODE} w={W}: {n:,} contexts, {time.time()-t0:.0f}s")
    print(f"  weight (bits per flipped bit of a5 into C1): min {mn:.4f}  mean {mean_all:.4f}  sd {sd:.4f}  max {mx:.4f}")
    for th in sorted(below):
        print(f"  fraction below {th}: {below[th]}/{n} = {below[th]/n:.3e}")
    print("  lowest-weight contexts [mean, image size, max fibre, params]:")
    lab = {'kern': '(c9, X, Y)', 'kernrand': '(c9, X, Y)', 'famB': '(a4^a6 mask, -, -, c8, c9)'}.get(MODE, '(a4, a6, a7, c8, c9)')
    for r in keep[:12]:
        print(f"    {r[0]:.4f}  img {r[1]:4d}  fibre {r[2]:3d}  {lab} = {[hex(x) for x in r[3]]}")
    print("  smallest images [image size, max fibre, mean weight, params]:")
    for r in keep_img[:8]:
        print(f"    img {r[0]:4d}  fibre {r[1]:3d}  w {r[2]:.4f}  {lab} = {[hex(x) for x in r[3]]}")
    nz = np.nonzero(hist)[0]
    print("  histogram (bin lower edge in bits: count):", {f"{i*0.05:.2f}": int(hist[i]) for i in nz[:40]})
    json.dump(dict(mode=MODE, w=W, n=n, min=mn, mean=mean_all, sd=sd, max=mx, below=below,
                   lowest=keep, smallest_image=keep_img, hist=hist.tolist()),
              open(f"exhaust_{MODE}_w{W}.json", 'w'))


if __name__ == '__main__':
    main()
