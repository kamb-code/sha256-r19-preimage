#!/usr/bin/env python3
"""Is per-context productivity heavy-tailed?  If so, screening contexts pays.

The attack samples a random context and then sweeps a0 over 2^32.  Every
context is given the same budget.  That is only optimal if contexts are
interchangeable.

If instead the number of C1/C2 fixed points per context is heavy-tailed -- if a
minority of contexts produce most of the fixed points -- then a cheap screening
pass (a short a0 sample per context, keep only the promising ones, then spend
the full sweep on those) beats uniform allocation.  That is a speedup for R=19
as it stands, and it applies unchanged at R=20.

The test: sample many contexts, give each a SHORT a0 budget, count deduplicated
fixed points, and compare the resulting distribution against the Poisson that
would hold if contexts were interchangeable.

    variance / mean == 1     Poisson: contexts interchangeable, screening is
                             worthless (all variation is counting noise)
    variance / mean >> 1     overdispersed: real context-to-context structure,
                             screening pays

We also report what fraction of all fixed points the top decile of contexts
holds, which is the directly actionable number: if the best 10% of contexts
hold 50% of the yield, screening at 10% cost roughly halves the work.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/administrator/sha/publish/code")
np.seterr(over="ignore")

from measure_w9_filters import (  # noqa: E402
    A_M1, A_M2, A_M3, C0_CONST, CONST_E0, E_M1, E_M2, E_M3, KU, MISS,
    ch, find_lo_seeds, maj, make_context, s0, s1, S0, S1, u32, w9g, _G,
)

LO = 0xFFFF


def score_context(sig, rng, nbatch, batch, iters, kseeds):
    """Deduplicated fixed points found for one context on a short a0 budget."""
    C = make_context(rng)
    return _score(sig, C, rng, nbatch, batch, iters, kseeds)


def score_context_fixed(sig, C, rng, nbatch, batch, iters, kseeds):
    """Score an ALREADY-BUILT context on a fresh a0 sample.

    Needed for the predictiveness test: the same context must be scored twice
    on independent a0 samples, to separate genuine context structure from
    per-sample noise.
    """
    return _score(sig, C, rng, nbatch, batch, iters, kseeds)


def _score(sig, C, rng, nbatch, batch, iters, kseeds):
    w9_init = w9g(C, np.uint32(0), np.uint32(0))
    seeds = find_lo_seeds(C, w9_init, kseeds, rng)
    conv = lo = 0
    for _ in range(nbatch):
        a0 = rng.integers(0, 1 << 32, size=batch, dtype=np.uint64).astype(np.uint32)
        e0 = a0 + CONST_E0
        W0 = a0 + C0_CONST
        gv = (u32(0) - S0(a0) - maj(a0, A_M1, A_M2) - E_M3
              - S1(e0) - ch(e0, E_M1, E_M2) - KU[1])
        F0 = C["W16r"] - s1(C["W14"]) - gv - w9_init - a0 - C0_CONST
        W1 = sig[F0]
        alive = W1 != MISS
        if not alive.any():
            continue
        a0a, gva, e0a, W0a, W1a = (x[alive] for x in (a0, gv, e0, W0, W1))
        a1a = W1a - gva
        e1a = A_M3 + a1a - S0(a0a) - maj(a0a, A_M1, A_M2)
        T2_2a = S0(a1a) + maj(a1a, a0a, A_M1)
        F12a = u32(0) - T2_2a - E_M2 - S1(e1a) - ch(e1a, e0a, E_M1) - KU[2]
        W16s = s1(C["W14"]) + w9_init + s0(W1a) + W0a - a1a
        for sa2, sa3 in seeds:
            a2c = np.full(a0a.size, sa2, dtype=np.uint32)
            a3c = np.full(a0a.size, sa3, dtype=np.uint32)
            seen = np.zeros(a0a.size, dtype=bool)
            for _it in range(iters):
                T16a = C["a6"] - C["S0a5"] - maj(C["a5"], C["a4"], a3c)
                Da3 = T16a + ch(C["e9"], C["e8"], a3c + C["T1_7"])
                T_c1 = C["W17r"] - s1(C["W15"]) - W1a - C["CONST_10"] - F12a + Da3
                W2v = sig[T_c1]
                h1 = W2v != MISS
                a2n = np.where(h1, W2v - F12a, a2c)
                e2v = A_M2 + a2n - T2_2a
                T2_3v = S0(a2n) + maj(a2n, a1a, a0a)
                F23v = u32(0) - T2_3v - A_M1 - S1(e2v) - ch(e2v, e1a, e0a) - KU[3]
                T_c2 = (C["W18r"] - s1(W16s) - C["W11_base"]
                        - np.where(h1, W2v, u32(0)) - F23v)
                W3v = sig[T_c2]
                h2 = W3v != MISS
                a3n = np.where(h2, W3v - F23v, a3c)
                both = h1 & h2
                cv = both & (a2n == a2c) & (a3n == a3c)
                fresh = cv & ~seen
                seen |= cv
                a2c, a3c = a2n, a3n
                if fresh.any():
                    idx = np.nonzero(fresh)[0]
                    conv += idx.size
                    resid = w9g(C, a2c[idx], a3c[idx]) - w9_init
                    lo += int(((resid & np.uint32(LO)) == 0).sum())
    return conv, lo


def worker(job):
    wid, nctx, nbatch, batch, iters, kseeds, seed = job
    sig = _G["sig"]
    out = []
    for i in range(nctx):
        rng = np.random.default_rng(seed + 1013 * i)
        out.append(score_context(sig, rng, nbatch, batch, iters, kseeds))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="/nvme0n1-disk/Kamvid/sigma0_u_table.npy")
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--contexts-per-job", type=int, default=20)
    ap.add_argument("--nbatch", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1 << 20)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--kseeds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=99001)
    a = ap.parse_args()

    print("loading table...", flush=True)
    _G["sig"] = np.load(a.table).view(np.uint32)
    import multiprocessing as mp
    jobs = [(w, a.contexts_per_job, a.nbatch, a.batch, a.iters, a.kseeds,
             a.seed + 100003 * w) for w in range(a.jobs)]
    t0 = time.time()
    with mp.get_context("fork").Pool(a.jobs) as pool:
        res = pool.map(worker, jobs)
    conv = np.array([c for r in res for c, _ in r], dtype=float)
    lo = np.array([l for r in res for _, l in r], dtype=float)
    n = conv.size
    a0_each = a.nbatch * a.batch

    print(f"\ncontexts sampled = {n}, a0 per context = {a0_each:,}  "
          f"({time.time()-t0:.0f}s)")
    for name, x in (("fixed points", conv), ("lo-pass events", lo)):
        m, v = x.mean(), x.var(ddof=1)
        print(f"\n{name}: mean {m:.1f}  var {v:.1f}  "
              f"variance/mean = {v/max(m,1e-9):.2f}")
        srt = np.sort(x)[::-1]
        top10 = srt[:max(1, n // 10)].sum() / max(srt.sum(), 1e-9)
        top25 = srt[:max(1, n // 4)].sum() / max(srt.sum(), 1e-9)
        print(f"  top 10% of contexts hold {top10*100:.1f}% of the yield")
        print(f"  top 25% of contexts hold {top25*100:.1f}% of the yield")
        print(f"  min {srt[-1]:.0f}  median {np.median(x):.0f}  max {srt[0]:.0f}")
        if v / max(m, 1e-9) > 2:
            gain = top10 / 0.10
            print(f"  => OVERDISPERSED: screening the top decile would give "
                  f"~{gain:.1f}x the yield per context attacked")
        else:
            print("  => consistent with Poisson; contexts are interchangeable "
                  "and screening does NOT pay")


if __name__ == "__main__":
    main()
