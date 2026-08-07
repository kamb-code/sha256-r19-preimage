#!/usr/bin/env python3
"""Measure the W9 lo/hi filter pass rates that set the complexity constant mu.

WHAT THIS MEASURES
------------------
The R=19 attack accepts a candidate when the true W9 matches the provisional
W9 used in the C0 step.  That 32-bit condition is applied as two 16-bit
filters (lo then hi).  The per-context success rate is, unconditionally,

    mu  =  N * P(lo AND hi)          N = unique fixed points per context

which is NOT the same as

    mu_ind = N * P(lo) * P(hi)       (valid only if the halves are independent)

The distinction matters here: the iteration is deliberately seeded on
lo-consistency, so independence is precisely what should not be assumed.  This
script measures N, P(lo) and P(hi) directly and reports mu_ind as a clearly
labelled PROJECTION.  It reports the unconditional mu only when full 32-bit
matches are actually observed; at the sample sizes reachable so far that count
is zero, and the script prints a Poisson upper bound instead of a value.

It works by sweeping a0 over random contexts and, for every C1/C2 fixed point,
recording the residual

    r = (W9_true - W9_provisional) mod 2^32

then counting how often (r & 0xffff) == 0 and how often (r >> 16) == 0.
Under a uniform model each rate would be 2^-16; neither is.  The script
reports the measured enhancement over uniform for each half.

WHY THE FILTERS ARE NOT UNIFORM
-------------------------------
The C1/C2 iteration is started from W9-lo-consistent seed pairs, and it
preferentially converges to fixed points that also satisfy the W9 condition.
The effect acts on the full 32-bit residual, not only the low half, so BOTH
filters are enhanced.

A coarse-binned chi-squared test does NOT detect this.  The enhancement sits
at a single residual value out of 2^16, and grouping into 256 equal-width bins
dilutes it by roughly an order of magnitude.  This script therefore measures at
exact residual resolution.  It also prints the 256-bin histogram so the
dilution can be seen directly: the exact-value enhancement is far larger than
the enhancement visible at bin resolution.

FIXED POINTS ARE COUNTED ONCE
-----------------------------
A lane that reaches a fixed point stays there for every remaining iteration.
Counting `converged` once per iteration therefore overcounts each fixed point
by roughly the number of iterations remaining after it converged (measured:
about 6x at 8 iterations).  Rates expressed per convergence are insensitive to
this, but any absolute count of fixed points per context is not, and mu depends
on that count.  This script deduplicates: each fixed point is counted once, and
it also reports the repeated count and the resulting inflation factor so the
two conventions can be compared.

USAGE
-----
    python3 measure_w9_filters.py --table /path/to/sigma0_u_table.npy \\
                                  --jobs 14 --nbatch 400

The sigma0(u)-u table is 16 GiB and must be resident in RAM; the workers are
forked so they share it copy-on-write.  A run of --nbatch 400 with 14 workers
sweeps about 5.9e9 a0 values and takes roughly an hour on a 28-core machine.

Statistics are noisy: context-to-context variance in the convergence count is
about a factor of two, and the hi-pass count is the limiting statistic.  Report
mu with an interval, not a point value.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# resolve sibling modules regardless of where the script is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extended_solver import backward_chain, compute_e_from_a, recover_W  # noqa: E402
from sha256_core import H0, K  # noqa: E402

# All arithmetic here is deliberately mod 2^32: uint32 wraparound IS the
# semantics of the SHA-256 round function, so NumPy's overflow warnings are
# expected rather than diagnostic.
np.seterr(over="ignore")

WIDTHS = (4, 6, 8, 12)      # half-widths at which lo/hi independence is tested

MISS = np.uint32(0xFFFFFFFF)
M = 0xFFFFFFFF
LO = 0xFFFF


def u32(x): return np.uint32(x & M)
def rotr(x, n): return (x >> np.uint32(n)) | (x << np.uint32(32 - n))
def S0(x): return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
def S1(x): return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
def s0(x): return rotr(x, 7) ^ rotr(x, 18) ^ (x >> np.uint32(3))
def s1(x): return rotr(x, 17) ^ rotr(x, 19) ^ (x >> np.uint32(10))
def ch(e, f, g): return (e & f) ^ (~e & g)
def maj(a, b, c): return (a & b) ^ (a & c) ^ (b & c)


A_M1, A_M2, A_M3, A_M4 = (u32(H0[i]) for i in range(4))
E_M1, E_M2, E_M3, E_M4 = (u32(H0[i]) for i in range(4, 8))
KU = [u32(k) for k in K]
T2_0IV = S0(A_M1) + maj(A_M1, A_M2, A_M3)
C0_CONST = u32(0) - T2_0IV - E_M4 - S1(E_M1) - ch(E_M1, E_M2, E_M3) - KU[0]
CONST_E0 = A_M4 - T2_0IV

_G = {}


def make_context(rng):
    """Random target hash + random free context words a[4..10]."""
    hb = os.urandom(32)
    ka = dict(backward_chain(hb, 19)[0])
    for r in range(4, 19):
        if r not in ka:
            ka[r] = int.from_bytes(os.urandom(4), "big")
    a4, a5, a6, a7, a8, a9, a10, a11 = (u32(ka[i]) for i in range(4, 12))
    T1_7 = a7 - (S0(a6) + maj(a6, a5, a4))
    T1_8 = a8 - (S0(a7) + maj(a7, a6, a5)); e8 = a4 + T1_8
    T1_9 = a9 - (S0(a8) + maj(a8, a7, a6)); e9 = a5 + T1_9
    T1_10 = a10 - (S0(a9) + maj(a9, a8, a7))
    T1_11 = a11 - (S0(a10) + maj(a10, a9, a8))
    e10 = a6 + T1_10
    C = dict(a4=a4, a5=a5, a6=a6, e8=e8, e9=e9, T1_7=T1_7,
             CONST_10=T1_10 - S1(e9) - KU[10],
             W11_base=T1_11 - T1_7 - S1(e10) - ch(e10, e9, e8) - KU[11],
             W9_base=T1_9 - S1(e8) - KU[9],
             S0a4=S0(a4), S0a5=S0(a5))
    ka0 = dict(ka)
    for r in (0, 1, 2, 3):
        ka0[r] = 0
    kW = recover_W(ka0, compute_e_from_a(ka0, 19), 19)
    C.update(W14=u32(kW.get(14, 0)), W15=u32(kW.get(15, 0)),
             W16r=u32(kW.get(16, 0)), W17r=u32(kW.get(17, 0)),
             W18r=u32(kW.get(18, 0)))
    return C


def w9g(C, a2, a3):
    t16 = C["a6"] - C["S0a5"] - maj(C["a5"], C["a4"], a3)
    e6 = a2 + t16
    e7 = a3 + C["T1_7"]
    t15 = C["a5"] - C["S0a4"] - maj(C["a4"], a3, a2)
    return C["W9_base"] - t15 - ch(C["e8"], e7, e6)


def find_lo_seeds(C, w9_init, k, rng, pool=1 << 22):
    """The production K_seeds: (a2,a3) pairs already lo-consistent with W9."""
    if k <= 0:
        return [(np.uint32(0), np.uint32(0))]
    tgt = np.uint32(int(w9_init) & LO)
    out, tries = [], 0
    while len(out) < k and tries < 8:
        a2 = rng.integers(0, 1 << 32, size=pool, dtype=np.uint64).astype(np.uint32)
        a3 = rng.integers(0, 1 << 32, size=pool, dtype=np.uint64).astype(np.uint32)
        for i in np.nonzero((w9g(C, a2, a3) & np.uint32(LO)) == tgt)[0][: k - len(out)]:
            out.append((a2[i], a3[i]))
        tries += 1
    while len(out) < k:
        out.append((np.uint32(0), np.uint32(0)))
    return out


def sweep(job):
    wid, nbatch, batch, iters, kseeds, seed = job
    sig = _G["sig"]
    rng = np.random.default_rng(seed)
    C = make_context(rng)
    w9_init = w9g(C, np.uint32(0), np.uint32(0))
    seeds = find_lo_seeds(C, w9_init, kseeds, rng)

    conv_n = conv_rep = lo_n = hi_n = full_n = 0
    hist = np.zeros(256, dtype=np.int64)
    joint = np.zeros((len(WIDTHS), 3), dtype=np.int64)   # N_Lk, N_Hk, N_LHk
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
        W16s_corr = s1(C["W14"]) + w9_init + s0(W1a) + W0a - a1a

        # Collect (a0,a2,a3) of every convergence in this batch, across ALL
        # seed chains, then deduplicate globally: the same mathematical fixed
        # point can be reached on different iterations, from different seeds,
        # or via duplicate table representatives, and must be counted once.
        hits_a0, hits_a2, hits_a3 = [], [], []
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
                T_c2 = (C["W18r"] - s1(W16s_corr) - C["W11_base"]
                        - np.where(h1, W2v, u32(0)) - F23v)
                W3v = sig[T_c2]
                h2 = W3v != MISS
                a3n = np.where(h2, W3v - F23v, a3c)
                both = h1 & h2
                conv = both & (a2n == a2c) & (a3n == a3c)
                fresh = conv & ~seen          # each fixed point counted ONCE
                seen |= conv
                a2c, a3c = a2n, a3n
                if conv.any():
                    conv_rep += int(conv.sum())
                if fresh.any():
                    idx = np.nonzero(fresh)[0]
                    hits_a0.append(a0a[idx]); hits_a2.append(a2c[idx])
                    hits_a3.append(a3c[idx])

        if hits_a0:
            ha0 = np.concatenate(hits_a0); ha2 = np.concatenate(hits_a2)
            ha3 = np.concatenate(hits_a3)
            trip = np.stack([ha0, ha2, ha3], axis=1)
            trip = np.unique(trip, axis=0)          # global dedup
            conv_n += trip.shape[0]
            resid = w9g(C, trip[:, 1], trip[:, 2]) - w9_init
            lo_res = (resid & np.uint32(LO)).astype(np.int64)
            hi_res = ((resid >> np.uint32(16)) & np.uint32(LO)).astype(np.int64)
            hist += np.bincount(lo_res >> 8, minlength=256)
            lo_n += int((lo_res == 0).sum())
            hi_n += int((hi_res == 0).sum())
            full_n += int((resid == 0).sum())
            # joint counts at narrower widths, to test lo/hi INDEPENDENCE
            # where joint events are actually observable
            for wi, k in enumerate(WIDTHS):
                m = (1 << k) - 1
                lk = (lo_res & m) == 0
                hk = (hi_res & m) == 0
                joint[wi, 0] += int(lk.sum())
                joint[wi, 1] += int(hk.sum())
                joint[wi, 2] += int((lk & hk).sum())
    return conv_n, lo_n, hi_n, full_n, hist, conv_rep, joint


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--table", default="sigma0_u_table.npy",
                   help="path to the 16 GiB sigma0(u)-u table (.npy, int32)")
    p.add_argument("--jobs", type=int, default=os.cpu_count() // 2 or 1)
    p.add_argument("--nbatch", type=int, default=400)
    p.add_argument("--batch", type=int, default=1 << 20)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--kseeds", type=int, default=4,
                   help="production uses 4; 0 gives a cold (0,0) start")
    p.add_argument("--seed", type=int, default=20260730)
    a = p.parse_args()

    if not Path(a.table).exists():
        sys.exit(f"table not found: {a.table}\n"
                 f"Pass --table with the path to sigma0_u_table.npy.")
    print(f"Loading {a.table} into RAM (16 GiB)...", flush=True)
    t0 = time.time()
    _G["sig"] = np.load(a.table).view(np.uint32)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    import multiprocessing as mp
    jobs = [(w, a.nbatch, a.batch, a.iters, a.kseeds, a.seed + 7919 * w)
            for w in range(a.jobs)]
    total = a.jobs * a.nbatch * a.batch
    print(f"K_seeds={a.kseeds}: {a.jobs} workers x {a.nbatch} x {a.batch:,} "
          f"= {total:,} a0 ({total/2**32:.2f} context-equivalents)", flush=True)

    t0 = time.time()
    with mp.get_context("fork").Pool(a.jobs) as pool:
        res = pool.map(sweep, jobs)
    conv = sum(r[0] for r in res); lo = sum(r[1] for r in res)
    hi = sum(r[2] for r in res); full = sum(r[3] for r in res)
    hist = sum(r[4] for r in res); conv_rep = sum(r[5] for r in res)
    joint = sum(r[6] for r in res)
    ctx = total / 2**32

    print(f"\nelapsed {time.time()-t0:.0f}s")
    print(f"fixed points (deduplicated)   = {conv:,}")
    print(f"fixed points (repeat-counted) = {conv_rep:,}")
    print(f"  inflation from repeat count = {conv_rep/max(conv,1):.2f}x")
    print(f"lo-pass events                = {lo:,}")
    print(f"hi-pass events                = {hi:,}")
    print(f"full 32-bit W9 matches        = {full:,}")
    if conv and lo and hi:
        pl, ph = lo / conv, hi / conv
        print(f"\nP(lo) = {pl:.3e}   enhancement {pl/2**-16:6.1f}x over 2^-16")
        print(f"P(hi) = {ph:.3e}   enhancement {ph/2**-16:6.1f}x over 2^-16")
        print(f"fixed points per context = {conv/ctx:,.0f}")

        print("\nLO/HI INDEPENDENCE (D = P(both)/[P(lo)P(hi)]; D=1 is independent)")
        print(f"  {'width':>6} {'N_Lk':>9} {'N_Hk':>9} {'N_LHk':>8} {'D':>8}")
        for wi, k in enumerate(WIDTHS):
            nl, nh, nlh = joint[wi]
            if nl and nh:
                d = (nlh / conv) / ((nl / conv) * (nh / conv))
                err = f"+/-{d/max(nlh,1)**0.5:.2f}" if nlh else "  n/a"
                print(f"  {k:>4}b {nl:>9,} {nh:>9,} {nlh:>8,} {d:>8.2f} {err}")

        print(f"\nDIRECT estimator (the one that belongs in the paper):")
        print(f"  full 32-bit matches N_LH = {full}")
        print(f"  context-equivalents C    = {ctx:.2f}")
        if full:
            mu_d = full / ctx
            print(f"  mu = N_LH/C = {mu_d:.3e}  -> {1/mu_d:.0f} contexts")
        else:
            ub = 3.0 / ctx
            print(f"  mu = 0 observed; 95% upper bound = {ub:.3e} "
                  f"-> at least {1/ub:.0f} contexts")
            print(f"  NOT ENOUGH DATA for a direct estimate; the derived value")
            print(f"  p_lo*p_hi*N = {pl*ph*(conv/ctx):.3e} assumes independence,")
            print(f"  which the table above tests. Scale up C until N_LH > 0.")
        exp = hist.sum() / 256
        print(f"\nlo-residual 256-bin histogram: bin0 = {hist[0]:,} vs mean "
              f"{exp:,.1f}  ({hist[0]/max(exp,1e-9):.1f}x at BIN resolution,"
              f" vs {pl/2**-16:.0f}x at EXACT resolution)")
        print("  -> this gap is why a 256-bin chi-squared test is underpowered")


if __name__ == "__main__":
    main()
