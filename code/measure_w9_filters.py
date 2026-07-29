#!/usr/bin/env python3
"""Test the claimed 36.5x W9-lo enhancement that underpins mu = 5.3e-4.

The paper's complexity chain is:

    62,745 C1/C2 convergences per context
    -> 35 lo-pass events per context          (claimed 36.5x above the
                                               uniform prediction of 0.96)
    -> mu = 35 * 2^-16 = 5.3e-4 per context

and it attributes the enhancement to starting the C1/C2 iteration from
W9-lo-consistent seed pairs (K_seeds=4): the iteration is said to
"preferentially converge to fixed points that also satisfy the lo constraint."

That mechanism is directly testable without a GPU.  For every convergence we
record the residual

    r = (W9_actual - W9_init) mod 2^32

and ask how often (r & 0xFFFF) == 0.  Under the uniform model that is 2^-16.
The paper needs 36.5x that.  We measure it with K_seeds=0 (cold (0,0) start,
no seeding -> expect NO enhancement) and with K_seeds=4 (production
configuration -> the paper's mechanism predicts the enhancement appears).

If K_seeds=4 shows no enhancement, mu = 5.3e-4 is unsupported and the
complexity table needs rederiving.  If it does show ~36.5x, the enhancement is
real and the 100x discrepancy with the observed finds lives in the hi filter
instead, which needs the H100 to settle.

We also histogram the residuals to check uniformity directly.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PUB = Path("/home/administrator/sha/publish/code")
sys.path.insert(0, str(PUB))

from extended_solver import backward_chain, compute_e_from_a, recover_W  # noqa: E402
from sha256_core import H0, K  # noqa: E402

MISS = np.uint32(0xFFFFFFFF)
SIGMA_TABLE = "/nvme0n1-disk/Kamvid/sigma0_u_table.npy"
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
    hb = os.urandom(32)
    ka = dict(backward_chain(hb, 19)[0])
    for r in range(4, 11):
        if r not in ka:
            ka[r] = int.from_bytes(os.urandom(4), "big")
    for r in range(11, 19):
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
    """K_seeds: (a2,a3) pairs whose W9 low 16 bits already match W9_init."""
    if k <= 0:
        return [(np.uint32(0), np.uint32(0))]
    tgt = np.uint32(int(w9_init) & LO)
    out = []
    tries = 0
    while len(out) < k and tries < 8:
        a2 = rng.integers(0, 1 << 32, size=pool, dtype=np.uint64).astype(np.uint32)
        a3 = rng.integers(0, 1 << 32, size=pool, dtype=np.uint64).astype(np.uint32)
        hit = np.nonzero((w9g(C, a2, a3) & np.uint32(LO)) == tgt)[0]
        for i in hit[: k - len(out)]:
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

    conv_n = 0
    lo_n = 0
    hi_n = 0
    full_n = 0
    hist = np.zeros(256, dtype=np.int64)   # histogram of lo-residual >> 8
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

        for sa2, sa3 in seeds:
            a2c = np.full(a0a.size, sa2, dtype=np.uint32)
            a3c = np.full(a0a.size, sa3, dtype=np.uint32)
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
                a2c, a3c = a2n, a3n
                if conv.any():
                    idx = np.nonzero(conv)[0]
                    conv_n += idx.size
                    resid = (w9g(C, a2c[idx], a3c[idx]) - w9_init)
                    lo_res = (resid & np.uint32(LO)).astype(np.int64)
                    hist += np.bincount(lo_res >> 8, minlength=256)
                    lo_hit = lo_res == 0
                    lo_n += int(lo_hit.sum())
                    hi_res = ((resid >> np.uint32(16)) & np.uint32(LO)).astype(np.int64)
                    hi_n += int((hi_res == 0).sum())
                    if lo_hit.any():
                        full_n += int((resid[lo_hit] == 0).sum())
    return conv_n, lo_n, hi_n, full_n, hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=14)
    p.add_argument("--nbatch", type=int, default=40)
    p.add_argument("--batch", type=int, default=1 << 20)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--kseeds", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260729)
    a = p.parse_args()

    print("Loading sigma0 table into RAM (16 GiB)...", flush=True)
    t0 = time.time()
    _G["sig"] = np.load(SIGMA_TABLE).view(np.uint32)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    import multiprocessing as mp
    jobs = [(w, a.nbatch, a.batch, a.iters, a.kseeds, a.seed + 7919 * w)
            for w in range(a.jobs)]
    total_a0 = a.jobs * a.nbatch * a.batch
    print(f"K_seeds={a.kseeds}: {a.jobs} workers x {a.nbatch} batches "
          f"x {a.batch:,} = {total_a0:,} a0 "
          f"({total_a0/2**32:.2f} context-equivalents)", flush=True)

    t0 = time.time()
    with mp.get_context("fork").Pool(a.jobs) as pool:
        res = pool.map(sweep, jobs)
    conv = sum(r[0] for r in res)
    lo = sum(r[1] for r in res)
    hi = sum(r[2] for r in res)
    full = sum(r[3] for r in res)
    hist = sum(r[4] for r in res)
    el = time.time() - t0

    print(f"\nelapsed {el:.0f}s")
    print(f"convergences        = {conv:,}")
    print(f"lo-pass events      = {lo:,}")
    print(f"hi-pass events      = {hi:,}   (upper 16 bits of residual == 0)")
    print(f"full W9 matches     = {full:,}")
    if conv:
        obs = lo / conv
        print(f"\nlo-pass rate per convergence = {obs:.3e}")
        print(f"uniform prediction (2^-16)   = {2**-16:.3e}")
        print(f"MEASURED LO ENHANCEMENT      = {obs/2**-16:.2f}x")
        print(f"paper claims                 = 36.5x")
        hobs = hi / conv
        print(f"\nhi-pass rate per convergence = {hobs:.3e}")
        print(f"MEASURED HI ENHANCEMENT      = {hobs/2**-16:.2f}x")
        print(f"paper ASSUMES hi is uniform  = 1.00x")
        print(f"\nimplied mu per context = "
              f"{obs*hobs*conv/(total_a0/2**32):.3e}  (paper: 5.3e-4)")
        if hobs > 0:
            print(f"implied expected contexts = "
                  f"{1/(obs*hobs*conv/(total_a0/2**32)):.1f}  "
                  f"(paper: 1875, observed finds: 3/20/34)")
        # chi2 of the lo-residual histogram
        exp = hist.sum() / 256
        chi2 = float(((hist - exp) ** 2 / exp).sum()) if exp > 0 else float("nan")
        print(f"\nlo-residual uniformity: chi2 = {chi2:.1f} (255 dof, "
              f"expect ~255 if uniform)")
        print(f"  bin0 count = {hist[0]:,}   mean bin = {exp:,.1f}")
    print(f"\nconvergences per context-equivalent = "
          f"{conv/(total_a0/2**32):,.0f}   (paper: 62,745)")
    if conv:
        print(f"lo-pass per context-equivalent      = "
              f"{lo/(total_a0/2**32):,.1f}   (paper: ~35)")


if __name__ == "__main__":
    main()
