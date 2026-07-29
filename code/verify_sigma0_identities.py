#!/usr/bin/env python3
"""Verify the sigma0 linear identities stated in the R=20 discussion.

Self-contained: no tables, no GPU, no imports beyond the standard library
(numpy only for the exhaustive pass, which is optional).

Claims checked
--------------
1. L = sigma0 XOR I has rank 31 over GF(2).
2. ker(L) = {0, 0x27f42515}   (the sigma0 fixed points).
3. There is a UNIQUE nonzero left-null mask lambda = 0xa8a42fe4, i.e.
       lambda . (sigma0(W) XOR W) = 0   for every W          [Proposition]
4. Modular subtraction destroys it:
       Pr_W[ lambda . (sigma0(W) - W) = 1 ] = 0.500017190
   a bias of 1.72e-05 = 2^-15.8, against an exact 0 for the XOR form.

Usage
-----
    python3 verify_sigma0_identities.py            # algebra + 2^24 sample
    python3 verify_sigma0_identities.py --full     # exhaustive over all 2^32
"""

from __future__ import annotations

import argparse

M = 0xFFFFFFFF


def sigma0(x: int) -> int:
    x &= M
    return (((x >> 7) | (x << 25)) ^ ((x >> 18) | (x << 14)) ^ (x >> 3)) & M


def parity(x: int) -> int:
    return bin(x).count("1") & 1


def gf2_analysis():
    """Rank, kernel and left-null space of L = sigma0 XOR I."""
    cols = [sigma0(1 << i) ^ (1 << i) for i in range(32)]

    # right kernel, by elimination while tracking the combination used
    pivots, kernel = {}, []
    for i in range(32):
        v, c = cols[i], 1 << i
        for b in sorted(pivots, reverse=True):
            if (v >> b) & 1:
                v ^= pivots[b][0]
                c ^= pivots[b][1]
        if v == 0:
            kernel.append(c)
        else:
            pivots[v.bit_length() - 1] = (v, c)
    rank = len(pivots)

    # left-null masks: same elimination on the transpose
    tcols = []
    for j in range(32):
        col = 0
        for i in range(32):
            if (cols[i] >> j) & 1:
                col |= 1 << i
        tcols.append(col)
    tp, left = {}, []
    for j in range(32):
        v, c = tcols[j], 1 << j
        for b in sorted(tp, reverse=True):
            if (v >> b) & 1:
                v ^= tp[b][0]
                c ^= tp[b][1]
        if v == 0:
            left.append(c)
        else:
            tp[v.bit_length() - 1] = (v, c)
    return rank, kernel, left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="exhaustive pass over all 2^32 W (needs numpy, ~4 min)")
    args = ap.parse_args()

    rank, kernel, left = gf2_analysis()
    print(f"[1] rank(sigma0 XOR I) = {rank}"
          f"   {'OK' if rank == 31 else 'MISMATCH (expected 31)'}")

    kern = sorted(k for k in kernel)
    fixed_ok = all(sigma0(k) == k for k in kern)
    print(f"[2] kernel basis = {[hex(k) for k in kern]}"
          f"   all are sigma0 fixed points: {'OK' if fixed_ok else 'NO'}")
    print(f"    -> ker(L) = {{0x00000000, {hex(kern[0]) if kern else '-'}}}")

    print(f"[3] left-null masks = {[hex(l) for l in left]}"
          f"   {'OK (unique)' if len(left) == 1 else 'unexpected count'}")
    if not left:
        return
    lam = left[0]
    print(f"    lambda = {lam:#010x}"
          f"   {'matches the paper' if lam == 0xA8A42FE4 else 'DIFFERS from paper'}")

    # sampled check of the identity, pure python
    bad = 0
    for w in range(0, 1 << 24, 997):
        if parity((sigma0(w) ^ w) & lam):
            bad += 1
    print(f"    sampled check of lambda.(sigma0(W) XOR W) = 0: "
          f"{bad} violations   {'OK' if bad == 0 else 'FAILS'}")

    if not args.full:
        print("\n(run with --full for the exhaustive 2^32 pass and the "
              "modular-subtraction bias)")
        return

    import numpy as np

    def v_sigma0(x):
        return (((x >> np.uint32(7)) | (x << np.uint32(25)))
                ^ ((x >> np.uint32(18)) | (x << np.uint32(14)))
                ^ (x >> np.uint32(3)))

    def v_parity(x):
        for k in (16, 8, 4, 2, 1):
            x = x ^ (x >> np.uint32(k))
        return (x & np.uint32(1)).astype(np.uint8)

    lam_u = np.uint32(lam)
    step = 1 << 26
    viol = 0
    ones = 0
    for base in range(0, 1 << 32, step):
        w = np.arange(step, dtype=np.uint32) + np.uint32(base)
        s = v_sigma0(w)
        viol += int(v_parity((s ^ w) & lam_u).sum())
        ones += int(v_parity((s - w) & lam_u).sum())
    n = 1 << 32
    print(f"\n[3-full] exhaustive violations of the XOR identity: {viol}"
          f"   {'PROVEN BY EXHAUSTION' if viol == 0 else 'FALSE'}")
    p = ones / n
    print(f"[4] Pr[lambda.(sigma0(W) - W) = 1] = {p:.9f}")
    print(f"    bias = {abs(p-0.5):.3e} = 2^{-__import__('math').log2(1/abs(p-0.5)):.1f}"
          f"   -> modular subtraction destroys the identity")


if __name__ == "__main__":
    main()
