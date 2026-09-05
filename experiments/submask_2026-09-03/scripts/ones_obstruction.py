#!/usr/bin/env python3
"""Is the all-ones digest structurally harder at R=20, or are the pods unlucky?

The two pods have accumulated ~1.95 expected preimages on the all-ones digest
with no hit, while random digests produced one hit in 1.20 expected.  Neither is
alarming alone; together they raise the question of whether the fourth
constraint can reach zero AT ALL for this particular target in this family.

A uniformity test on the low 4-12 bits (already done, 55k candidates) cannot see
an obstruction that lives deeper.  The decisive test is the SCALING LADDER on
this specific target: count candidates whose fourth-constraint residual has its
low k bits zero, for k up to 26, and check the counts halve at each step.  If
they track 2^-k out to 24-26 bits, no obstruction exists and the pods are simply
unlucky.  If they die at some depth, the target is obstructed and the run should
be stopped.

Runs the all-ones target and, as a matched control, random targets in the same
process.  Multi-core.
"""
import multiprocessing as mp
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, U32, MISS, ZERO, K, IV, S0, S1, s0, s1, Ch, Maj,
                            T2, u32, digest, recover_W, backward_chain, make_context)

TABLE = "/nvme0n1-disk/Kamvid/sigma0_u_table.npy"
R = 20
KMAX = 26
ONES = bytes([0xFF] * 32)


def worker(args):
    wid, seed, n_a0, use_ones = args
    tbl = np.load(TABLE, mmap_mode="r").view(np.uint32)
    rng = np.random.default_rng(seed)
    if use_ones:
        h = ONES
    else:
        msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
        pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
        h = digest([struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)], R)
    ctx = make_context(rng, R)
    ab, eb = backward_chain(h, R)
    a = dict(ab); a.update(ctx); a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(8, R):
        e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M
    a4, a5, a6, a7, a8, a9, a10 = (a[i] for i in range(4, 11))
    am1, am2, am3, am4 = a[-1], a[-2], a[-3], a[-4]
    em1, em2, em3, em4 = e[-1], e[-2], e[-3], e[-4]
    e8, e9, e10 = e[8], e[9], e[10]
    T1_7 = (a7 - T2(a6, a5, a4)) & M
    c6 = (a6 - S0(a5)) & M
    W9base = ((a9 - T2(a8, a7, a6)) - K[9]) & M
    W10base = ((a10 - T2(a9, a8, a7)) - S1(e9) - K[10]) & M
    W11base = ((a[11] - T2(a10, a9, a8)) - S1(e10) - K[11]) & M
    Wr = {r: recover_W(a, e, r) for r in range(12, R)}
    K0p = (Wr[16] - s1(Wr[14])) & M; K1p = (Wr[17] - s1(Wr[15])) & M
    K2p = (Wr[18] - s1(Wr[16])) & M
    K3p = (Wr[19] - s1(Wr[17]) - Wr[12]) & M
    T2iv = T2(am1, am2, am3)
    C0c = (-T2iv - em4 - S1(em1) - Ch(em1, em2, em3) - K[0]) & M
    Ce0 = (am4 - T2iv) & M
    W9hat = (W9base - (a5 - S0(a4) - Maj(a4, 0, 0)) - S1(e8)
             - Ch(e8, T1_7, (c6 - Maj(a5, a4, 0)) & M)) & M
    D = ((a6 - S0(a5) - Maj(a5, a4, 0)) + Ch(e9, e8, T1_7)) & M
    KC0 = (K0p - W9hat) & M
    KC1 = (K1p - W10base + D) & M
    KC2 = (K2p - W11base + T1_7 + Ch(e10, e9, e8)) & M

    counts = np.zeros(KMAX + 1, dtype=np.int64)
    n_sub = 0
    B = 1 << 21
    done = 0
    while done < n_a0:
        b = min(B, n_a0 - done); done += b
        A0 = rng.integers(0, 1 << 32, size=b, dtype=np.uint64).astype(U32)
        E0 = (A0 + u32(Ce0)) & MISS
        W0 = (A0 + u32(C0c)) & MISS
        G = (-(S0(A0) + Maj(A0, u32(am1), u32(am2))) - u32(em3) - S1(E0)
             - Ch(E0, u32(em1), u32(em2)) - u32(K[1])) & MISS
        W1 = np.asarray(tbl[(u32(KC0) - W0 - G) & MISS]); keep = W1 != MISS
        A0, E0, G, W1 = A0[keep], E0[keep], G[keep], W1[keep]
        A1 = (W1 - G) & MISS
        E1 = (u32(am3) + A1 - (S0(A0) + Maj(A0, u32(am1), u32(am2)))) & MISS
        F12 = (-(S0(A1) + Maj(A1, A0, u32(am1))) - u32(em2) - S1(E1)
               - Ch(E1, E0, u32(em1)) - u32(K[2])) & MISS
        W2 = np.asarray(tbl[(u32(KC1) - W1 - F12) & MISS]); keep = W2 != MISS
        A0, A1, E0, E1, F12, W2 = (x[keep] for x in (A0, A1, E0, E1, F12, W2))
        A2 = (W2 - F12) & MISS
        e2 = (u32(am2) + A2 - (S0(A1) + Maj(A1, A0, u32(am1)))) & MISS
        F23 = (-(S0(A2) + Maj(A2, A1, A0)) - u32(em1) - S1(e2)
               - Ch(e2, E1, E0) - u32(K[3])) & MISS
        W3 = np.asarray(tbl[(u32(KC2) - W2 - F23) & MISS]); keep = W3 != MISS
        A0, A1, A2, E0, E1, e2, F23, W3 = (x[keep] for x in
                                           (A0, A1, A2, E0, E1, e2, F23, W3))
        A3 = (W3 - F23) & MISS
        sub = np.nonzero(Maj(u32(a4), A3, A2) == A3)[0]
        n_sub += sub.size
        if sub.size == 0:
            continue
        A0s, A1s, A2s, A3s = A0[sub], A1[sub], A2[sub], A3[sub]
        E0s, E1s, e2s, W3s = E0[sub], E1[sub], e2[sub], W3[sub]
        e3 = (u32(am1) + A3s - (S0(A2s) + Maj(A2s, A1s, A0s))) & MISS
        W4 = (u32(a4) - (S0(A3s) + Maj(A3s, A2s, A1s)) - E0s - S1(e3)
              - Ch(e3, e2s, E1s) - u32(K[4])) & MISS
        c3 = (s0(W4) + W3s - u32(K3p)) & MISS
        for k in range(KMAX + 1):
            counts[k] += int(((c3 & U32((1 << k) - 1)) == ZERO).sum())
    return dict(a0=done, sub=n_sub, counts=counts, ones=use_ones)


def report(label, sub, counts):
    print(f"\n{label}: {sub:,} candidates")
    print(f"  {'k':>3} {'observed':>10} {'predicted':>11} {'ratio':>7} {'z':>7}")
    for k in range(0, KMAX + 1, 2):
        pred = sub / (1 << k)
        if pred < 0.05 and counts[k] == 0:
            continue
        z = (counts[k] - pred) / np.sqrt(pred) if pred > 0 else 0.0
        print(f"  {k:>3} {counts[k]:>10,} {pred:>11.2f} {counts[k]/pred:>7.2f} {z:>+7.2f}")
    # conditional halving, the independent statistic
    ok = [k for k in range(16, KMAX) if counts[k] >= 8]
    if ok:
        n = sum(counts[k] for k in ok); m = sum(counts[k + 1] for k in ok)
        z = (m - n / 2) / np.sqrt(n / 4)
        print(f"  conditional halving over steps {ok[0]}..{ok[-1]+1}: "
              f"{m} of {n} (expected {n/2:.1f}), z = {z:+.2f}")


def main():
    ncore = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    per = int(sys.argv[2]) if len(sys.argv) > 2 else (1 << 31)
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    tot = {True: dict(a0=0, sub=0, c=np.zeros(KMAX + 1, np.int64)),
           False: dict(a0=0, sub=0, c=np.zeros(KMAX + 1, np.int64))}
    t0 = time.time()
    half = ncore // 2
    for rnd in range(rounds):
        jobs = [(w, 700000 + rnd * 997 + w, per, w < half) for w in range(ncore)]
        with mp.Pool(ncore) as p:
            for r in p.imap_unordered(worker, jobs):
                d = tot[r['ones']]
                d['a0'] += r['a0']; d['sub'] += r['sub']; d['c'] += r['counts']
        print(f"\n=== round {rnd+1}/{rounds}, {time.time()-t0:.0f}s")
        report("ALL-ONES target", tot[True]['sub'], tot[True]['c'])
        report("random targets (matched control)", tot[False]['sub'], tot[False]['c'])
        sys.stdout.flush()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
