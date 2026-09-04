#!/usr/bin/env python3
"""C3 low-bit uniformity at R=20 on the ALL-ONES target (the fleet's production
target), using the CPU numpy path of submask_family (one root, on-disk table).
Mirrors attack_context but keeps the C3 residual of every collapsed-condition
candidate.  Usage: c3_ones_check.py [n_a0_per_ctx] [n_ctx] [hash]"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, U32, MISS, ZERO, K, IV, S0, S1, s0, s1, Ch, Maj, T2,
                            recover_W, backward_chain, make_context, u32, load_table)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 28
NC = int(sys.argv[2]) if len(sys.argv) > 2 else 8
h = bytes.fromhex(sys.argv[3]) if len(sys.argv) > 3 else b"\xff" * 32
R = 20
tbl = load_table("/nvme0n1-disk/Kamvid/sigma0_u_table.npy")
rng = np.random.default_rng(20260904)
c3_all = []
tot = dict(a0=0, sol=0, sub=0)
t0 = time.time()
for ci in range(NC):
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
    assert e8 == M and e9 == M and a4 == a5
    T1_7 = (a7 - T2(a6, a5, a4)) & M
    c6 = (a6 - S0(a5)) & M
    W9base = ((a9 - T2(a8, a7, a6)) - K[9]) & M
    W10base = ((a10 - T2(a9, a8, a7)) - S1(e9) - K[10]) & M
    W11base = ((a[11] - T2(a10, a9, a8)) - S1(e10) - K[11]) & M
    Wr = {r: recover_W(a, e, r) for r in range(12, R)}
    K0p = (Wr[16] - s1(Wr[14])) & M
    K1p = (Wr[17] - s1(Wr[15])) & M
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
    B = 1 << 20
    for s in range(0, N, B):
        A0 = (np.arange(s, s + B, dtype=np.uint64) & M).astype(U32)
        tot['a0'] += B
        E0 = (A0 + u32(Ce0)) & MISS
        W0 = (A0 + u32(C0c)) & MISS
        G = (-(S0(A0) + Maj(A0, u32(am1), u32(am2))) - u32(em3) - S1(E0)
             - Ch(E0, u32(em1), u32(em2)) - u32(K[1])) & MISS
        W1 = np.asarray(tbl[(u32(KC0) - W0 - G) & MISS]); keep = W1 != MISS
        A0, E0, W0, G, W1 = (x[keep] for x in (A0, E0, W0, G, W1))
        A1 = (W1 - G) & MISS
        E1 = (u32(am3) + A1 - (S0(A0) + Maj(A0, u32(am1), u32(am2)))) & MISS
        F12 = (-(S0(A1) + Maj(A1, A0, u32(am1))) - u32(em2) - S1(E1)
               - Ch(E1, E0, u32(em1)) - u32(K[2])) & MISS
        W2 = np.asarray(tbl[(u32(KC1) - W1 - F12) & MISS]); keep = W2 != MISS
        A0, A1, E0, E1, W1, F12, W2 = (x[keep] for x in (A0, A1, E0, E1, W1, F12, W2))
        A2 = (W2 - F12) & MISS
        e2 = (u32(am2) + A2 - (S0(A1) + Maj(A1, A0, u32(am1)))) & MISS
        F23 = (-(S0(A2) + Maj(A2, A1, A0)) - u32(em1) - S1(e2)
               - Ch(e2, E1, E0) - u32(K[3])) & MISS
        W3 = np.asarray(tbl[(u32(KC2) - W2 - F23) & MISS]); keep = W3 != MISS
        A0, A1, A2, E0, E1, e2, F23, W3 = (x[keep] for x in (A0, A1, A2, E0, E1, e2, F23, W3))
        A3 = (W3 - F23) & MISS
        tot['sol'] += A0.size
        idx = np.nonzero(Maj(u32(a4), A3, A2) == A3)[0]
        tot['sub'] += idx.size
        if idx.size:
            e3 = (u32(am1) + A3[idx] - (S0(A2[idx]) + Maj(A2[idx], A1[idx], A0[idx]))) & MISS
            W4 = (u32(a4) - (S0(A3[idx]) + Maj(A3[idx], A2[idx], A1[idx])) - E0[idx]
                  - S1(e3) - Ch(e3, e2[idx], E1[idx]) - u32(K[4])) & MISS
            c3 = (s0(W4) + W3[idx] - u32(K3p)) & MISS
            c3_all.append(c3.copy())
    print(f"ctx {ci}: a0 {tot['a0']:,} sol {tot['sol']:,} sub {tot['sub']:,} "
          f"[{time.time()-t0:.0f}s]", flush=True)

c3 = np.concatenate(c3_all)
n = c3.size
print(f"\ntarget {h.hex()}  R=20  swept a0 {tot['a0']:,}  sub rate {tot['sub']/tot['sol']:.4e} "
      f"(pred {(0.75**32):.4e})  candidates {n:,}")
for bits in (4, 6, 8, 10, 12):
    for side in ("low", "high"):
        v = (c3 & ((1 << bits) - 1)) if side == "low" else (c3 >> (32 - bits))
        cnt = np.bincount(v.astype(np.int64), minlength=1 << bits)
        exp = n / (1 << bits)
        chi2 = float(((cnt - exp) ** 2 / exp).sum()); dof = (1 << bits) - 1
        z = (chi2 - dof) / np.sqrt(2 * dof)
        print(f"  {side:>4} {bits:2d} bits: chi2 {chi2:9.1f} dof {dof:5d} z {z:+.2f}")
print("  trailing-zero ladder (observed / expected n/2^k):")
for k in range(0, 21, 2):
    o = int(((c3 & ((1 << k) - 1)) == 0).sum())
    print(f"    k={k:2d}: {o:9,d} / {n/2**k:11.1f}  ratio {o/(n/2**k):.3f}")
print(f"  exact zeros: {int((c3 == 0).sum())}")
