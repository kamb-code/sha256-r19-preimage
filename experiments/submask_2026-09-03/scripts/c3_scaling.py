#!/usr/bin/env python3
"""Does the fourth constraint really cost a full 2^32 at R = 20?

The R=20 estimate of 2^47.3 swept a0 per preimage assumes P(C3 == 0) = 2^-32
for candidates that already satisfy C0, C1, C2 and the collapsed consistency
condition.  A binned chi-squared on the low bits cannot see a spike confined to
the exact 32-bit value -- the same blind spot the main paper flags for its own
W9 filter.  The honest test is the SCALING: count candidates whose C3 residual
has its low k bits zero, for k = 0..26, and check the counts halve with each
extra bit.  If the rate tracks 2^-k out to k = 20 or beyond, extrapolating to
k = 32 is well supported; if it flattens or dies, the estimate is wrong.

Runs the submask family at R = 20 across all cores.  Prints cumulative counts
every few minutes and keeps any deep hit for inspection.
"""
import multiprocessing as mp
import os, struct, sys, time
import numpy as np

sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, U32, MISS, ZERO, K, IV, S0, S1, s0, s1, Ch, Maj,
                            T2, u32, digest, recover_W, backward_chain, make_context)

TABLE = "/nvme0n1-disk/Kamvid/sigma0_u_table.npy"
R = 20
KMAX = 26
BATCH = 1 << 20


def worker(args):
    wid, seed, n_a0 = args
    tbl = np.load(TABLE, mmap_mode="r").view(np.uint32)
    rng = np.random.default_rng(seed)
    counts = np.zeros(KMAX + 1, dtype=np.int64)
    n_sol = 0
    n_sub = 0
    deep = []
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

    done = 0
    while done < n_a0:
        b = min(BATCH, n_a0 - done); done += b
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
        n_sol += A0.size
        if A0.size == 0:
            continue
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
        # trailing-zero depth of the residual, capped at KMAX
        z = np.zeros(c3.size, dtype=np.int64)
        for k in range(1, KMAX + 1):
            z += ((c3 & U32((1 << k) - 1)) == ZERO)
        counts[0] += c3.size
        for k in range(1, KMAX + 1):
            counts[k] += int(((c3 & U32((1 << k) - 1)) == ZERO).sum())
        for j in np.nonzero((c3 & U32((1 << 18) - 1)) == ZERO)[0][:4]:
            aa = dict(a)
            aa.update({0: int(A0s[j]), 1: int(A1s[j]), 2: int(A2s[j]), 3: int(A3s[j])})
            ee = dict(e)
            for rr in range(R):
                ee[rr] = (aa[rr - 4] + aa[rr] - T2(aa[rr - 1], aa[rr - 2],
                                                   aa[rr - 3])) & M
            Wm = [recover_W(aa, ee, rr) for rr in range(16)]
            deep.append((int(c3[j]), h.hex(), [int(w) for w in Wm]))
    return dict(a0=done, sol=n_sol, sub=n_sub, counts=counts, deep=deep, h=h.hex())


def main():
    ncore = int(sys.argv[1]) if len(sys.argv) > 1 else 26
    per = int(sys.argv[2]) if len(sys.argv) > 2 else (1 << 32)
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    tot = dict(a0=0, sol=0, sub=0)
    counts = np.zeros(KMAX + 1, dtype=np.int64)
    deep = []
    t0 = time.time()
    for rnd in range(rounds):
        with mp.Pool(ncore) as p:
            for r in p.imap_unordered(worker,
                                      [(w, 900000 + rnd * 1000 + w, per) for w in range(ncore)]):
                tot['a0'] += r['a0']; tot['sol'] += r['sol']; tot['sub'] += r['sub']
                counts += r['counts']; deep += r['deep']
        el = time.time() - t0
        print(f"\n=== after round {rnd+1}/{rounds}: {tot['a0']:,} swept a0 in {el:.0f}s "
              f"({tot['a0']/el:.2e} a0/s)")
        print(f"    triangular solutions {tot['sol']:,} ({tot['sol']/tot['a0']:.4f}/a0)")
        print(f"    submask (eps==0)     {tot['sub']:,} ({tot['sub']/max(tot['sol'],1):.4e}/solution)")
        print(f"    C3 low-k zero, against the uniform prediction sub/2^k:")
        print(f"      {'k':>3} {'observed':>12} {'predicted':>12} {'ratio':>8}")
        for k in range(0, KMAX + 1, 2):
            pred = tot['sub'] / (1 << k)
            if pred < 0.02 and counts[k] == 0:
                continue
            print(f"      {k:>3} {counts[k]:>12,} {pred:>12.2f} "
                  f"{counts[k]/pred if pred else float('nan'):>8.2f}")
        if deep:
            print(f"    deepest residuals kept: {len(deep)}; "
                  f"min |C3| trailing zeros = "
                  f"{max((bin(c ^ (c-1)).count('1')-1) if c else 32 for c, _, _ in deep)}")
        sys.stdout.flush()
    with open("/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/c3_deep.txt", "w") as f:
        for c3v, hh, Wm in deep:
            f.write(f"{c3v:08x} {hh} {' '.join(f'{w:08x}' for w in Wm)}\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
