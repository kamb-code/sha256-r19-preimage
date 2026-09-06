#!/usr/bin/env python3
"""Tie a4 = phi(a0) at full width, real table.

Idea: the barrier is a4 -> C0 (heavy).  A dependency on the SWEPT word is
harmless, so let the family word v be a function of a0: v = phi(a0).  Then
per swept a0 the whole context (a5=v, a8(v), a9(v)) and the constants
KC0, KC1, KC2, kappa3 are recomputed (arithmetic only, no lookups), the three
lookups proceed as usual, and c3 is tested.  Dimension count says this is a
reparameterisation (a 32-bit tie replaces the 32-bit choice of v) and cannot
change P(c3 == 0 | candidate) from 2^-32.  This measures it: the low-k-bit
ladder of c3 on candidates, for several phi, against uniform.
"""
import multiprocessing as mp, struct, sys, time
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, U32, MISS, ZERO, K, IV, S0, S1, s0, s1, Ch, Maj,
                            T2, u32, digest, recover_W, backward_chain)

TABLE = "/nvme0n1-disk/Kamvid/sigma0_u_table.npy"
R = 20
KMAX = 24


def phi(A0, kind, c):
    if kind == 'a0+c':
        return (A0 + u32(c)) & MISS
    if kind == 'S0(a0)+c':
        return (S0(A0) + u32(c)) & MISS
    if kind == '~a0':
        return ~A0
    if kind == 'a0^c':
        return A0 ^ u32(c)
    raise ValueError(kind)


def worker(args):
    wid, seed, n_a0, kind = args
    tbl = np.load(TABLE, mmap_mode="r").view(np.uint32)
    rng = np.random.default_rng(seed)
    msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
    pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
    h = digest([struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)], R)
    ab, _ = backward_chain(h, R)
    A = {r: u32(ab[r]) for r in range(12, 20)}
    a6c, a7c, a10c, a11c, c = (int(rng.integers(0, 1 << 32, dtype=np.uint64)) for _ in range(5))
    am1, am2, am3, am4 = (u32(x) for x in IV[:4])
    em1, em2, em3, em4 = (u32(x) for x in IV[4:])
    T2iv = T2(IV[0], IV[1], IV[2])
    C0c = u32(-T2iv - IV[7] - S1(IV[4]) - Ch(IV[4], IV[5], IV[6]) - K[0])
    Ce0 = u32(IV[3] - T2iv)
    Kc = [u32(k) for k in K]
    counts = np.zeros(KMAX + 1, dtype=np.int64)
    n_sub = 0; done = 0; B = 1 << 20
    while done < n_a0:
        b = min(B, n_a0 - done); done += b
        A0 = rng.integers(0, 1 << 32, size=b, dtype=np.uint64).astype(U32)
        V = phi(A0, kind, c)
        a6 = np.full(b, a6c, U32); a7 = np.full(b, a7c, U32)
        a10 = np.full(b, a10c, U32); a11 = np.full(b, a11c, U32)
        a4 = V; a5 = V
        a8 = (MISS - V + S0(a7) + Maj(a7, a6, V)) & MISS
        a9 = (MISS - V + S0(a8) + Maj(a8, a7, a6)) & MISS
        a = {4: a4, 5: a5, 6: a6, 7: a7, 8: a8, 9: a9, 10: a10, 11: a11}
        a.update({r: np.full(b, A[r], U32) for r in range(12, 20)})
        e = {}
        for r in range(8, 20):
            e[r] = (a[r - 4] + a[r] - (S0(a[r - 1]) + Maj(a[r - 1], a[r - 2], a[r - 3]))) & MISS
        assert bool((e[8] == MISS).all()) and bool((e[9] == MISS).all())
        W = {}
        for r in range(12, 20):
            W[r] = (a[r] - (S0(a[r - 1]) + Maj(a[r - 1], a[r - 2], a[r - 3])) - e[r - 4]
                    - S1(e[r - 1]) - Ch(e[r - 1], e[r - 2], e[r - 3]) - Kc[r]) & MISS
        T1_7 = (a7 - (S0(a6) + Maj(a6, a5, a4))) & MISS
        c6 = (a6 - S0(a5)) & MISS
        W9base = (a9 - (S0(a8) + Maj(a8, a7, a6)) - Kc[9]) & MISS
        W10base = (a10 - (S0(a9) + Maj(a9, a8, a7)) - S1(e[9]) - Kc[10]) & MISS
        W11base = (a11 - (S0(a10) + Maj(a10, a9, a8)) - S1(e[10]) - Kc[11]) & MISS
        K0p = (W[16] - s1(W[14])) & MISS
        K1p = (W[17] - s1(W[15])) & MISS
        K2p = (W[18] - s1(W[16])) & MISS
        K3p = (W[19] - s1(W[17]) - W[12]) & MISS
        z = np.zeros(b, U32)
        W9hat = (W9base - (a5 - S0(a4) - Maj(a4, z, z)) - S1(e[8])
                 - Ch(e[8], T1_7, (c6 - Maj(a5, a4, z)) & MISS)) & MISS
        D = ((a6 - S0(a5) - Maj(a5, a4, z)) + Ch(e[9], e[8], T1_7)) & MISS
        KC0 = (K0p - W9hat) & MISS
        KC1 = (K1p - W10base + D) & MISS
        KC2 = (K2p - W11base + T1_7 + Ch(e[10], e[9], e[8])) & MISS

        E0 = (A0 + Ce0) & MISS
        W0 = (A0 + C0c) & MISS
        G = (-(S0(A0) + Maj(A0, am1, am2)) - em3 - S1(E0) - Ch(E0, em1, em2) - Kc[1]) & MISS
        W1 = np.asarray(tbl[(KC0 - W0 - G) & MISS]); keep = W1 != MISS
        A0, E0, G, W1, KC1, KC2, K3p, a4 = (x[keep] for x in (A0, E0, G, W1, KC1, KC2, K3p, a4))
        A1 = (W1 - G) & MISS
        E1 = (am3 + A1 - (S0(A0) + Maj(A0, am1, am2))) & MISS
        F12 = (-(S0(A1) + Maj(A1, A0, am1)) - em2 - S1(E1) - Ch(E1, E0, em1) - Kc[2]) & MISS
        W2 = np.asarray(tbl[(KC1 - W1 - F12) & MISS]); keep = W2 != MISS
        A0, A1, E0, E1, F12, W2, KC2, K3p, a4 = (x[keep] for x in (A0, A1, E0, E1, F12, W2, KC2, K3p, a4))
        A2 = (W2 - F12) & MISS
        e2 = (am2 + A2 - (S0(A1) + Maj(A1, A0, am1))) & MISS
        F23 = (-(S0(A2) + Maj(A2, A1, A0)) - em1 - S1(e2) - Ch(e2, E1, E0) - Kc[3]) & MISS
        W3 = np.asarray(tbl[(KC2 - W2 - F23) & MISS]); keep = W3 != MISS
        A0, A1, A2, E0, E1, e2, F23, W3, K3p, a4 = (x[keep] for x in
                                                    (A0, A1, A2, E0, E1, e2, F23, W3, K3p, a4))
        A3 = (W3 - F23) & MISS
        sub = np.nonzero(Maj(a4, A3, A2) == A3)[0]
        n_sub += sub.size
        if sub.size == 0:
            continue
        A0s, A1s, A2s, A3s = A0[sub], A1[sub], A2[sub], A3[sub]
        E0s, E1s, e2s, W3s, K3s, a4s = E0[sub], E1[sub], e2[sub], W3[sub], K3p[sub], a4[sub]
        e3 = (am1 + A3s - (S0(A2s) + Maj(A2s, A1s, A0s))) & MISS
        W4 = (a4s - (S0(A3s) + Maj(A3s, A2s, A1s)) - E0s - S1(e3) - Ch(e3, e2s, E1s) - Kc[4]) & MISS
        c3 = (s0(W4) + W3s - K3s) & MISS
        for k in range(KMAX + 1):
            counts[k] += int(((c3 & U32((1 << k) - 1)) == ZERO).sum())
    return dict(a0=done, sub=n_sub, counts=counts, kind=kind)


def report(label, a0, sub, counts):
    print(f"\n{label}: {a0:,} swept a0, {sub:,} candidates ({sub/a0:.3e} per a0; "
          f"family prediction 0.254*(3/4)^32 = {0.634**3*0.75**32:.3e})")
    print(f"  {'k':>3} {'observed':>10} {'predicted':>11} {'ratio':>7} {'z':>7}")
    for k in range(0, KMAX + 1, 2):
        pred = sub / (1 << k)
        if pred < 0.05 and counts[k] == 0:
            continue
        z = (counts[k] - pred) / np.sqrt(pred) if pred > 0 else 0.0
        print(f"  {k:>3} {counts[k]:>10,} {pred:>11.2f} {counts[k]/pred:>7.2f} {z:>+7.2f}")
    ok = [k for k in range(8, KMAX) if counts[k] >= 8]
    if ok:
        n = sum(counts[k] for k in ok); m = sum(counts[k + 1] for k in ok)
        z = (m - n / 2) / np.sqrt(n / 4)
        print(f"  conditional halving over steps {ok[0]}..{ok[-1]+1}: {m} of {n} "
              f"(expected {n/2:.1f}), z = {z:+.2f}")


def main():
    ncore = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    per = int(sys.argv[2]) if len(sys.argv) > 2 else (1 << 27)
    kinds = ['a0+c', 'S0(a0)+c', '~a0', 'a0^c']
    t0 = time.time()
    tot = {k: dict(a0=0, sub=0, c=np.zeros(KMAX + 1, np.int64)) for k in kinds}
    jobs = [(w, 910000 + w, per, kinds[w % len(kinds)]) for w in range(ncore)]
    with mp.Pool(ncore) as p:
        for r in p.imap_unordered(worker, jobs):
            d = tot[r['kind']]
            d['a0'] += r['a0']; d['sub'] += r['sub']; d['c'] += r['counts']
    print(f"elapsed {time.time()-t0:.0f}s")
    for k in kinds:
        report(f"tie a4 = {k}", tot[k]['a0'], tot[k]['sub'], tot[k]['c'])
    allc = sum(tot[k]['c'] for k in kinds); alls = sum(tot[k]['sub'] for k in kinds)
    report("ALL ties pooled", sum(tot[k]['a0'] for k in kinds), alls, allc)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
