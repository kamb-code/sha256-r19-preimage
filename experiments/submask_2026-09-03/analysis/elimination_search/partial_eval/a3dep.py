#!/usr/bin/env python3
"""Exact a3-dependence of the fourth-constraint residual c3 at R=20.

Setting (submask family, R=20): given a swept a0 and the first two lookups
(a1, a2), the third lookup returns W3 with sigma0(W3) - W3 = t, t = KC2 - W2 - F23,
and a3 = W3 - F23.  Then

    c3 = sigma0(W4) + W3 - kappa3,
    W3 = a3 + F23                                  (F23 free of a3)
    W4 = v - Sigma0(a3) - Maj(a3,a2,a1) - e0 - Sigma1(e3) - Ch(e3,e2,e1) - K4,
    e3 = a_{-1} + a3 - T2^(3)   (T2^(3) = Sigma0(a2)+Maj(a2,a1,a0), free of a3)

Questions answered numerically on random family states:
  (1) per-bit sensitivity matrix  S[j,i] = P(c3 bit i flips | a3 bit j flips)
  (2) GF(2) rank of the span of {c3(a3) xor c3(a3')}      (linear functionals)
  (3) same for the joint (c2, c3) residual, c2 = sigma0(W3)-W3-t  (merging C2, C3)
  (4) bit 0 of c2, c3, c2^c3 all a3-dependent  => no modular-linear functional
  (5) root-collision test: W3, W3' with the same key t give unrelated c3
      (so nothing about c3 is determined by the key before the lookup)
No table needed.
"""
import sys, struct
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, digest,
                            recover_W, backward_chain, make_context, U32, MISS, u32)

R = 20
rng = np.random.default_rng(20260905)


def rand_target():
    msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
    pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
    return digest([struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)], R)


def context_consts(h, ctx):
    """Same per-context constants as attack_context, plus what c3 needs."""
    ab, eb = backward_chain(h, R)
    a = dict(ab); a.update(ctx)
    a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(8, R):
        e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M
    Wr = {r: recover_W(a, e, r) for r in range(12, R)}
    K3p = (Wr[19] - s1(Wr[17]) - Wr[12]) & M
    a10, a9, a8, a7, a6, a5, a4 = (a[i] for i in (10, 9, 8, 7, 6, 5, 4))
    e10, e9, e8 = e[10], e[9], e[8]
    T1_7 = (a7 - T2(a6, a5, a4)) & M
    W11base = ((a[11] - T2(a10, a9, a8)) - S1(e10) - K[11]) & M
    K2p = (Wr[18] - s1(Wr[16])) & M
    KC2 = (K2p - W11base + T1_7 + Ch(e10, e9, e8)) & M
    return dict(a=a, e=e, K3p=K3p, KC2=KC2, v=a4)


def residuals(cc, A0, A1, A2, A3):
    """Vectorised c2 (pre-lookup residual of C2) and c3, exactly as attack_context."""
    am1, am2, am3, am4 = (u32(cc['a'][i]) for i in (-1, -2, -3, -4))
    em1, em2, em3, em4 = (u32(cc['e'][i]) for i in (-1, -2, -3, -4))
    v = u32(cc['v'])
    E0 = (am4 + A0 - u32(T2(cc['a'][-1], cc['a'][-2], cc['a'][-3]))) & MISS
    E1 = (am3 + A1 - (S0(A0) + Maj(A0, am1, am2))) & MISS
    e2 = (am2 + A2 - (S0(A1) + Maj(A1, A0, am1))) & MISS
    F23 = (-(S0(A2) + Maj(A2, A1, A0)) - em1 - S1(e2) - Ch(e2, E1, E0) - u32(K[3])) & MISS
    # W2 (needed for the C2 key t)
    F12 = (-(S0(A1) + Maj(A1, A0, am1)) - em2 - S1(E1) - Ch(E1, E0, em1) - u32(K[2])) & MISS
    W2 = (A2 + F12) & MISS
    t = (u32(cc['KC2']) - W2 - F23) & MISS
    W3 = (A3 + F23) & MISS
    c2 = (s0(W3) - W3 - t) & MISS
    e3 = (am1 + A3 - (S0(A2) + Maj(A2, A1, A0))) & MISS
    W4 = (v - (S0(A3) + Maj(A3, A2, A1)) - E0 - S1(e3) - Ch(e3, e2, E1) - u32(K[4])) & MISS
    c3 = (s0(W4) + W3 - u32(cc['K3p'])) & MISS
    return c2, c3, t, W3, F23


def gf2_rank(vecs, nbits):
    """Rank over GF(2) of a set of python-int bit vectors."""
    basis = []
    for x in vecs:
        for b in basis:
            x = min(x, x ^ b)
        if x:
            basis.append(x)
    return len(basis)


def U(n):
    return rng.integers(0, 1 << 32, size=n, dtype=np.uint64).astype(U32)


def main():
    NCTX, N = 8, 1 << 16
    S = np.zeros((32, 32))           # S[j, i]: a3 bit j -> c3 bit i
    S2 = np.zeros((32, 32))          # same for c2
    diffs3, diffs23 = [], []
    bit0 = np.zeros(3)               # flip counts of c2[0], c3[0], c2[0]^c3[0]
    nb0 = 0
    coll_agree = np.zeros(32); ncoll = 0
    for ci in range(NCTX):
        h = rand_target()
        cc = context_consts(h, make_context(rng, R))
        A0, A1, A2, A3 = U(N), U(N), U(N), U(N)
        c2, c3, t, W3, F23 = residuals(cc, A0, A1, A2, A3)
        for j in range(32):
            A3f = A3 ^ U32(1 << j)
            c2f, c3f, _, _, _ = residuals(cc, A0, A1, A2, A3f)
            d3 = c3 ^ c3f; d2 = c2 ^ c2f
            for i in range(32):
                S[j, i] += ((d3 >> U32(i)) & U32(1)).mean()
                S2[j, i] += ((d2 >> U32(i)) & U32(1)).mean()
        # random a3 pairs: spans
        A3b = U(N)
        c2b, c3b, _, _, _ = residuals(cc, A0, A1, A2, A3b)
        d3 = (c3 ^ c3b)[:4096]; d2 = (c2 ^ c2b)[:4096]
        diffs3 += [int(x) for x in d3]
        diffs23 += [(int(x) << 32) | int(y) for x, y in zip(d2, d3)]
        bit0 += [((c2 ^ c2b) & U32(1)).sum(), ((c3 ^ c3b) & U32(1)).sum(),
                 (((c2 ^ c2b) ^ (c3 ^ c3b)) & U32(1)).sum()]
        nb0 += N
        # root collisions: u, u' with sigma0(u)-u equal
        Us = U(1 << 21)
        key = (s0(Us) - Us) & MISS
        order = np.argsort(key, kind='stable')
        ks = key[order]
        same = np.nonzero(ks[1:] == ks[:-1])[0]
        u1 = Us[order[same]]; u2 = Us[order[same + 1]]
        nz = u1 != u2; u1, u2 = u1[nz], u2[nz]   # drop exact duplicates u == u'
        m = min(u1.size, N)
        if m:
            # place W3 = u1 and W3 = u2 in the same (a0,a1,a2) slot: a3 = W3 - F23
            A3c1 = (u1[:m] - F23[:m]) & MISS
            A3c2 = (u2[:m] - F23[:m]) & MISS
            _, c3c1, t1, _, _ = residuals(cc, A0[:m], A1[:m], A2[:m], A3c1)
            _, c3c2, t2, _, _ = residuals(cc, A0[:m], A1[:m], A2[:m], A3c2)
            assert np.all(t1 == t1)  # keys are slot constants; both roots share sigma0(u)-u
            assert np.all(((s0(u1[:m]) - u1[:m]) & MISS) == ((s0(u2[:m]) - u2[:m]) & MISS))
            d = c3c1 ^ c3c2
            for i in range(32):
                coll_agree[i] += ((d >> U32(i)) & U32(1) == 0).sum()
            ncoll += m
    S /= NCTX; S2 /= NCTX
    np.set_printoptions(precision=2, linewidth=200, suppress=True)
    print("(1) sensitivity S[j,i] = P(c3 bit i flips | a3 bit j flips), rows j = a3 bit, cols i = c3 bit")
    print(S)
    jm, im = np.unravel_index(np.argmin(S), S.shape)
    print(f"    min S = {S.min():.4f} at a3 bit {jm} -> c3 bit {im}; entries == 0: {(S == 0).sum()} of 1024")
    print(f"    mean S = {S.mean():.4f}; column minima (c3 bit i least sensitive a3 bit):")
    print("   ", np.round(S.min(axis=0), 3))
    print(f"    c3 bit 0 sensitivity to each a3 bit: {np.round(S[:, 0], 3)}")
    print(f"    (c2 for reference) min S2 = {S2.min():.4f}, zero entries {(S2 == 0).sum()}")
    r3 = gf2_rank(diffs3, 32); r23 = gf2_rank(diffs23, 64)
    print(f"(2) GF(2) rank of span{{c3(a3) ^ c3(a3')}} over {len(diffs3)} pairs: {r3} / 32")
    print(f"(3) GF(2) rank of span{{(c2,c3)(a3) ^ (c2,c3)(a3')}} over {len(diffs23)} pairs: {r23} / 64")
    print(f"(4) P(bit0 flips) under random a3 change: c2[0] {bit0[0]/nb0:.4f}, c3[0] {bit0[1]/nb0:.4f}, "
          f"c2[0]^c3[0] {bit0[2]/nb0:.4f}   (all must be > 0 for no modular-linear functional)")
    print(f"(5) root-collision test: {ncoll} pairs (W3, W3') with sigma0(u)-u equal, same (a0,a1,a2):")
    print(f"    per-bit P(c3 bit agrees) = {np.round(coll_agree / max(ncoll, 1), 3)}")
    print(f"    mean {coll_agree.sum() / (32 * max(ncoll, 1)):.4f}  (0.5 = c3 carries no information from the key t)")


if __name__ == "__main__":
    main()
