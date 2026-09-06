#!/usr/bin/env python3
"""Planted 20-round preimages in family contexts, and what the sweep does at the
planted a0.  Uses the single-root sigma0 table (memory-mapped).

For each plant: family context (v,a6,a7,a10,a11 random, a8/a9 derived), random
a0..a3, W0..W11 recovered from the state, W12..W15 random, digest = 20-round
compression.  The plant is then a genuine solution of {C0,C1,C2,C3} for
(context, digest) -- one of the ~1 per context the counting predicts.

Arm A: a3 random            -> eps = Maj(v,a3,a2) - a3 is nonzero w.p. 1-(3/4)^32.
Arm B: a3 forced into the submask set (a3_i = a2_i or v_i) -> eps = 0.

At the planted a0 we run exactly the attack's three lookups and compare the
returned triple with the plant, and check the true C0 with the true W9.
"""
import struct, sys
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, recover_W, backward_chain,
                            forward, digest, make_context, u32, U32, MISS)

R = 20
TABLE = "/nvme0n1-disk/Kamvid/sigma0_u_table.npy"
tbl = np.load(TABLE, mmap_mode="r").view(np.uint32)
rng = np.random.default_rng(777)
rnd = lambda: int(rng.integers(0, 1 << 32, dtype=np.uint64))


def plant(force_submask):
    ctx = make_context(rng, R)
    v = ctx[4]
    a = dict(ctx); a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
    a[0], a[1], a[2] = rnd(), rnd(), rnd()
    if force_submask:
        m = rnd(); a[3] = (a[2] & m) | (v & ~m & M)
    else:
        a[3] = rnd()
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(0, 12):
        e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M
    W = [recover_W(a, e, r) for r in range(12)] + [rnd() for _ in range(4)]
    h = digest(W, R)
    af, ef, _ = forward(W, R)
    assert all(af[r] == a[r] for r in range(12)), "plant inconsistent"
    return ctx, {k: a[k] for k in range(4)}, W, h


def lookups_at(h, ctx, a0):
    """The attack's constants and three lookups at one a0 (mirrors attack_context)."""
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
    K0p = (Wr[16] - s1(Wr[14])) & M; K1p = (Wr[17] - s1(Wr[15])) & M
    K2p = (Wr[18] - s1(Wr[16])) & M; K3p = (Wr[19] - s1(Wr[17]) - Wr[12]) & M
    T2iv = T2(am1, am2, am3)
    C0c = (-T2iv - em4 - S1(em1) - Ch(em1, em2, em3) - K[0]) & M
    Ce0 = (am4 - T2iv) & M
    W9hat = (W9base - (a5 - S0(a4) - Maj(a4, 0, 0)) - S1(e8)
             - Ch(e8, T1_7, (c6 - Maj(a5, a4, 0)) & M)) & M
    D = ((a6 - S0(a5) - Maj(a5, a4, 0)) + Ch(e9, e8, T1_7)) & M
    KC0 = (K0p - W9hat) & M; KC1 = (K1p - W10base + D) & M
    KC2 = (K2p - W11base + T1_7 + Ch(e10, e9, e8)) & M
    E0 = (a0 + Ce0) & M; W0 = (a0 + C0c) & M
    G = (-(S0(a0) + Maj(a0, am1, am2)) - em3 - S1(E0) - Ch(E0, em1, em2) - K[1]) & M
    key0 = (KC0 - W0 - G) & M
    W1 = int(tbl[key0])
    if W1 == M: return None, (key0,)
    A1 = (W1 - G) & M
    E1 = (am3 + A1 - (S0(a0) + Maj(a0, am1, am2))) & M
    F12 = (-(S0(A1) + Maj(A1, a0, am1)) - em2 - S1(E1) - Ch(E1, E0, em1) - K[2]) & M
    key1 = (KC1 - W1 - F12) & M
    W2 = int(tbl[key1])
    if W2 == M: return None, (key0, key1)
    A2 = (W2 - F12) & M
    e2 = (am2 + A2 - (S0(A1) + Maj(A1, a0, am1))) & M
    F23 = (-(S0(A2) + Maj(A2, A1, a0)) - em1 - S1(e2) - Ch(e2, E1, E0) - K[3]) & M
    key2 = (KC2 - W2 - F23) & M
    W3 = int(tbl[key2])
    if W3 == M: return None, (key0, key1, key2)
    A3 = (W3 - F23) & M
    return (a0, A1, A2, A3), (key0, key1, key2)


def true_keys(ctx, unk, h):
    """The sigma0(u)-u keys that the PLANT itself satisfies (so we can tell a table
    miss on the plant's root from a genuinely different solution)."""
    a = dict(ctx); a.update(unk); a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
    ab, eb = backward_chain(h, R); a.update(ab)
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(R):
        e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M
    W = {r: recover_W(a, e, r) for r in range(R)}
    return [(s0(W[1 + t]) - W[1 + t]) & M for t in range(3)], [W[1], W[2], W[3]]


for arm, force in (("A: random a3 (eps generic)", False), ("B: a3 in the submask set (eps = 0)", True)):
    n = 400 if force else 2000
    st = dict(eps0=0, exact=0, other_triple=0, miss=0, plant_root_stored=0, c0true_of_other=0)
    for _ in range(n):
        ctx, unk, W, h = plant(force)
        v = ctx[4]
        eps = (Maj(v, unk[3], unk[2]) - unk[3]) & M
        if eps == 0: st['eps0'] += 1
        trip, keys = lookups_at(h, ctx, unk[0])
        tk, tw = true_keys(ctx, unk, h)
        # is the plant's own root stored for every key?  (single-root table keeps the largest root)
        stored = all(int(tbl[k]) == w for k, w in zip(tk, tw))
        if stored: st['plant_root_stored'] += 1
        if trip is None:
            st['miss'] += 1
        elif tuple(trip) == (unk[0], unk[1], unk[2], unk[3]):
            st['exact'] += 1
        else:
            st['other_triple'] += 1
            # the other triple: does it satisfy the true C0 (it never can unless its own eps = 0 and c3 ...)
            a = dict(ctx); a.update({0: trip[0], 1: trip[1], 2: trip[2], 3: trip[3]})
            ab, eb = backward_chain(h, R); a.update(ab)
            a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
            e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
            for r in range(R):
                e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M
            Wm = {r: recover_W(a, e, r) for r in range(R)}
            c0 = (s0(Wm[1]) + Wm[0] + Wm[9] + s1(Wm[14]) - Wm[16]) & M
            if c0 == 0: st['c0true_of_other'] += 1
    print(f"arm {arm}: {n} plants (each a genuine 20-round preimage of its digest in its context)")
    print(f"   eps == 0                      : {st['eps0']}   (predicted {n*0.75**32:.3f} in arm A, all in arm B)")
    print(f"   plant's roots stored (c^3)    : {st['plant_root_stored']}   (predicted {n*0.633673**3:.1f})")
    print(f"   sweep at the planted a0 returns the plant : {st['exact']}")
    print(f"   returns a different triple    : {st['other_triple']}   (of which satisfy true C0: {st['c0true_of_other']})")
    print(f"   table miss                    : {st['miss']}")
