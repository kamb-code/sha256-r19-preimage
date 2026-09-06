#!/usr/bin/env python3
"""Full-width numeric checks for the 'unexamined assumptions' at R=20.

(1) Exact closed form of the C0 lookup constant's v-dependence in the family:
        W9hat = Sigma0(v) - v + Sigma0(a6) - a7 - 1 - K9 - Sigma1(-1)   (mod 2^32)
    so C0's target carries Sigma0(v) - v (the 'one heavy edge') PLUS whatever
    W16 - sigma1(W14) picks up through a8(v), a9(v).
(2) Avalanche: Hamming weight of the change in KC0, KC1, KC2, kappa3 per single
    bit flip of each free context word (v, a6, a7, a10, a11), a8/a9 recomputed
    to keep the family.  A word that moved kappa3 but not KC0..KC2 would be a
    'neutral word' allowing one candidate to be tested against many kappa3.
(3) v with a8, a9 FROZEN (family broken): the bare Sigma0(v)-v + Sigma1(e8) edge.
(4) Per-candidate v on planted 20-round states in the family: flip one bit of
    v, keep a0, redo the three lookups; does the same (a1,a2,a3) come back?
"""
import struct, sys
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, forward, digest,
                            recover_W, backward_chain, make_context)

R = 20
TABLE = "/nvme0n1-disk/Kamvid/sigma0_u_table.npy"


def hw(x):
    return bin(x & M).count("1")


def family_ctx(v, a6, a7, a10, a11):
    c = {4: v, 5: v, 6: a6, 7: a7}
    c[8] = (M - v + S0(a7) + Maj(a7, a6, v)) & M
    c[9] = (M - v + S0(c[8]) + Maj(c[8], a7, a6)) & M
    c[10], c[11] = a10, a11
    return c


def constants(h, ctx):
    """KC0, KC1, KC2, kappa3 exactly as attack_context computes them."""
    ab, eb = backward_chain(h, R)
    a = dict(ab); a.update(ctx)
    a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(8, R):
        e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M
    a4, a5, a6, a7, a8, a9, a10 = (a[i] for i in range(4, 11))
    e8, e9, e10 = e[8], e[9], e[10]
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
    W9hat = (W9base - (a5 - S0(a4) - Maj(a4, 0, 0)) - S1(e8)
             - Ch(e8, T1_7, (c6 - Maj(a5, a4, 0)) & M)) & M
    D = ((a6 - S0(a5) - Maj(a5, a4, 0)) + Ch(e9, e8, T1_7)) & M
    KC0 = (K0p - W9hat) & M
    KC1 = (K1p - W10base + D) & M
    KC2 = (K2p - W11base + T1_7 + Ch(e10, e9, e8)) & M
    return dict(KC0=KC0, KC1=KC1, KC2=KC2, k3=K3p, W9hat=W9hat, K0p=K0p, D=D,
                T1_7=T1_7, e=e, a=a)


def rand_target(rng):
    msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
    pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
    return digest([struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)], R)


def main():
    rng = np.random.default_rng(20260905)
    r32 = lambda: int(rng.integers(0, 1 << 32, dtype=np.uint64))

    # ---- (1) closed form of W9hat, and of D and T1_7, in the family
    bad = 0
    for _ in range(2000):
        v, a6, a7, a10, a11 = (r32() for _ in range(5))
        h = rand_target(rng)
        c = constants(h, family_ctx(v, a6, a7, a10, a11))
        pred_W9hat = (S0(v) - v + S0(a6) - a7 - 1 - K[9] - S1(M)) & M
        pred_D = (a6 - S0(v) - v + M) & M                # Ch(e9=-1, e8=-1, .) = e8 = -1
        pred_T17 = (a7 - S0(a6) - v) & M
        if c['W9hat'] != pred_W9hat or c['D'] != pred_D or c['T1_7'] != pred_T17:
            bad += 1
    print(f"(1) closed forms in the family, 2000 random contexts/targets: "
          f"{'ALL EXACT' if bad == 0 else str(bad) + ' MISMATCHES'}")
    print("    W9hat = Sigma0(v) - v + Sigma0(a6) - a7 - 1 - K9 - Sigma1(-1)")
    print("    D     = a6 - Sigma0(v) - v - 1          (C1 constant: Sigma0(v)+v)")
    print("    T1_7  = a7 - Sigma0(a6) - v             (C2 constant: linear in v)")

    # ---- (2) avalanche per flipped bit, family maintained
    names = ['v', 'a6', 'a7', 'a10', 'a11']
    acc = {n: np.zeros(4) for n in names}
    T = 300
    for _ in range(T):
        w = [r32() for _ in range(5)]
        h = rand_target(rng)
        base = constants(h, family_ctx(*w))
        for i, n in enumerate(names):
            w2 = list(w); w2[i] ^= 1 << int(rng.integers(0, 32))
            c2 = constants(h, family_ctx(*w2))
            for j, key in enumerate(('KC0', 'KC1', 'KC2', 'k3')):
                acc[n][j] += hw(base[key] ^ c2[key])
    print(f"\n(2) mean Hamming change per single-bit flip, family maintained (a8,a9 recomputed), {T} trials")
    print(f"    {'word':>5} {'KC0':>7} {'KC1':>7} {'KC2':>7} {'kappa3':>7}")
    for n in names:
        print(f"    {n:>5} " + " ".join(f"{x/T:7.2f}" for x in acc[n]))
    print("    (full avalanche ~16; a neutral word for kappa3 would show ~0 in KC0..KC2 and >0 in kappa3)")

    # ---- (3) v with a8,a9 frozen: the bare edge Sigma0(v)-v (+Sigma1(e8), Ch(e8,..) since e8 != -1 now)
    acc3 = np.zeros(4); accW9 = 0.0
    for _ in range(T):
        w = [r32() for _ in range(5)]
        h = rand_target(rng)
        ctx = family_ctx(*w)
        v2 = w[0] ^ (1 << int(rng.integers(0, 32)))
        ctx2 = dict(ctx); ctx2[4] = v2; ctx2[5] = v2         # a8, a9 kept
        b = constants(h, ctx); c2 = constants(h, ctx2)
        for j, key in enumerate(('KC0', 'KC1', 'KC2', 'k3')):
            acc3[j] += hw(b[key] ^ c2[key])
        accW9 += hw((S0(w[0]) - w[0]) ^ (S0(v2) - v2))
    print(f"\n(3) v flipped with a8, a9 FROZEN (family broken, e8,e9 no longer -1), {T} trials")
    print(f"    KC0 {acc3[0]/T:.2f}  KC1 {acc3[1]/T:.2f}  KC2 {acc3[2]/T:.2f}  kappa3 {acc3[3]/T:.2f}"
          f"   | bare Sigma0(v)-v term alone: {accW9/T:.2f} bits")

    # ---- (4) per-candidate v on planted family states, with the real table
    tbl = np.load(TABLE, mmap_mode="r").view(np.uint32)
    same_a1 = same_all = 0; N = 400
    for _ in range(N):
        v, a6, a7, a10, a11 = (r32() for _ in range(5))
        ctx = family_ctx(v, a6, a7, a10, a11)
        # plant: random a0..a3 satisfying the collapsed condition, random a12..a19 -> digest
        a = {i: r32() for i in range(0, 4)}
        a[3] = (a[2] & ~(a[2] ^ v)) | (a[3] & (a[2] ^ v))      # a3_i = a2_i where a2_i == v_i
        a.update(ctx); a.update({i: r32() for i in range(12, 20)})
        a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
        e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
        for r in range(0, R):
            e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M
        Wm = [recover_W(a, e, r) for r in range(16)]
        # the true W1..W3 must be table roots for the plant to be findable; force by
        # checking; if the stored root differs, skip (single-root table)
        h = digest(Wm, R)
        af, ef, Wf = forward(Wm, R)
        if any(af[r] != a[r] for r in range(12, 20)):
            continue                                        # schedule mismatch: not a preimage, skip
        # the plant is a preimage only if W16..W19 recovered == schedule; a random plant is not.
        # We only need the LOOKUP behaviour, so use the chain constants from this (non-)preimage
        # state directly: KC's computed from its own a12..a19, and test whether the true W1
        # is what the table returns, before and after flipping v.
    # simpler and exact: measure directly whether the C0 lookup INDEX changes when v flips
    # (it does iff KC0 changes; KC0 avalanche measured above), and how many bits of a1.
    idx_same = 0; a1_change = 0.0; n = 0
    for _ in range(N):
        w = [r32() for _ in range(5)]
        h = rand_target(rng)
        b = constants(h, family_ctx(*w))
        w2 = list(w); w2[0] ^= 1 << int(rng.integers(0, 32))
        c2 = constants(h, family_ctx(*w2))
        a0 = r32()
        am1, am2, am3, am4 = IV[0], IV[1], IV[2], IV[3]
        em1, em2, em3, em4 = IV[4], IV[5], IV[6], IV[7]
        T2iv = T2(am1, am2, am3)
        C0c = (-T2iv - em4 - S1(em1) - Ch(em1, em2, em3) - K[0]) & M
        Ce0 = (am4 - T2iv) & M
        E0 = (a0 + Ce0) & M; W0 = (a0 + C0c) & M
        G = (-(S0(a0) + Maj(a0, am1, am2)) - em3 - S1(E0) - Ch(E0, em1, em2) - K[1]) & M
        i1 = (b['KC0'] - W0 - G) & M; i2 = (c2['KC0'] - W0 - G) & M
        W1a = int(tbl[i1]); W1b = int(tbl[i2])
        if i1 == i2:
            idx_same += 1
        if W1a != M and W1b != M:
            a1_change += hw(W1a ^ W1b); n += 1
    print(f"\n(4) per-candidate v: flip one bit of v, same a0, redo the C0 lookup ({N} trials)")
    print(f"    identical lookup index: {idx_same}/{N};  mean Hamming change of a1 when both hit: "
          f"{a1_change/max(n,1):.2f} bits ({n} pairs)")


if __name__ == "__main__":
    main()
