#!/usr/bin/env python3
"""Numeric confirmation of the a11 trace at R=20.

For random 20-round digests and random submask-family contexts, flip bits of
a11 and record whether KC0, KC1, KC2, the collapse condition (v) and kappa3
change, and by how many bits (edge weights, mean Hamming weight per single-bit
flip).  Then decompose the change in KC0 into its terms, repeat under every
realizable extra saturation, and finally run the real pipeline (with the
sigma0 table) on the same a0 sweep for two values of a11 and intersect the
candidate sets.
"""
import struct, sys, time
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, U32, MISS, ZERO, K, IV, S0, S1, s0, s1, Ch, Maj, T2, u32,
                            digest, recover_W, backward_chain, make_context, load_table)

R = 20
TABLE = "/nvme0n1-disk/Kamvid/sigma0_u_table.npy"


def rand_digest(rng):
    msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
    pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
    return digest([struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)], R)


def constants(h, ctx):
    """Exactly the per-context constants of attack_context(), plus the pieces."""
    ab, eb = backward_chain(h, R)
    a = dict(ab); a.update(ctx); a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
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
    return dict(KC0=KC0, KC1=KC1, KC2=KC2, collapse_v=a4, kappa3=K3p,
                W=Wr, e=e, a=a, W9hat=W9hat, D=D, T1_7=T1_7, W11base=W11base)


def hw(x): return bin(x & M).count("1")


def perturb(rng, make_ctx, label, n_ctx=40, bits=32):
    keys = ('KC0', 'KC1', 'KC2', 'collapse_v', 'kappa3')
    changed = {k: 0 for k in keys}
    hwsum = {k: 0.0 for k in keys}
    n = 0
    ok_family = 0
    for _ in range(n_ctx):
        h = rand_digest(rng)
        ctx = make_ctx(rng, h)
        c0 = constants(h, ctx)
        e = c0['e']; a = c0['a']
        ok_family += int(e[8] == M and e[9] == M and a[4] == a[5])
        for b in range(bits):
            c2 = dict(ctx); c2[11] = ctx[11] ^ (1 << b)
            c1 = constants(h, c2)
            n += 1
            for k in keys:
                d = c0[k] ^ c1[k]
                changed[k] += int(d != 0)
                hwsum[k] += hw(d)
    print(f"\n[{label}]  {n_ctx} contexts x {bits} single-bit flips of a11 = {n} flips; "
          f"family conditions held in {ok_family}/{n_ctx}")
    print(f"   {'constant':12s} {'changed':>9s} {'mean HW of change':>18s}")
    for k in keys:
        print(f"   {k:12s} {changed[k]:>5d}/{n:<4d} {hwsum[k]/n:>14.2f}")
    return changed


def decompose_KC0(rng, n_ctx=40):
    """Split dKC0 into the a11-carrying terms of W16 and s1(W14); show that
    freezing S1(e15) artificially still leaves KC0 moving (S0(a11) in e12)."""
    print("\n[decomposition of the a11-dependence of KC0]")
    print("   KC0 = W16 - s1(W14) - W9hat;  W16 = A16 - T2(A15,A14,A13) - e12 - S1(e15) - Ch(e15,e14,e13) - K16")
    cnt = dict(S1e15=0, e12=0, Ch15=0, s1W14=0, W9hat=0, KC0_frozen_S1e15=0, KC0_frozen_S1e15_e12=0)
    n = 0
    for _ in range(n_ctx):
        h = rand_digest(rng); ctx = make_context(rng, R)
        c0 = constants(h, ctx)
        for b in range(32):
            c2 = dict(ctx); c2[11] = ctx[11] ^ (1 << b)
            c1 = constants(h, c2)
            n += 1
            e0, e1 = c0['e'], c1['e']
            t0 = dict(S1e15=S1(e0[15]), e12=e0[12], Ch15=Ch(e0[15], e0[14], e0[13]),
                      s1W14=s1(c0['W'][14]), W9hat=c0['W9hat'])
            t1 = dict(S1e15=S1(e1[15]), e12=e1[12], Ch15=Ch(e1[15], e1[14], e1[13]),
                      s1W14=s1(c1['W'][14]), W9hat=c1['W9hat'])
            for k in t0:
                cnt[k] += int(t0[k] != t1[k])
            # hypothetical: put S1(e15) back to its unperturbed value
            kc0_f = (c1['KC0'] + S1(e1[15]) - S1(e0[15])) & M
            cnt['KC0_frozen_S1e15'] += int(kc0_f != c0['KC0'])
            kc0_ff = (kc0_f + e1[12] - e0[12]) & M
            cnt['KC0_frozen_S1e15_e12'] += int(kc0_ff != c0['KC0'])
    for k, v in cnt.items():
        print(f"   {k:24s} moved in {v}/{n} flips")


# ---- realizable extra saturations, numerically ----
def ctx_a10_eq_A12(rng, h):
    ab, _ = backward_chain(h, R); c = make_context(rng, R); c[10] = ab[12]; return c

def ctx_a10_eq_a9(rng, h):
    c = make_context(rng, R); c[10] = c[9]; return c

def ctx_e10_ones(rng, h):
    c = make_context(rng, R); c[10] = (M - c[6] + T2(c[9], c[8], c[7])) & M; return c

def ctx_e14_ones(rng, h):
    ab, _ = backward_chain(h, R); c = make_context(rng, R)
    c[10] = (M - ab[14] + T2(ab[13], ab[12], c[11])) & M; return c

def ctx_e11_ones(rng, h, tries=64, iters=200):
    """e11 = a7 + a11 - T2(a10,a9,a8) = -1 via a7, with a8, a9 the family words (which
    depend on a7): fixed point on a7.  Random restarts; returns None on failure."""
    for _ in range(tries):
        c = make_context(rng, R)
        v = c[4]
        a7 = c[7]
        for _i in range(iters):
            a8 = (M - v + S0(a7) + Maj(a7, c[6], v)) & M
            a9 = (M - v + S0(a8) + Maj(a8, a7, c[6])) & M
            a7n = (M - c[11] + T2(c[10], a9, a8)) & M
            if a7n == a7:
                c[7], c[8], c[9] = a7, a8, a9
                e11 = (a7 + c[11] - T2(c[10], a9, a8)) & M
                e8 = (v + a8 - T2(a7, c[6], v)) & M
                e9 = (v + a9 - T2(a8, a7, c[6])) & M
                assert e11 == M and e8 == M and e9 == M
                return c
            a7 = a7n
    return None


def candidate_sets(tbl, h, ctx, n_a0, seed):
    """Run the real triangular solve on a fixed a0 stream; return the set of
    (a0,a1,a2,a3) candidates and the subset passing the collapse."""
    c = constants(h, ctx)
    a, e = c['a'], c['e']
    am1, am2, am3, am4 = a[-1], a[-2], a[-3], a[-4]
    em1, em2, em3, em4 = e[-1], e[-2], e[-3], e[-4]
    T2iv = T2(am1, am2, am3)
    C0c = (-T2iv - em4 - S1(em1) - Ch(em1, em2, em3) - K[0]) & M
    Ce0 = (am4 - T2iv) & M
    KC0, KC1, KC2 = c['KC0'], c['KC1'], c['KC2']
    rng = np.random.default_rng(seed)
    A0 = rng.integers(0, 1 << 32, size=n_a0, dtype=np.uint64).astype(U32)
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
    A0, A1, A2, W3, F23 = (x[keep] for x in (A0, A1, A2, W3, F23))
    A3 = (W3 - F23) & MISS
    cand = set(zip(A0.tolist(), A1.tolist(), A2.tolist(), A3.tolist()))
    sub = Maj(u32(a[4]), A3, A2) == A3
    sub_set = set(zip(A0[sub].tolist(), A1[sub].tolist(), A2[sub].tolist(), A3[sub].tolist()))
    return cand, sub_set, dict(KC0=KC0, KC1=KC1, KC2=KC2, kappa3=c['kappa3'])


if __name__ == "__main__":
    rng = np.random.default_rng(20260905)
    t0 = time.time()
    perturb(rng, lambda r, h: make_context(r, R), "submask family, base (v,a6,a7,a10 random)")
    decompose_KC0(rng)
    perturb(rng, ctx_a10_eq_A12, "family + a10 = A12 (kills Maj(A12,a11,a10) in e13)")
    perturb(rng, ctx_a10_eq_a9, "family + a10 = a9 (kills Maj(a11,a10,a9) in e12, W12)")
    perturb(rng, ctx_e10_ones, "family + e10 = -1 via a10 (kills Ch(e10,.,.) in W11, Ch(e11,e10,e9) data)")
    perturb(rng, ctx_e14_ones, "family + e14 = -1 via a10 (a10 now carries a11 through Maj(A13,A12,a11))")
    # e11 = -1 via a7 needs a fixed point; report how often one is found
    found = 0; tried = 0
    def ctx_e11(r, h):
        global found, tried
        while True:
            tried += 1
            c = ctx_e11_ones(r, h)
            if c is not None:
                found += 1
                return c
    perturb(rng, ctx_e11, "family + e11 = -1 via a7 (fixed point on a7; a7 now carries a11)", n_ctx=20)
    print(f"   (fixed point on a7 found in {found}/{tried} attempts)")

    # kappa3 as a function of a11 alone: is it additive / a table form?
    print("\n[shape of kappa3(a11) at fixed everything else]")
    h = rand_digest(rng); ctx = make_context(rng, R)
    base = constants(h, ctx)['kappa3']
    add_ok = 0; n = 0; hws = 0
    for _ in range(256):
        c2 = dict(ctx); c2[11] = int(rng.integers(0, 1 << 32, dtype=np.uint64))
        k = constants(h, c2)['kappa3']
        d = (k - base) & M
        add_ok += int(d == ((c2[11] - ctx[11]) & M) or d == ((ctx[11] - c2[11]) & M))
        n += 1
    for b in range(32):
        c2 = dict(ctx); c2[11] = ctx[11] ^ (1 << b)
        hws += hw(constants(h, c2)['kappa3'] ^ base)
    print(f"   kappa3(a11) - kappa3(a11_0) == +/-(a11 - a11_0) in {add_ok}/{n} random a11; "
          f"mean HW of dkappa3 per single-bit flip = {hws/32:.2f}  (additive would be ~1.5, full avalanche ~16)")

    # end-to-end: same a0 stream, two a11 values, real table
    print("\n[end-to-end candidate sets with the real table]")
    tbl = load_table(TABLE)
    n_a0 = 1 << 21
    tot_c = tot_int = tot_s = tot_sint = 0
    for trial in range(4):
        h = rand_digest(rng); ctx = make_context(rng, R)
        ctx2 = dict(ctx); ctx2[11] = ctx[11] ^ (1 << int(rng.integers(0, 32)))
        cA, sA, kA = candidate_sets(tbl, h, ctx, n_a0, seed=1000 + trial)
        cB, sB, kB = candidate_sets(tbl, h, ctx2, n_a0, seed=1000 + trial)
        print(f"   trial {trial}: a11 flip bit -> KC0 {'moved' if kA['KC0']!=kB['KC0'] else 'same'}, "
              f"KC1 {'moved' if kA['KC1']!=kB['KC1'] else 'same'}, KC2 {'moved' if kA['KC2']!=kB['KC2'] else 'same'}, "
              f"kappa3 {'moved' if kA['kappa3']!=kB['kappa3'] else 'same'}; "
              f"candidates {len(cA)} vs {len(cB)}, common {len(cA & cB)}; "
              f"collapse-passing {len(sA)} vs {len(sB)}, common {len(sA & sB)}")
        tot_c += len(cA); tot_int += len(cA & cB); tot_s += len(sA); tot_sint += len(sA & sB)
    print(f"   total: {tot_int}/{tot_c} candidates shared between the two a11 values "
          f"(expected for independent sets: ~{tot_c * tot_c / 4 / (2**32 * 0.25):.1f} by a0-collision "
          f"with identical (a1,a2,a3) essentially 0); {tot_sint}/{tot_s} collapse-passing shared")
    print(f"\ndone in {time.time()-t0:.0f}s")
