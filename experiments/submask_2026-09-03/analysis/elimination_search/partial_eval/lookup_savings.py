#!/usr/bin/env python3
"""Lookup accounting and the collapse-condition pre-filter, on the real pipeline (R=20).

Per swept a0 the pipeline does: 1 C0 lookup, c C1 lookups (c = 0.6337 image
fraction), c^2 C2 lookups.  The collapsed condition Maj(v,a3,a2)==a3, i.e.
(a2^a3)&(a3^v)==0, forces a3[i] = a2[i] wherever a2[i] == v[i].  Before the C2
lookup we know a2 and v, hence the forced positions.  This script measures:

  * lookups per swept a0 by stage,
  * h = popcount(a2 ^ v)  (free bits of a3) and L = trailing forced bits,
  * the ONLY sound pre-lookup C2 test: if L >= 19 the low L bits of
    W3 = a3 + F23 are known, so (sigma0(W3) - W3) mod 2^(L-18) is known and
    must equal t mod 2^(L-18); otherwise the lookup cannot return a
    collapse-surviving root and can be skipped.
  * soundness: the pre-test never rejects a candidate that passes the collapse.
  * an optional exhaustive alternative for small h: enumerate the 2^h admissible
    a3 and test C2 directly instead of the lookup (captures all roots).
"""
import sys, struct, time
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, digest,
                            recover_W, backward_chain, make_context, U32, MISS, ZERO, u32,
                            load_table)

R = 20


def rand_target(rng):
    msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
    pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
    return digest([struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)], R)


def tz32(x):
    """trailing zeros of uint32 array (32 for x == 0)."""
    x = x.astype(np.uint64)
    low = x & (~x + np.uint64(1)) & np.uint64(0xFFFFFFFF)
    out = np.full(x.shape, 32, dtype=np.int64)
    nz = low != 0
    out[nz] = np.log2(low[nz].astype(np.float64)).astype(np.int64)
    return out


def sweep(tbl, h, ctx, n_a0, rng, st):
    ab, eb = backward_chain(h, R)
    a = dict(ab); a.update(ctx)
    a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
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
    v = u32(a4)

    B = 1 << 20
    done = 0
    while done < n_a0:
        b = min(B, n_a0 - done); done += b
        st['a0'] += b; st['lk0'] += b
        A0 = rng.integers(0, 1 << 32, size=b, dtype=np.uint64).astype(U32)
        E0 = (A0 + u32(Ce0)) & MISS
        W0 = (A0 + u32(C0c)) & MISS
        G = (-(S0(A0) + Maj(A0, u32(am1), u32(am2))) - u32(em3) - S1(E0)
             - Ch(E0, u32(em1), u32(em2)) - u32(K[1])) & MISS
        W1 = np.asarray(tbl[(u32(KC0) - W0 - G) & MISS])
        keep = W1 != MISS
        A0, E0, W0, G, W1 = (x[keep] for x in (A0, E0, W0, G, W1))
        st['lk1'] += A0.size
        A1 = (W1 - G) & MISS
        E1 = (u32(am3) + A1 - (S0(A0) + Maj(A0, u32(am1), u32(am2)))) & MISS
        F12 = (-(S0(A1) + Maj(A1, A0, u32(am1))) - u32(em2) - S1(E1)
               - Ch(E1, E0, u32(em1)) - u32(K[2])) & MISS
        W2 = np.asarray(tbl[(u32(KC1) - W1 - F12) & MISS])
        keep = W2 != MISS
        A0, A1, E0, E1, W1, F12, W2 = (x[keep] for x in (A0, A1, E0, E1, W1, F12, W2))
        A2 = (W2 - F12) & MISS
        e2 = (u32(am2) + A2 - (S0(A1) + Maj(A1, A0, u32(am1)))) & MISS
        F23 = (-(S0(A2) + Maj(A2, A1, A0)) - u32(em1) - S1(e2)
               - Ch(e2, E1, E0) - u32(K[3])) & MISS
        t = (u32(KC2) - W2 - F23) & MISS
        n2 = A0.size
        st['lk2'] += n2
        # ---- what is known BEFORE the C2 lookup: a2, v -> forced positions of a3
        m = A2 ^ v                                   # free mask; forced = ~m
        x = m.astype(np.uint64)
        x = x - ((x >> np.uint64(1)) & np.uint64(0x55555555))
        x = (x & np.uint64(0x33333333)) + ((x >> np.uint64(2)) & np.uint64(0x33333333))
        x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F)
        hfree = ((x * np.uint64(0x01010101)) >> np.uint64(24)) & np.uint64(0xFF)
        st['hist_h'] += np.bincount(hfree.astype(np.int64), minlength=33)[:33]
        L = tz32(m)                                  # trailing forced bits
        for l in np.bincount(L, minlength=33)[:33].nonzero()[0]:
            st['hist_L'][l] += int((L == l).sum())
        # exact pre-test: k = L - 18 low bits of sigma0(W3)-W3 are determined by
        # W3 mod 2^L = (a2 + F23) mod 2^L
        pre_ok = np.ones(n2, dtype=bool)
        idx = np.nonzero(L >= 19)[0]
        if idx.size:
            W3lo = (A2[idx] + F23[idx]) & MISS       # bits < L correct
            kbits = (L[idx] - 18).astype(np.uint32)
            mask = ((np.uint64(1) << kbits.astype(np.uint64)) - np.uint64(1)).astype(U32)
            pred = (s0(W3lo) - W3lo) & mask
            pre_ok[idx] = pred == (t[idx] & mask)
            st['pre_eligible'] += idx.size
            st['pre_skipped'] += int((~pre_ok[idx]).sum())
        # ---- the real lookup, for everyone (to verify soundness)
        W3 = np.asarray(tbl[t])
        keep = W3 != MISS
        A3 = (W3 - F23) & MISS
        st['sol'] += int(keep.sum())
        hit = keep & (Maj(v, A3, A2) == A3)
        st['eps0'] += int(hit.sum())
        # soundness: no collapse survivor was pre-rejected
        st['unsound'] += int((hit & ~pre_ok).sum())
        # C3 on survivors
        j = np.nonzero(hit)[0]
        if j.size:
            e3 = (u32(am1) + A3[j] - (S0(A2[j]) + Maj(A2[j], A1[j], A0[j]))) & MISS
            W4 = (v - (S0(A3[j]) + Maj(A3[j], A2[j], A1[j])) - E0[j]
                  - S1(e3) - Ch(e3, e2[j], E1[j]) - u32(K[4])) & MISS
            c3 = (s0(W4) + W3[j] - u32(K3p)) & MISS
            st['c3'] += int((c3 == ZERO).sum())
            # partial-evaluation check: c3 recomputed from forced bits only (a3 := a2
            # on forced positions, 0 on free) agrees with the true c3 in how many bits?
            A3guess = A2[j] & ~m[j]
            e3g = (u32(am1) + A3guess - (S0(A2[j]) + Maj(A2[j], A1[j], A0[j]))) & MISS
            W4g = (v - (S0(A3guess) + Maj(A3guess, A2[j], A1[j])) - E0[j]
                   - S1(e3g) - Ch(e3g, e2[j], E1[j]) - u32(K[4])) & MISS
            c3g = (s0(W4g) + ((A3guess + F23[j]) & MISS) - u32(K3p)) & MISS
            d = c3 ^ c3g
            for i in range(32):
                st['guess_agree'][i] += int(((d >> U32(i)) & U32(1) == 0).sum())
            st['guess_n'] += j.size


def main():
    tbl = load_table("/nvme0n1-disk/Kamvid/sigma0_u_table.npy")
    rng = np.random.default_rng(20260905)
    n_ctx, n_a0 = int(sys.argv[1]) if len(sys.argv) > 1 else 4, 1 << 22
    st = dict(a0=0, lk0=0, lk1=0, lk2=0, sol=0, eps0=0, c3=0, unsound=0,
              pre_eligible=0, pre_skipped=0, hist_h=np.zeros(33, dtype=np.int64),
              hist_L=np.zeros(33, dtype=np.int64), guess_agree=np.zeros(32), guess_n=0)
    t0 = time.time()
    for ci in range(n_ctx):
        h = rand_target(rng)
        sweep(tbl, h, make_context(rng, R), n_a0, rng, st)
        print(f"ctx {ci}: a0 {st['a0']:,} lookups C0 {st['lk0']:,} C1 {st['lk1']:,} C2 {st['lk2']:,} "
              f"sol {st['sol']:,} eps0 {st['eps0']:,} c3 {st['c3']} [{time.time()-t0:.0f}s]", flush=True)
    A = st['a0']
    print(f"\nlookups per swept a0: C0 {st['lk0']/A:.4f}  C1 {st['lk1']/A:.4f}  C2 {st['lk2']/A:.4f}  "
          f"total {(st['lk0']+st['lk1']+st['lk2'])/A:.4f}   (model 1 + c + c^2 = {1+0.633673+0.633673**2:.4f})")
    print(f"triangular solutions per a0 {st['sol']/A:.4f} (model c^3 = {0.633673**3:.4f}); "
          f"collapse survivors {st['eps0']} = {st['eps0']/max(st['sol'],1):.3e} per solution ((3/4)^32 = {0.75**32:.3e})")
    print(f"C3 == 0: {st['c3']}")
    print(f"pre-lookup C2 test: eligible (L>=19) {st['pre_eligible']} of {st['lk2']:,} C2 lookups "
          f"= {st['pre_eligible']/max(st['lk2'],1):.3e} (model 2^-19 = {2**-19:.3e}); "
          f"skipped {st['pre_skipped']} = {st['pre_skipped']/max(st['lk2'],1):.3e} (model 2^-19*2/3 = {2**-19*2/3:.3e})")
    print(f"soundness: collapse survivors rejected by the pre-test = {st['unsound']} (must be 0)")
    hl = st['hist_L']; tot = hl.sum()
    print("trailing forced bits L (P model 2^-(L+1)):", {int(l): f"{hl[l]/tot:.2e}" for l in range(33) if hl[l]})
    hh = st['hist_h']; th = hh.sum()
    if th:
        print(f"free-bit count h = popcount(a2^v) over {th} sampled C2 lookups: mean {np.dot(np.arange(33), hh)/th:.2f}, "
              f"P(h<=5) = {hh[:6].sum()/th:.2e} (model {sum(__import__('math').comb(32,k) for k in range(6))/2**32:.2e})")
    if st['guess_n']:
        g = st['guess_agree'] / st['guess_n']
        print(f"c3 from forced bits only (free bits := 0), per-bit agreement with true c3 over {st['guess_n']} survivors: "
              f"mean {g.mean():.3f}, min {g.min():.3f}, max {g.max():.3f}")


if __name__ == "__main__":
    main()
