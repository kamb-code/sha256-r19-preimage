#!/usr/bin/env python3
"""The best ALTERNATIVE assignment found by the tracer, run with the real table:

   context : a1 (random), a5 = a6 = w, e9 = e10 = 0xFFFFFFFF (via a9, a10), a7, a8, a11
   sweep   : a0
   absorb  : C1 -> a2, C2 -> a3, C3 -> a4   (three sigma0(u)-u lookups)
   provisional residual in C1 : Maj(w, a4, a3) - a4  (collapses bitwise, (3/4)^32)
   filter  : C0, an exact 32-bit condition

This is the current attack translated by one index (a1,a2,a3 | v=a4  ->  a2,a3,a4 | w=a5)
with C0 and C3 exchanging the roles of absorber-with-collapse and filter.  It should
tie the published cost exactly: c^3 solutions per swept a0, (3/4)^32 collapse, and a
uniform 32-bit C0 residual.  Also runs a planted-preimage recovery test.
"""
import multiprocessing as mp, struct, sys, time
import numpy as np
sys.path.insert(0, '/home/administrator/sha/publish/code')
from submask_family import (M, U32, MISS, ZERO, K, IV, S0, S1, s0, s1, Ch, Maj, T2, u32,
                            digest, recover_W, backward_chain, forward)
TABLE = '/nvme0n1-disk/Kamvid/sigma0_u_table.npy'
R = 20
KMAX = 24


def make_ctx(rng, w=None):
    ctx = {1: int(rng.integers(0, 1 << 32, dtype=np.uint64))}
    if w is None: w = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    ctx[5] = w; ctx[6] = w
    for i in (7, 8, 11):
        ctx[i] = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    a5, a6, a7, a8 = ctx[5], ctx[6], ctx[7], ctx[8]
    ctx[9] = (M - a5 + S0(a8) + Maj(a8, a7, a6)) & M      # e9 = a5 + a9 - T2(a8,a7,a6) = -1
    ctx[10] = (M - a6 + S0(ctx[9]) + Maj(ctx[9], a8, a7)) & M   # e10 = -1
    return ctx


def constants(h, ctx):
    ab, eb = backward_chain(h, R)
    a = dict(ab); a.update(ctx)
    a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(9, R):          # e9.. need only a5..; e_r for r>=9 are context/chain
        e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M
    assert e[9] == M and e[10] == M and a[5] == a[6]
    w = a[5]; a7, a8, a9, a10, a11 = a[7], a[8], a[9], a[10], a[11]
    c8 = (a8 - S0(a7) - Maj(a7, a6 := w, w)) & M           # e8 = a4 + c8
    Wr = {r: recover_W(a, e, r) for r in range(12, R) if r >= 14}
    # W12, W14 need e8.. : W14 = a14 - T2(a13,a12,a11) - e10 - S1(e13) - Ch(e13,e12,e11)
    # e12 = a8 + a12 - T2(a11,a10,a9): context.  recover_W handles it since e[12] is set.
    W14 = recover_W(a, e, 14); W15 = recover_W(a, e, 15); W16 = recover_W(a, e, 16)
    W17 = recover_W(a, e, 17); W18 = recover_W(a, e, 18); W19 = recover_W(a, e, 19)
    K0p = (W16 - s1(W14)) & M
    K1p = (W17 - s1(W15)) & M
    K2p = (W18 - s1(W16)) & M
    K3p = (W19 - s1(W17)) & M
    W10base = (a10 - T2(a9, a8, a7) - S1(M) - K[10]) & M
    W11base = (a11 - T2(a10, a9, a8) - S1(M) - K[11]) & M
    W12base = (a[12] - T2(a11, a10, a9) - S1(e[11]) - Ch(e[11], M, M) - K[12]) & M
    W9base = (a9 - T2(a8, a7, w) - K[9]) & M
    KC1 = (K1p - W10base + w - S0(w) + c8) & M
    KC2 = (K2p - W11base + a7 - S0(w) - w - 1) & M
    KC3 = (K3p - W12base + c8) & M
    return dict(a=a, e=e, w=w, c8=c8, K0p=K0p, KC1=KC1, KC2=KC2, KC3=KC3,
                W9base=W9base, K3p=K3p)


def sweep(tbl, cst, A0):
    """vectorised chain for an array of a0.  Returns dict of arrays for candidates
    passing the collapsed condition, plus counters."""
    a = cst['a']; w = u32(cst['w']); c8 = u32(cst['c8'])
    am1, am2, am3, am4 = (u32(a[i]) for i in (-1, -2, -3, -4))
    em1, em2, em3, em4 = (u32(IV[4 + i]) for i in range(4))   # e_{-1}..e_{-4}
    a1 = u32(a[1]); a7 = u32(a[7])
    K1, K2, K3, K4 = (u32(K[i]) for i in (1, 2, 3, 4))
    n0 = A0.size
    E0 = (am4 + A0 - (S0(am1) + Maj(am1, am2, am3))) & MISS
    W0 = (A0 - (S0(am1) + Maj(am1, am2, am3)) - em4 - S1(em1) - Ch(em1, em2, em3) - u32(K[0])) & MISS
    E1 = (am3 + a1 - (S0(A0) + Maj(A0, am1, am2))) & MISS
    W1 = (a1 - (S0(A0) + Maj(A0, am1, am2)) - em3 - S1(E0) - Ch(E0, em1, em2) - K1) & MISS
    F2 = (-(S0(a1) + Maj(a1, A0, am1)) - em2 - S1(E1) - Ch(E1, E0, em1) - K2) & MISS   # W2 - a2
    W2 = np.asarray(tbl[(u32(cst['KC1']) - W1 - F2) & MISS]); keep = W2 != MISS
    A0, E0, W0, E1, W1, F2, W2 = (x[keep] for x in (A0, E0, W0, E1, W1, F2, W2))
    n1 = A0.size
    A2 = (W2 - F2) & MISS
    E2 = (am2 + A2 - (S0(a1) + Maj(a1, A0, am1))) & MISS
    F3 = (-(S0(A2) + Maj(A2, a1, A0)) - em1 - S1(E2) - Ch(E2, E1, E0) - K3) & MISS       # W3 - a3
    W3 = np.asarray(tbl[(u32(cst['KC2']) - W2 - F3) & MISS]); keep = W3 != MISS
    A0, E0, W0, E1, W1, A2, E2, F3, W3 = (x[keep] for x in (A0, E0, W0, E1, W1, A2, E2, F3, W3))
    n2 = A0.size
    A3 = (W3 - F3) & MISS
    E3 = (am1 + A3 - (S0(A2) + Maj(A2, a1, A0))) & MISS
    F4 = (-(S0(A3) + Maj(A3, A2, a1)) - E0 - S1(E3) - Ch(E3, E2, E1) - K4) & MISS         # W4 - a4
    W4 = np.asarray(tbl[(u32(cst['KC3']) - W3 - F4) & MISS]); keep = W4 != MISS
    A0, E0, W0, E1, W1, A2, E2, A3, E3, F4, W4, W3 = (x[keep] for x in
        (A0, E0, W0, E1, W1, A2, E2, A3, E3, F4, W4, W3))
    n3 = A0.size
    A4 = (W4 - F4) & MISS
    hit = Maj(w, A4, A3) == A4                       # collapsed condition of THIS frame
    idx = np.nonzero(hit)[0]
    A0, E0, W0, W1, A2, A3, A4 = (x[idx] for x in (A0, E0, W0, W1, A2, A3, A4))
    # the C0 residual, from the true W9
    E5 = (a1 + w - (S0(A4) + Maj(A4, A3, A2))) & MISS
    E6 = (A2 + w - (S0(w) + Maj(w, A4, A3))) & MISS
    E7 = (A3 + a7 - (S0(w) + w)) & MISS
    E8 = (A4 + c8) & MISS
    W9 = (u32(cst['W9base']) - E5 - S1(E8) - Ch(E8, E7, E6)) & MISS
    r0 = (u32(cst['K0p']) - W9 - s0(W1) - W0) & MISS
    return dict(n0=n0, n1=n1, n2=n2, n3=n3, nsub=idx.size, r0=r0, A0=A0, A2=A2, A3=A3, A4=A4)


def worker(args):
    wid, seed, n_a0 = args
    tbl = np.load(TABLE, mmap_mode='r').view(np.uint32)
    rng = np.random.default_rng(seed)
    msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
    pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
    h = digest([struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)], R)
    cst = constants(h, make_ctx(rng))
    counts = np.zeros(KMAX + 1, dtype=np.int64)
    tot = dict(n0=0, n1=0, n2=0, n3=0, nsub=0, ver=0, c0zero=0)
    done = 0; B = 1 << 21
    while done < n_a0:
        b = min(B, n_a0 - done); done += b
        A0 = rng.integers(0, 1 << 32, size=b, dtype=np.uint64).astype(U32)
        out = sweep(tbl, cst, A0)
        for k in ('n0', 'n1', 'n2', 'n3', 'nsub'): tot[k] += out[k]
        r0 = out['r0']
        for k in range(KMAX + 1):
            counts[k] += int(((r0 & U32((1 << k) - 1)) == ZERO).sum())
        # full verification of a sample of candidates: rebuild the message, check
        # C1, C2, C3 hold exactly and C0 residual equals r0
        for j in range(min(4, r0.size)):
            aa = dict(cst['a'])
            aa.update({0: int(out['A0'][j]), 2: int(out['A2'][j]), 3: int(out['A3'][j]), 4: int(out['A4'][j])})
            ee = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
            for rr in range(R):
                ee[rr] = (aa[rr - 4] + aa[rr] - T2(aa[rr - 1], aa[rr - 2], aa[rr - 3])) & M
            Wm = [recover_W(aa, ee, rr) for rr in range(R)]
            res = [(Wm[16 + t] - s1(Wm[14 + t]) - Wm[9 + t] - s0(Wm[1 + t]) - Wm[t]) & M for t in range(4)]
            assert res[1] == 0 and res[2] == 0 and res[3] == 0, res
            assert res[0] == int(r0[j]), (hex(res[0]), hex(int(r0[j])))
            tot['ver'] += 1
            if res[0] == 0:
                tot['c0zero'] += 1
                assert digest(Wm[:16], R) == h
    return tot, counts


def plant_test(n=40, seed=7):
    """Plant full states in this frame, hash them, and check the chain recovers
    (a2,a3,a4) from the true a0 whenever the table stores the planted roots."""
    tbl = np.load(TABLE, mmap_mode='r').view(np.uint32)
    rng = np.random.default_rng(seed)
    rec = stored = 0
    for _ in range(n):
        ctx = make_ctx(rng)
        a = {i: int(rng.integers(0, 1 << 32, dtype=np.uint64)) for i in range(12)}
        a.update(ctx)
        # impose the collapsed condition of this frame on the plant: Maj(w,a4,a3) == a4
        w = ctx[5]; a3 = a[3]
        free = (a3 ^ w) & M                        # a4 free where a3 != w, forced to a3 (= w) elsewhere
        a[4] = ((a3 & ~free) | (int(rng.integers(0, 1 << 32, dtype=np.uint64)) & free)) & M
        assert Maj(w, a[4], a3) == a[4]
        a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
        e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
        for r in range(12):
            e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M
        Wm = [recover_W(a, e, r) for r in range(12)]
        # message words W12..W15 are free: derive them from the schedule-consistent
        # state?  no: a12.. are what the schedule produces.  Build W12..15 random and
        # let the forward pass define a12..a19; then the digest's backward chain
        # returns those, and the schedule constraints C0..C3 hold iff W16..19 from the
        # schedule equal the forward ones -- which they do by construction.
        Wm += [int(rng.integers(0, 1 << 32, dtype=np.uint64)) for _ in range(4)]
        h = digest(Wm, R)
        cst = constants(h, ctx)
        W2, W3, W4 = Wm[2], Wm[3], Wm[4]
        ok = all(int(tbl[(s0(x) - x) & M]) == x for x in (W2, W3, W4))
        stored += ok
        out = sweep(tbl, cst, np.array([a[0]], dtype=U32))
        found = out['nsub'] == 1 and int(out['A4'][0]) == a[4] and int(out['r0'][0]) == 0
        if ok:
            assert found, 'planted preimage with stored roots NOT recovered'
            rec += 1
        elif found:
            rec += 1
    print(f'plant test: {n} plants, {stored} with all three roots stored, {rec} recovered '
          f'({"OK" if rec >= stored else "FAIL"}); expected stored fraction c^3 = 0.254')


if __name__ == '__main__':
    plant_test()
    ncore = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    per = int(sys.argv[2]) if len(sys.argv) > 2 else (1 << 28)
    t0 = time.time()
    tot = dict(n0=0, n1=0, n2=0, n3=0, nsub=0, ver=0, c0zero=0); counts = np.zeros(KMAX + 1, np.int64)
    with mp.Pool(ncore) as p:
        for t, c in p.imap_unordered(worker, [(w, 4242 + w, per) for w in range(ncore)]):
            for k in tot: tot[k] += t[k]
            counts += c
    print(f'\nshifted frame, {ncore} contexts/targets, {tot["n0"]:,} swept a0, {time.time()-t0:.0f}s')
    print(f'  C1 lookup hits {tot["n1"]/tot["n0"]:.4f}  C2 {tot["n2"]/max(tot["n1"],1):.4f}  '
          f'C3 {tot["n3"]/max(tot["n2"],1):.4f}   (c = 0.6337 each)')
    print(f'  triangular solutions per swept a0: {tot["n3"]/tot["n0"]:.4f}  (c^3 = 0.2545)')
    print(f'  collapsed condition Maj(w,a4,a3)==a4: {tot["nsub"]:,} of {tot["n3"]:,} = '
          f'{tot["nsub"]/max(tot["n3"],1):.4e}  ((3/4)^32 = {0.75**32:.4e})')
    print(f'  candidates re-verified from the message (C1,C2,C3 exact, C0 residual matches): {tot["ver"]}')
    print(f'  C0 residual ladder over {tot["nsub"]:,} candidates:')
    print(f'  {"k":>3} {"observed":>10} {"predicted":>11} {"ratio":>7} {"z":>7}')
    for k in range(0, KMAX + 1, 2):
        pred = tot['nsub'] / (1 << k)
        if pred < 0.05 and counts[k] == 0: continue
        z = (counts[k] - pred) / np.sqrt(pred) if pred > 0 else 0
        print(f'  {k:>3} {counts[k]:>10,} {pred:>11.2f} {counts[k]/pred:>7.2f} {z:>+7.2f}')
    print(f'  per-preimage cost implied: 2^{np.log2(tot["n0"]/tot["nsub"]) + 32:.2f} swept a0 '
          f'(published frame: 2^47.3 one-root)')
