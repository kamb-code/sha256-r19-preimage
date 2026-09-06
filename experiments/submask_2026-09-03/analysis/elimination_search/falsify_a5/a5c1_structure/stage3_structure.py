#!/usr/bin/env python3
"""Stage 3: STRUCTURE of the a5 -> C1 dependence for a given condition set.

  (a) exact closed form of the edge, verified against the full W-recovery;
  (b) 32x32 sensitivity matrix S[i][j] = P(bit j of idxC1 flips | bit i of a5
      flips), deterministic cells, GF(2) rank of the majority Jacobian,
      GF(2)-linearity test, distinct additive differences per input bit;
  (c) exhaustive 2^32 scan of the residual rho(a5) = idxC1(a5) - idxC1(a5p)
      at fixed state and a5p: counts of rho = 0 and rho = 0 mod 2^k, k=8,16,24;
  (d) full histogram of idxC1 over all 2^32 a5 (one state): number of distinct
      values, max multiplicity, sum n_v^2, hence E_{a5p} P(rho = 0).
usage: stage3_structure.py "<defs dict>" [--scan N] [--hist]
"""
import sys, ast, time, os
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from a5edge import (realise, targets, make_states, rand_words, legal, popcount,
                    M, U, MM, R, S0, S1, Ch, Maj, T2, IV)

NPROC = 14
CH = 1 << 24


def full_state(defs, rng):
    st = make_states(rng, 1)
    free = {i: rand_words(rng, 1) for i in range(4, 12)}
    free.update({i: st[i] for i in range(12, R)})
    ctx = realise(defs, free, 1)
    a = {i: int(st[i][0]) for i in list(range(4)) + list(range(12, R)) + [-1, -2, -3, -4]}
    a.update({i: int(ctx[i][0]) for i in range(4, 12)})
    return a


def idx_c1_vec(a_scalar, a5):
    """exact C1 table index over an array of a5 values, other words scalar."""
    n = a5.shape[0]
    a = {i: np.broadcast_to(U(v), (n,)) for i, v in a_scalar.items()}
    a[5] = a5
    _, _, T = targets(a)
    return T[4]


def closed_form(a, a5):
    """-Sigma0(a5) - Maj(a5,a4,a3) + Sigma1(a5+c9) + Ch(a5+c9, e8, e7(a5)) + const."""
    n = a5.shape[0]
    g = lambda i: np.broadcast_to(U(a[i]), (n,))
    c9 = (g(9) - T2(g(8), g(7), g(6))) & MM
    e8 = (g(4) + g(8) - T2(g(7), g(6), g(5))) & MM      # context value (a5 collapsed)
    e7 = (g(3) + g(7) - S0(g(6)) - Maj(g(6), a5, g(4))) & MM
    e9 = (a5 + c9) & MM
    return (-S0(a5) - Maj(a5, g(4), g(3)) + S1(e9) + Ch(e9, e8, e7)) & MM


def gf2_rank(rows):
    rows = [int(r) for r in rows]
    rank = 0
    for bit in range(32):
        piv = None
        for k in range(rank, len(rows)):
            if (rows[k] >> bit) & 1:
                piv = k; break
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for k in range(len(rows)):
            if k != rank and (rows[k] >> bit) & 1:
                rows[k] ^= rows[rank]
        rank += 1
    return rank


def structure(defs, rng, n=1 << 14):
    st = make_states(rng, n)
    free = {i: rand_words(rng, n) for i in range(4, 12)}
    free.update({i: st[i] for i in range(12, R)})
    ctx = realise(defs, free, n)
    a = {}
    a.update({i: st[i] for i in list(range(4)) + list(range(12, R)) + [-1, -2, -3, -4]})
    a.update(ctx)
    # (a) closed form check
    _, _, T = targets(a)
    f0 = T[4]
    a2 = dict(a); a2[5] = np.zeros(n, dtype=U)
    _, _, T0 = targets(a2)
    def cf_vec(a, a5):
        c9 = (a[9] - T2(a[8], a[7], a[6])) & MM
        e8 = (a[4] + a[8] - T2(a[7], a[6], a[5])) & MM
        e7 = (a[3] + a[7] - S0(a[6]) - Maj(a[6], a5, a[4])) & MM
        e9 = (a5 + c9) & MM
        return (-S0(a5) - Maj(a5, a[4], a[3]) + S1(e9) + Ch(e9, e8, e7)) & MM
    const = (T0[4] - cf_vec(a2, np.zeros(n, dtype=U))) & MM
    ok = np.array_equal(f0, (const + cf_vec(a, a[5])) & MM)
    print(f"closed form idxC1 = const - S0(a5) - Maj(a5,a4,a3) + S1(a5+c9) + Ch(a5+c9,e8,e7(a5)): "
          f"{'EXACT' if ok else 'FAIL'} on {n} states")
    e8vals = np.unique((a[4] + a[8] - T2(a[7], a[6], a[5])) & MM)
    print(f"  e8 takes {e8vals.size} value(s)" + (f": 0x{int(e8vals[0]):08x}" if e8vals.size == 1 else ""))
    # (b) sensitivity matrix
    S = np.zeros((32, 32))
    ndist = np.zeros(32, dtype=np.int64)
    for i in range(32):
        ai = dict(a); ai[5] = a[5] ^ U(1 << i)
        _, _, Ti = targets(ai)
        d = Ti[4] ^ f0
        for j in range(32):
            S[i, j] = ((d >> U(j)) & U(1)).mean()
        ndist[i] = np.unique((Ti[4] - f0) & MM).size
    det = ((S == 0) | (S == 1))
    print(f"  sensitivity matrix: mean {S.mean():.3f} (x32 = {32*S.mean():.2f} bits/flip); "
          f"deterministic cells {det.sum()}/1024; always-flip cells {(S == 1).sum()}; never {(S == 0).sum()}")
    fully_det_out = [j for j in range(32) if det[:, j].all()]
    print(f"  output bits whose sensitivity to every input bit is deterministic (GF(2)-affine bits): {fully_det_out}")
    fully_det_in = [i for i in range(32) if det[i, :].all()]
    print(f"  input bits with fully deterministic effect: {fully_det_in}")
    J = [sum(1 << j for j in range(32) if S[i, j] > 0.5) for i in range(32)]
    J1 = [sum(1 << j for j in range(32) if S[i, j] == 1.0) for i in range(32)]
    print(f"  GF(2) rank of majority Jacobian (S>0.5): {gf2_rank(J)}; of the always-flip part: {gf2_rank(J1)}")
    print(f"  distinct additive differences f(a5^e_i)-f(a5) over {n} states, per input bit: "
          f"min {ndist.min()} median {int(np.median(ndist))} max {ndist.max()}  (bitwise/borrow-free map would give <= 2)")
    # GF(2)-linearity: f(x^d)^f(x) == f(y^d)^f(y)?
    d = rand_words(rng, n)
    ad = dict(a); ad[5] = a[5] ^ d
    _, _, Td = targets(ad)
    ay = dict(a); ay[5] = rand_words(rng, n)
    _, _, Ty = targets(ay)
    ayd = dict(ay); ayd[5] = ay[5] ^ d
    _, _, Tyd = targets(ayd)
    L = (Td[4] ^ f0) ^ (Tyd[4] ^ Ty[4])
    print(f"  GF(2)-linearity: P[whole-word derivative independent of base point] = {(L == 0).mean():.2e}; "
          f"per-bit agreement: min {1 - max(((L >> U(j)) & U(1)).mean() for j in range(32)):.3f} "
          f"bit0 {1 - ((L & U(1))).mean():.3f}")
    # single-flip residual weight in the additive sense
    np.set_printoptions(linewidth=200, precision=2)
    print("  S (rows: flipped a5 bit i = 0..31; cols: idxC1 bit j), rounded:")
    for i in range(32):
        print("   ", "".join('#' if S[i, j] == 1 else ('.' if S[i, j] == 0 else ('+' if S[i, j] > 0.5 else '-' if S[i, j] > 0.05 else ',')) for j in range(32)))
    return S


# ---------------- exhaustive scan ----------------
_G = {}


def fast_params(a, rng):
    """Closed-form parameters for a fixed scalar state, verified EXACT against
    the full W-recovery on 2^20 random a5 (and a5 = 0)."""
    g = lambda i: U(a[i] & M)
    c9 = int((g(9) - T2(g(8), g(7), g(6))) & MM)
    e8 = int((g(4) + g(8) - T2(g(7), g(6), g(5))) & MM)
    c7 = int((g(3) + g(7) - S0(g(6))) & MM)
    p = dict(c9=c9, e8=e8, c7=c7, a4=a[4], a3=a[3], a6=a[6], const=0)
    f0 = int(idx_c1_vec(a, np.zeros(1, dtype=U))[0])
    p['const'] = int((U(f0) - fast_f(p, np.zeros(1, dtype=U))[0]) & MM)
    x = rand_words(rng, 1 << 20)
    assert np.array_equal(fast_f(p, x), idx_c1_vec(a, x)), "closed form mismatch"
    return p


def fast_f(p, a5):
    n = a5.shape[0]
    b = lambda v: np.broadcast_to(U(v & M), (n,))
    e9 = (a5 + b(p['c9'])) & MM
    e7 = (b(p['c7']) - Maj(b(p['a6']), a5, b(p['a4']))) & MM
    return (b(p['const']) - S0(a5) - Maj(a5, b(p['a4']), b(p['a3'])) + S1(e9)
            + Ch(e9, b(p['e8']), e7)) & MM


def _scan_chunk(lo):
    a5 = np.arange(lo, lo + CH, dtype=np.uint64).astype(U)
    f = fast_f(_G['p'], a5)
    rho = (f - U(_G['fp'])) & MM
    c = [int((rho == 0).sum())]
    for k in (8, 16, 24):
        c.append(int(((rho & U((1 << k) - 1)) == 0).sum()))
    # also the XOR-residual weight distribution: how many a5 have rho with <= 4 bits set
    c.append(int((popcount(rho) <= 4).sum()))
    return c


def scan(defs, rng, a5p=None, tag=""):
    a = full_state(defs, rng)
    if a5p is None:
        a5p = a[5]
    p = fast_params(a, rng)
    fp = int(fast_f(p, np.array([a5p], dtype=U))[0])
    _G['p'] = p; _G['a5p'] = a5p; _G['fp'] = fp
    t0 = time.time()
    with Pool(NPROC) as p:
        parts = p.map(_scan_chunk, range(0, 1 << 32, CH), chunksize=4)
    tot = np.array(parts).sum(axis=0)
    print(f"  scan{tag} a5p=0x{a5p:08x}: #rho=0: {tot[0]} (bijection: 1)   "
          f"#rho=0 mod 2^8: {tot[1]} (2^24={1<<24})  mod 2^16: {tot[2]} (2^16={1<<16})  "
          f"mod 2^24: {tot[3]} (2^8=256)   #popcount(rho)<=4: {tot[4]} (uniform: {sum(__import__('math').comb(32,k) for k in range(5))})   {time.time()-t0:.0f}s")
    return tot


def _hist_chunk(lo):
    a5 = np.arange(lo, lo + CH, dtype=np.uint64).astype(U)
    f = fast_f(_G['p'], a5)
    mm = np.memmap(_G['path'], dtype=U, mode='r+', shape=(1 << 32,))
    mm[lo:lo + CH] = f
    mm.flush()
    return 0


def histogram(defs, rng):
    a = full_state(defs, rng)
    path = '/dev/shm/a5c1_f.u32'
    mm = np.memmap(path, dtype=U, mode='w+', shape=(1 << 32,))
    del mm
    _G['p'] = fast_params(a, rng); _G['path'] = path
    t0 = time.time()
    with Pool(NPROC) as p:
        p.map(_hist_chunk, range(0, 1 << 32, CH), chunksize=4)
    print(f"  histogram: f computed over 2^32 a5 in {time.time()-t0:.0f}s; sorting ...", flush=True)
    arr = np.fromfile(path, dtype=U)
    os.unlink(path)
    arr.sort()
    print(f"  sorted in {time.time()-t0:.0f}s", flush=True)
    # run lengths, chunked
    B = 1 << 28
    distinct = 0; sumsq = 0; maxn = 0
    carry = 0; prev = None
    for lo in range(0, 1 << 32, B):
        c = arr[lo:lo + B]
        change = np.nonzero(c[1:] != c[:-1])[0] + 1
        bounds = np.concatenate(([0], change, [c.size]))
        lens = np.diff(bounds)
        if prev is not None and c[0] == prev:
            lens[0] += carry
        else:
            if prev is not None:
                distinct += 1; sumsq += carry * carry; maxn = max(maxn, carry)
        carry = int(lens[-1]); prev = c[-1]
        if lens.size > 1:
            l = lens[:-1].astype(np.int64)
            distinct += l.size; sumsq += int((l * l).sum()); maxn = max(maxn, int(l.max()))
    distinct += 1; sumsq += carry * carry; maxn = max(maxn, carry)
    del arr
    print(f"  distinct values {distinct} = {distinct / 2**32:.6f} x 2^32 (bijection 1.0, random map 0.632); "
          f"max multiplicity {maxn}; sum n_v^2 = {sumsq} = {sumsq / 2**32:.4f} x 2^32 "
          f"(bijection 1, random map 2); E_a5p P(rho=0) = {sumsq / 2**64:.3e} = {sumsq / 2**32:.3f} x 2^-32")
    return distinct, maxn, sumsq


if __name__ == "__main__":
    defs = ast.literal_eval(sys.argv[1])
    nscan = 0; do_hist = False
    if '--scan' in sys.argv:
        nscan = int(sys.argv[sys.argv.index('--scan') + 1])
    if '--hist' in sys.argv:
        do_hist = True
    rng = np.random.default_rng(7)
    print("defs =", defs, "legal:", legal(defs, rng))
    structure(defs, rng)
    for s in range(nscan):
        scan(defs, rng, tag=f"#{s}")
    if nscan >= 3:
        # also a5p = 0 and a5p = -1 on one more state
        scan(defs, rng, a5p=0, tag=" (a5p=0)")
        scan(defs, rng, a5p=M, tag=" (a5p=-1)")
    if do_hist:
        histogram(defs, rng)
