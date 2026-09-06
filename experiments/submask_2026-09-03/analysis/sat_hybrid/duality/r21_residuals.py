#!/usr/bin/env python3
"""Part (c)(iii): run the real triangular pipeline at R=21 with the 16 GB table in
symmetric family-type contexts and measure the residuals c3 (W19) and c4 (W20) on
candidates that pass C0, C1, C2 and the collapse.  Random digests.

Usage: r21_residuals.py <kind> <n_a0_log2> <seed>
kind in: family, quad, allequal (a4..a12 = v with Sigma0(v)-v = -1 so e8..e12 = -1)
"""
import sys, time, json, struct
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, u32, U32, MISS, ZERO,
                            digest, backward_chain, recover_W, forward)

np.seterr(over="ignore")
TABLE = "/nvme0n1-disk/Kamvid/sigma0_u_table_rebuilt.npy"
R = 21


def find_v_allequal():
    """v with Sigma0(v) - v == -1 mod 2^32 (then a4..a12 = v gives e8..e12 = -1)."""
    out = []
    B = 1 << 26
    for lo in range(0, 1 << 32, B):
        v = np.arange(lo, lo + B, dtype=np.uint64).astype(U32)
        r = (S0(v) - v)
        hit = np.nonzero(r == MISS)[0]
        out += [int(v[i]) for i in hit]
    return out


def make_ctx(kind, rng, v=None):
    ctx = {i: int(rng.integers(0, 1 << 32, dtype=np.uint64)) for i in range(4, 13)}
    if v is None:
        v = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    if kind == "family":
        m = 5
    elif kind == "quad":
        m = 7
    elif kind == "allequal":
        for i in range(4, 13): ctx[i] = v
        return ctx
    for i in range(4, m + 1): ctx[i] = v
    for i in range(8, 8 + (m - 3)):
        ctx[i] = (M - ctx[i - 4] + S0(ctx[i - 1]) + Maj(ctx[i - 1], ctx[i - 2], ctx[i - 3])) & M
    return ctx


def candidates(tbl, h, ctx, n_a0, rng):
    """Triangular solve + collapse at R=21; returns arrays A0..A3 of candidates and the
    per-context constants needed for c3, c4."""
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
    assert e8 == M and e9 == M and a4 == a5, "context is not in the family"
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
    K4p = (Wr[20] - s1(Wr[18]) - Wr[13]) & M
    T2iv = T2(am1, am2, am3)
    C0c = (-T2iv - em4 - S1(em1) - Ch(em1, em2, em3) - K[0]) & M
    Ce0 = (am4 - T2iv) & M
    W9hat = (W9base - (a5 - S0(a4) - Maj(a4, 0, 0)) - S1(e8)
             - Ch(e8, T1_7, (c6 - Maj(a5, a4, 0)) & M)) & M
    D = ((a6 - S0(a5) - Maj(a5, a4, 0)) + Ch(e9, e8, T1_7)) & M
    KC0 = (K0p - W9hat) & M
    KC1 = (K1p - W10base + D) & M
    KC2 = (K2p - W11base + T1_7 + Ch(e10, e9, e8)) & M
    out = []
    B = 1 << 20
    done = 0; st = dict(a0=0, sol=0, eps0=0)
    while done < n_a0:
        b = min(B, n_a0 - done); done += b; st['a0'] += b
        A0 = rng.integers(0, 1 << 32, size=b, dtype=np.uint64).astype(U32)
        E0 = (A0 + u32(Ce0)) & MISS
        W0 = (A0 + u32(C0c)) & MISS
        G = (-(S0(A0) + Maj(A0, u32(am1), u32(am2))) - u32(em3) - S1(E0)
             - Ch(E0, u32(em1), u32(em2)) - u32(K[1])) & MISS
        W1 = np.asarray(tbl[(u32(KC0) - W0 - G) & MISS])
        keep = W1 != MISS
        A0, E0, W0, G, W1 = (x[keep] for x in (A0, E0, W0, G, W1))
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
        W3 = np.asarray(tbl[(u32(KC2) - W2 - F23) & MISS])
        keep = W3 != MISS
        A0, A1, A2, E0, E1, e2, F23, W3 = (x[keep] for x in (A0, A1, A2, E0, E1, e2, F23, W3))
        A3 = (W3 - F23) & MISS
        st['sol'] += A0.size
        hit = Maj(u32(a4), A3, A2) == A3
        st['eps0'] += int(hit.sum())
        idx = np.nonzero(hit)[0]
        for j in idx:
            out.append((int(A0[j]), int(A1[j]), int(A2[j]), int(A3[j])))
    consts = dict(a=a, e=e, K3p=K3p, K4p=K4p)
    return out, st, consts


def residuals(cands, consts):
    a = consts['a']; res = []
    for (x0, x1, x2, x3) in cands:
        aa = dict(a); aa.update({0: x0, 1: x1, 2: x2, 3: x3})
        ee = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
        for r in range(R):
            ee[r] = (aa[r - 4] + aa[r] - T2(aa[r - 1], aa[r - 2], aa[r - 3])) & M
        W = [recover_W(aa, ee, r) for r in range(R)]
        c3 = (s0(W[4]) + W[3] - consts['K3p']) & M
        c4 = (s0(W[5]) + W[4] - consts['K4p']) & M
        # sanity: C0..C2 hold exactly
        ok = all(((s1(W[14 + j]) + W[9 + j] + s0(W[1 + j]) + W[j]) & M) == recover_W(aa, ee, 16 + j) for j in range(3))
        res.append((c3, c4, ok))
    return res


def main():
    kind, log2n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    t0 = time.time()
    tbl = np.load(TABLE, mmap_mode="r").view(np.uint32)
    rng = np.random.default_rng(seed)
    vs = None
    if kind == "allequal":
        vs = find_v_allequal()
        print(f"v with Sigma0(v)-v = -1: {[hex(v) for v in vs]}  [{time.time()-t0:.0f}s]", flush=True)
        if not vs:
            print("no such v: the all-equal context cannot satisfy e8=e9=-1; skipping pipeline"); return
    n_ctx = 8
    allres = []; tot = dict(a0=0, sol=0, eps0=0)
    for ci in range(n_ctx):
        msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
        pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
        h = digest([struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)], R)
        v = vs[ci % len(vs)] if vs else None
        ctx = make_ctx(kind, rng, v)
        cands, st, consts = candidates(tbl, h, ctx, 1 << log2n, rng)
        for k in tot: tot[k] += st[k]
        res = residuals(cands, consts)
        allres += res
        print(f"ctx {ci}: a0 {st['a0']:,} sol {st['sol']:,} eps0 {st['eps0']:,} "
              f"C0-C2 exact on all: {all(r[2] for r in res)}  [{time.time()-t0:.0f}s]", flush=True)
    c3 = np.array([r[0] for r in allres], dtype=np.uint64); c4 = np.array([r[1] for r in allres], dtype=np.uint64)
    n = len(allres)
    print(f"\n{kind}: {n:,} candidates from {tot['a0']:,} swept a0 ({tot['sol']/max(tot['a0'],1):.4f} sol/a0, "
          f"collapse {tot['eps0']/max(tot['sol'],1):.3e})")
    for k in (4, 8, 12, 16):
        z3 = int(((c3 & ((1 << k) - 1)) == 0).sum()); z4 = int(((c4 & ((1 << k) - 1)) == 0).sum())
        zj = int((((c3 & ((1 << k) - 1)) == 0) & ((c4 & ((1 << k) - 1)) == 0)).sum())
        exp = n / 2 ** k; expj = n / 4 ** k
        print(f"  low {k:2d} bits zero: c3 {z3} (exp {exp:.1f}), c4 {z4} (exp {exp:.1f}), both {zj} (exp {expj:.2f})")
    # chi-square on low 8 bits, and on c3^c4, c3-c4
    for name, arr in (("c3", c3), ("c4", c4), ("c3^c4", c3 ^ c4), ("c3-c4", (c3 - c4) & np.uint64(M))):
        cnt = np.bincount((arr & np.uint64(255)).astype(np.int64), minlength=256)
        chi = float(((cnt - n / 256) ** 2 / (n / 256)).sum())
        z = (chi - 255) / np.sqrt(2 * 255)
        print(f"  chi2 low 8 bits of {name}: {chi:.1f} (df 255, z={z:+.2f}); distinct values {len(np.unique(arr)):,}/{n:,}")
    hi = np.stack([c3 >> np.uint64(24), c4 >> np.uint64(24)]).astype(np.float64)
    print(f"  corr(high byte c3, high byte c4) = {np.corrcoef(hi)[0,1]:+.4f}")
    json.dump(dict(kind=kind, n=n, tot=tot, c3=[int(x) for x in c3], c4=[int(x) for x in c4]),
              open(f"r21_{kind}_{seed}.json", "w"))
    print(f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
