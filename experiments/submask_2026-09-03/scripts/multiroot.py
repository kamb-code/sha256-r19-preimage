#!/usr/bin/env python3
"""Multi-root tables for the submask family.

The published table keeps ONE representative u per value of sigma0(u)-u and
discards the rest.  Since u -> sigma0(u)-u behaves as a random map, the number
of roots of a random target is Poisson(1): 36.8% of targets have none, 36.8%
have one, 26.4% have two or more.  Keeping one root therefore finds only
E[min(r,1)] = 0.632 of the E[r] = 1.0 available continuations at each of the
three lookups, so a single-representative solver sees (0.632)^3 = 0.253 of the
solutions and a fully branching one would see 1.0 -- a factor of 3.96.

This builds a SECOND-root table (one more 16 GB array) and measures the actual
gain from branching over up to two roots per lookup, which should recover
(0.896/0.632)^3 = 2.85 of the 3.96.

    python3 multiroot.py --build      # ~5 min, writes the second-root table
    python3 multiroot.py --measure    # compares 1-root and 2-root yields
"""
import argparse, os, struct, sys, time
import numpy as np

sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, U32, MISS, ZERO, K, IV, S0, S1, s0, s1, Ch, Maj, T2,
                            u32, digest, recover_W, backward_chain, make_context,
                            load_table)

FIRST = "/nvme0n1-disk/Kamvid/sigma0_u_table.npy"
SECOND = "/nvme0n1-disk/Kamvid/sigma0_u_table_root2.npy"
CHUNK = 1 << 24


def Tmap(u):
    return (s0(u) - u) & MISS


def build():
    first = np.load(FIRST, mmap_mode="r").view(np.uint32)
    second = np.full(1 << 32, M, dtype=np.uint32)
    t0 = time.time()
    for s in range(0, 1 << 32, CHUNK):
        u = np.arange(s, s + CHUNK, dtype=np.uint64).astype(U32)
        v = Tmap(u)
        f = np.asarray(first[v])
        m = (f != u) & (np.asarray(second[v]) == MISS)   # not the stored root, slot free
        if m.any():
            second[v[m]] = u[m]
        if (s // CHUNK) % 64 == 63:
            print(f"  {(s + CHUNK) / 2**32 * 100:5.1f}%  {time.time() - t0:6.0f}s", flush=True)
    n1 = int((np.asarray(first[::64]) != MISS).sum()) * 64
    n2 = int((second[::64] != MISS).sum()) * 64
    print(f"first-root coverage  ~{n1 / 2**32 * 100:.2f}%   (Poisson(1): 63.21%)")
    print(f"second-root coverage ~{n2 / 2**32 * 100:.2f}%   (Poisson(1): 26.42%)")
    np.save(SECOND, second)
    print(f"saved {SECOND}  [{time.time() - t0:.0f}s]")


def attack_multiroot(t1, t2, h, ctx, R, n_a0, rng, use_two):
    """Same triangular solve as submask_family.attack_context, but branching over
    the first and (optionally) second root at each of the three lookups."""
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
    T2iv = T2(am1, am2, am3)
    C0c = (-T2iv - em4 - S1(em1) - Ch(em1, em2, em3) - K[0]) & M
    Ce0 = (am4 - T2iv) & M
    W9hat = (W9base - (a5 - S0(a4) - Maj(a4, 0, 0)) - S1(e8)
             - Ch(e8, T1_7, (c6 - Maj(a5, a4, 0)) & M)) & M
    D = ((a6 - S0(a5) - Maj(a5, a4, 0)) + Ch(e9, e8, T1_7)) & M
    KC0 = (K0p - W9hat) & M
    KC1 = (K1p - W10base + D) & M
    KC2 = (K2p - W11base + T1_7 + Ch(e10, e9, e8)) & M

    def roots(tgt):
        """Stack the branches: returns (index_into_input, root)."""
        r1 = np.asarray(t1[tgt]); ok1 = r1 != MISS
        idx = [np.nonzero(ok1)[0]]; val = [r1[ok1]]
        if use_two:
            r2 = np.asarray(t2[tgt]); ok2 = r2 != MISS
            idx.append(np.nonzero(ok2)[0]); val.append(r2[ok2])
        return np.concatenate(idx), np.concatenate(val)

    st = dict(a0=0, sol=0, ver=0)
    B = 1 << 19
    done = 0
    while done < n_a0:
        b = min(B, n_a0 - done); done += b; st['a0'] += b
        A0 = rng.integers(0, 1 << 32, size=b, dtype=np.uint64).astype(U32)
        E0 = (A0 + u32(Ce0)) & MISS
        W0 = (A0 + u32(C0c)) & MISS
        G = (-(S0(A0) + Maj(A0, u32(am1), u32(am2))) - u32(em3) - S1(E0)
             - Ch(E0, u32(em1), u32(em2)) - u32(K[1])) & MISS
        i, W1 = roots((u32(KC0) - W0 - G) & MISS)
        A0, E0, G = A0[i], E0[i], G[i]
        A1 = (W1 - G) & MISS
        E1 = (u32(am3) + A1 - (S0(A0) + Maj(A0, u32(am1), u32(am2)))) & MISS
        F12 = (-(S0(A1) + Maj(A1, A0, u32(am1))) - u32(em2) - S1(E1)
               - Ch(E1, E0, u32(em1)) - u32(K[2])) & MISS
        i, W2 = roots((u32(KC1) - W1 - F12) & MISS)
        A0, A1, E0, E1, F12 = A0[i], A1[i], E0[i], E1[i], F12[i]
        A2 = (W2 - F12) & MISS
        e2 = (u32(am2) + A2 - (S0(A1) + Maj(A1, A0, u32(am1)))) & MISS
        F23 = (-(S0(A2) + Maj(A2, A1, A0)) - u32(em1) - S1(e2)
               - Ch(e2, E1, E0) - u32(K[3])) & MISS
        i, W3 = roots((u32(KC2) - W2 - F23) & MISS)
        A0, A1, A2, F23 = A0[i], A1[i], A2[i], F23[i]
        A3 = (W3 - F23) & MISS
        st['sol'] += A0.size
        if A0.size == 0:
            continue
        hit = Maj(u32(a4), A3, A2) == A3
        for j in np.nonzero(hit)[0]:
            aa = dict(a)
            aa.update({0: int(A0[j]), 1: int(A1[j]), 2: int(A2[j]), 3: int(A3[j])})
            ee = dict(e)
            for rr in range(R):
                ee[rr] = (aa[rr - 4] + aa[rr] - T2(aa[rr - 1], aa[rr - 2], aa[rr - 3])) & M
            Wm = [recover_W(aa, ee, rr) for rr in range(16)]
            if digest(Wm, R) == h:
                st['ver'] += 1
    return st


def measure(n_a0, targets):
    t1 = np.load(FIRST, mmap_mode="r").view(np.uint32)
    t2 = np.load(SECOND, mmap_mode="r").view(np.uint32)
    R = 19
    res = {}
    for use_two in (False, True):
        rng = np.random.default_rng(31337)
        tot = dict(a0=0, sol=0, ver=0)
        t0 = time.time()
        for ti in range(targets):
            msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
            pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
            h = digest([struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)], R)
            st = attack_multiroot(t1, t2, h, make_context(rng, R), R, n_a0, rng, use_two)
            for k in tot:
                tot[k] += st[k]
        el = time.time() - t0
        lab = "two roots" if use_two else "one root "
        res[use_two] = tot
        print(f"{lab}: {tot['a0']:,} a0 -> {tot['sol']:,} solutions "
              f"({tot['sol']/tot['a0']:.4f}/a0), {tot['ver']:,} verified preimages, "
              f"{el:.0f}s  [{tot['a0']/max(tot['ver'],1):,.0f} a0 each]", flush=True)
    g_sol = res[True]['sol'] / max(res[False]['sol'], 1)
    g_ver = res[True]['ver'] / max(res[False]['ver'], 1)
    print(f"\ngain from the second root: solutions {g_sol:.2f}x, preimages {g_ver:.2f}x")
    print(f"  predicted for 2 roots: (0.896/0.632)^3 = {(0.896/0.632)**3:.2f}x")
    print(f"  ceiling for all roots: (1.000/0.632)^3 = {(1/0.6321)**3:.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--a0", type=int, default=1 << 21)
    ap.add_argument("--targets", type=int, default=8)
    a = ap.parse_args()
    if a.build:
        build()
    if a.measure:
        measure(a.a0, a.targets)
