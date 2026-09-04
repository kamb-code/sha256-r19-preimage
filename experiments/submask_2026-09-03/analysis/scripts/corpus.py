#!/usr/bin/env python3
"""Build two corpora for pattern mining.

A) 19-round preimages found by the submask family, with the full internal state
   and the context they came from.  Anything the construction does NOT force is
   a potential lever.
B) 20-round candidates that satisfy C0, C1, C2 and the collapsed consistency
   condition, with their fourth-constraint residual C3 and the state that
   produced it.  Structure in C3 -- any dependence on a quantity we control --
   would cut the 20-round cost, which is a full 2^32 filter today.

    python3 corpus.py r19 <n_targets> <a0_per_ctx> out.npz
    python3 corpus.py r20 <n_targets> <a0_per_ctx> out.npz
"""
import struct
import sys
import time

import numpy as np

sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, U32, MISS, ZERO, K, IV, S0, S1, s0, s1, Ch, Maj, T2,
                            u32, digest, recover_W, backward_chain, make_context,
                            load_table, forward)

TABLE = "/nvme0n1-disk/Kamvid/sigma0_u_table.npy"
TABLE2 = "/nvme0n1-disk/Kamvid/sigma0_u_table_root2.npy"


def run(mode, n_targets, n_a0, out):
    t1 = np.load(TABLE, mmap_mode="r").view(np.uint32)
    R = 19 if mode == "r19" else 20
    rng = np.random.default_rng(20260904)
    cols = {k: [] for k in ("a0", "a1", "a2", "a3", "v", "ctx", "tgt", "c3",
                            "W0", "W1", "W2", "W3", "W4", "e0", "e1", "e2", "e3")}
    Wall = []
    t0 = time.time()
    for ti in range(n_targets):
        msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
        pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
        h = digest([struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)], R)
        ctx = make_context(rng, R)
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
        K3p = ((Wr[19] - s1(Wr[17]) - Wr[12]) & M) if R >= 20 else 0
        T2iv = T2(am1, am2, am3)
        C0c = (-T2iv - em4 - S1(em1) - Ch(em1, em2, em3) - K[0]) & M
        Ce0 = (am4 - T2iv) & M
        W9hat = (W9base - (a5 - S0(a4) - Maj(a4, 0, 0)) - S1(e8)
                 - Ch(e8, T1_7, (c6 - Maj(a5, a4, 0)) & M)) & M
        D = ((a6 - S0(a5) - Maj(a5, a4, 0)) + Ch(e9, e8, T1_7)) & M
        KC0 = (K0p - W9hat) & M
        KC1 = (K1p - W10base + D) & M
        KC2 = (K2p - W11base + T1_7 + Ch(e10, e9, e8)) & M

        B = 1 << 21
        done = 0
        while done < n_a0:
            b = min(B, n_a0 - done); done += b
            A0 = rng.integers(0, 1 << 32, size=b, dtype=np.uint64).astype(U32)
            E0 = (A0 + u32(Ce0)) & MISS
            W0 = (A0 + u32(C0c)) & MISS
            G = (-(S0(A0) + Maj(A0, u32(am1), u32(am2))) - u32(em3) - S1(E0)
                 - Ch(E0, u32(em1), u32(em2)) - u32(K[1])) & MISS
            W1 = np.asarray(t1[(u32(KC0) - W0 - G) & MISS]); keep = W1 != MISS
            A0, E0, W0, G, W1 = (x[keep] for x in (A0, E0, W0, G, W1))
            A1 = (W1 - G) & MISS
            E1 = (u32(am3) + A1 - (S0(A0) + Maj(A0, u32(am1), u32(am2)))) & MISS
            F12 = (-(S0(A1) + Maj(A1, A0, u32(am1))) - u32(em2) - S1(E1)
                   - Ch(E1, E0, u32(em1)) - u32(K[2])) & MISS
            W2 = np.asarray(t1[(u32(KC1) - W1 - F12) & MISS]); keep = W2 != MISS
            A0, A1, E0, E1, W0, W1, F12, W2 = (x[keep] for x in
                                               (A0, A1, E0, E1, W0, W1, F12, W2))
            A2 = (W2 - F12) & MISS
            e2 = (u32(am2) + A2 - (S0(A1) + Maj(A1, A0, u32(am1)))) & MISS
            F23 = (-(S0(A2) + Maj(A2, A1, A0)) - u32(em1) - S1(e2)
                   - Ch(e2, E1, E0) - u32(K[3])) & MISS
            W3 = np.asarray(t1[(u32(KC2) - W2 - F23) & MISS]); keep = W3 != MISS
            A0, A1, A2, E0, E1, e2, W0, W1, W2, F23, W3 = (x[keep] for x in
                (A0, A1, A2, E0, E1, e2, W0, W1, W2, F23, W3))
            A3 = (W3 - F23) & MISS
            hit = Maj(u32(a4), A3, A2) == A3
            idx = np.nonzero(hit)[0]
            if idx.size == 0:
                continue
            A0, A1, A2, A3, E0, E1, e2, W0, W1, W2, W3 = (x[idx] for x in
                (A0, A1, A2, A3, E0, E1, e2, W0, W1, W2, W3))
            e3 = (u32(am1) + A3 - (S0(A2) + Maj(A2, A1, A0))) & MISS
            W4 = (u32(a4) - (S0(A3) + Maj(A3, A2, A1)) - E0 - S1(e3)
                  - Ch(e3, e2, E1) - u32(K[4])) & MISS
            c3 = (s0(W4) + W3 - u32(K3p)) & MISS if R >= 20 else np.zeros_like(W4)
            for nm, arr in (("a0", A0), ("a1", A1), ("a2", A2), ("a3", A3),
                            ("W0", W0), ("W1", W1), ("W2", W2), ("W3", W3),
                            ("W4", W4), ("e0", E0), ("e1", E1), ("e2", e2),
                            ("e3", e3), ("c3", c3)):
                cols[nm].append(np.asarray(arr, dtype=np.uint32))
            n = A0.size
            cols["v"].append(np.full(n, a4, np.uint32))
            cols["ctx"].append(np.full(n, ti, np.uint32))
            cols["tgt"].append(np.frombuffer(h[:4], dtype=">u4").astype(np.uint32).repeat(n))
            if R == 19:
                for j in range(min(n, 40)):
                    aa = dict(a); aa.update({0: int(A0[j]), 1: int(A1[j]),
                                             2: int(A2[j]), 3: int(A3[j])})
                    ee = dict(e)
                    for rr in range(R):
                        ee[rr] = (aa[rr - 4] + aa[rr] - T2(aa[rr - 1], aa[rr - 2],
                                                           aa[rr - 3])) & M
                    Wm = [recover_W(aa, ee, rr) for rr in range(16)]
                    assert digest(Wm, R) == h, "corpus contains a non-preimage"
                    Wall.append(Wm)
        if (ti + 1) % 5 == 0:
            got = sum(len(x) for x in cols["a0"])
            print(f"  target {ti+1}/{n_targets}: {got:,} rows  [{time.time()-t0:.0f}s]", flush=True)
    data = {k: np.concatenate(v) if v else np.zeros(0, np.uint32) for k, v in cols.items()}
    data["messages"] = np.array(Wall, dtype=np.uint32) if Wall else np.zeros((0, 16), np.uint32)
    np.savez_compressed(out, **data)
    print(f"wrote {out}: {len(data['a0']):,} rows, {len(data['messages']):,} full messages "
          f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
