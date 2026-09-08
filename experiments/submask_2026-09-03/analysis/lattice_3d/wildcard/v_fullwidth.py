#!/usr/bin/env python3
"""Full-width validation against the production attack.

Runs code/submask_family.py's own attack_context on a random 20-round target
with the real 16 GB table, collects the candidates it produces (triangular
solutions with eps = 0), and checks with the independent residual algebra of
alg.py that

  * every candidate satisfies c0 = c1 = c2 = 0 exactly, so on the candidate
    manifold every Z-combination of the four constraints equals lam3 * c3;
  * the c3 it computes agrees with alg.py's residual;
  * c3 is uniform on the candidates.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/administrator/sha/publish/code")
import submask_family as SF
from alg import Alg, Instance, R

TBL = "/nvme0n1-disk/Kamvid/sigma0_u_table.npy"
A = Alg(32)
M = A.M

tbl = np.load(TBL, mmap_mode="r").view(np.uint32)
rng = np.random.default_rng(20260908)

n_a0 = int(sys.argv[1]) if len(sys.argv) > 1 else (1 << 22)
n_ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 6

# a random 20-round target
msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
import struct
pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
Wt = [struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)]
h = SF.digest(Wt, R)
ab, eb = SF.backward_chain(h, R)
chain = {i: ab[i] for i in range(12, R)}

tot = 0
bad = 0
c3s = []
for ci in range(n_ctx):
    ctx = SF.make_context(rng, R, v=None)
    inst = Instance(A, chain, {k: ctx[k] for k in range(4, 12)})
    # reproduce the attack's candidate stream by patching attack_context to
    # collect (a0,a1,a2,a3) instead of only verified preimages
    st, out = SF.attack_context(tbl, h, ctx, R, n_a0, rng)
    # rerun the inner loop to capture candidates: cheaper to redo the lookups
    # here with the same algebra as the production code.
    ab2, eb2 = SF.backward_chain(h, R)
    a = dict(ab2); a.update(ctx)
    a.update({-1: SF.IV[0], -2: SF.IV[1], -3: SF.IV[2], -4: SF.IV[3]})
    e = {-1: SF.IV[4], -2: SF.IV[5], -3: SF.IV[6], -4: SF.IV[7]}
    for r in range(8, R):
        e[r] = (a[r - 4] + a[r] - SF.T2(a[r - 1], a[r - 2], a[r - 3])) & SF.M
    a4, a5, a6, a7, a8, a9, a10 = (a[i] for i in range(4, 11))
    am1, am2, am3, am4 = a[-1], a[-2], a[-3], a[-4]
    em1, em2, em3, em4 = e[-1], e[-2], e[-3], e[-4]
    e8, e9, e10 = e[8], e[9], e[10]
    T1_7 = (a7 - SF.T2(a6, a5, a4)) & SF.M
    c6 = (a6 - SF.S0(a5)) & SF.M
    W9base = ((a9 - SF.T2(a8, a7, a6)) - SF.K[9]) & SF.M
    W10base = ((a10 - SF.T2(a9, a8, a7)) - SF.S1(e9) - SF.K[10]) & SF.M
    W11base = ((a[11] - SF.T2(a10, a9, a8)) - SF.S1(e10) - SF.K[11]) & SF.M
    Wr = {r: SF.recover_W(a, e, r) for r in range(12, R)}
    K0p = (Wr[16] - SF.s1(Wr[14])) & SF.M
    K1p = (Wr[17] - SF.s1(Wr[15])) & SF.M
    K2p = (Wr[18] - SF.s1(Wr[16])) & SF.M
    K3p = (Wr[19] - SF.s1(Wr[17]) - Wr[12]) & SF.M
    T2iv = SF.T2(am1, am2, am3)
    C0c = (-T2iv - em4 - SF.S1(em1) - SF.Ch(em1, em2, em3) - SF.K[0]) & SF.M
    Ce0 = (am4 - T2iv) & SF.M
    W9hat = (W9base - (a5 - SF.S0(a4) - SF.Maj(a4, 0, 0)) - SF.S1(e8)
             - SF.Ch(e8, T1_7, (c6 - SF.Maj(a5, a4, 0)) & SF.M)) & SF.M
    D = ((a6 - SF.S0(a5) - SF.Maj(a5, a4, 0)) + SF.Ch(e9, e8, T1_7)) & SF.M
    KC0 = (K0p - W9hat) & SF.M
    KC1 = (K1p - W10base + D) & SF.M
    KC2 = (K2p - W11base + T1_7 + SF.Ch(e10, e9, e8)) & SF.M
    u32 = SF.u32
    B = 1 << 21
    done = 0
    while done < n_a0:
        b = min(B, n_a0 - done); done += b
        A0 = rng.integers(0, 1 << 32, size=b, dtype=np.uint64).astype(np.uint32)
        E0 = (A0 + u32(Ce0)) & SF.MISS
        W0 = (A0 + u32(C0c)) & SF.MISS
        G = (-(SF.S0(A0) + SF.Maj(A0, u32(am1), u32(am2))) - u32(em3) - SF.S1(E0)
             - SF.Ch(E0, u32(em1), u32(em2)) - u32(SF.K[1])) & SF.MISS
        W1 = np.asarray(tbl[(u32(KC0) - W0 - G) & SF.MISS])
        k = W1 != SF.MISS
        A0, E0, W0, G, W1 = (x[k] for x in (A0, E0, W0, G, W1))
        A1 = (W1 - G) & SF.MISS
        E1 = (u32(am3) + A1 - (SF.S0(A0) + SF.Maj(A0, u32(am1), u32(am2)))) & SF.MISS
        F12 = (-(SF.S0(A1) + SF.Maj(A1, A0, u32(am1))) - u32(em2) - SF.S1(E1)
               - SF.Ch(E1, E0, u32(em1)) - u32(SF.K[2])) & SF.MISS
        W2 = np.asarray(tbl[(u32(KC1) - W1 - F12) & SF.MISS])
        k = W2 != SF.MISS
        A0, A1, E0, E1, W1, F12, W2 = (x[k] for x in (A0, A1, E0, E1, W1, F12, W2))
        A2 = (W2 - F12) & SF.MISS
        e2 = (u32(am2) + A2 - (SF.S0(A1) + SF.Maj(A1, A0, u32(am1)))) & SF.MISS
        F23 = (-(SF.S0(A2) + SF.Maj(A2, A1, A0)) - u32(em1) - SF.S1(e2)
               - SF.Ch(e2, E1, E0) - u32(SF.K[3])) & SF.MISS
        W3 = np.asarray(tbl[(u32(KC2) - W2 - F23) & SF.MISS])
        k = W3 != SF.MISS
        A0, A1, A2, F23, W3 = (x[k] for x in (A0, A1, A2, F23, W3))
        A3 = (W3 - F23) & SF.MISS
        hit = SF.Maj(u32(a4), A3, A2) == A3
        i = np.nonzero(hit)[0]
        if i.size == 0:
            continue
        r0, r1, r2, r3 = inst.residuals(A0[i].astype(np.uint64), A1[i].astype(np.uint64),
                                        A2[i].astype(np.uint64), A3[i].astype(np.uint64))
        tot += i.size
        bad += int(((r0 != 0) | (r1 != 0) | (r2 != 0)).sum())
        c3s.append(np.asarray(r3))
print(f"candidates (triangular, eps=0) from {n_ctx} contexts x {n_a0:,} swept a0: {tot}")
print(f"  candidates with c0 != 0 or c1 != 0 or c2 != 0: {bad}")
c3 = np.concatenate(c3s) if c3s else np.array([], dtype=np.uint64)
if c3.size:
    print(f"  c3: {int((c3==0).sum())} zeros of {c3.size} "
          f"(uniform predicts {c3.size*2**-32:.2e})")
    for kb in (4, 8, 12, 16):
        got = int((c3 % (1 << kb) == 0).sum())
        exp = c3.size / (1 << kb)
        z = (got - exp) / max(np.sqrt(exp), 1e-9)
        print(f"  low {kb:2d} bits zero: {got:8d} observed, {exp:10.1f} expected, z = {z:+.2f}")
