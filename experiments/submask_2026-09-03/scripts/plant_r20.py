#!/usr/bin/env python3
"""End-to-end check of the R=20 pipeline on PLANTED preimages.

Build message blocks whose internal state lies in the submask family
(a4 = a5 = v, e8 = e9 = 0xFFFFFFFF), hash them for 20 rounds, and run the exact
sweep code of gpu_submask.py (CPU, on-disk tables) over a small a0 window that
contains the true a0.  The pipeline must report a VERIFIED hit whenever the
true W1, W2, W3 are among the stored table roots, and must never report one
otherwise.  This is the only test that exercises the fourth-constraint (C3)
formula on a genuine preimage.
"""
import os, struct, sys
import numpy as np
import torch
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import M, K, IV, S0, S1, s0, Ch, Maj, T2, digest, recover_W
import gpu_submask as G

R = 20
TDIR = "/nvme0n1-disk/Kamvid"
tbls = G.load_tables_cpu([f"{TDIR}/sigma0_u_table.npy", f"{TDIR}/sigma0_u_table_root2.npy"])
dev = torch.device("cpu")
G.CHUNK = 1 << 12


def plant(rng):
    """Random block steered so that a4 = a5 = v, e8 = e9 = -1."""
    v = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    W = [int(x) for x in rng.integers(0, 1 << 32, size=16, dtype=np.uint64)]
    a = {-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]}
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(16):
        base = (e[r - 4] + S1(e[r - 1]) + Ch(e[r - 1], e[r - 2], e[r - 3]) + K[r]) & M
        T2r = T2(a[r - 1], a[r - 2], a[r - 3])
        if r == 2:                           # a2 free but recorded
            pass
        elif r == 3:                         # force Maj(v, a3, a2) == a3: a3 = a2 on mask bits, v elsewhere
            m = int(rng.integers(0, 1 << 32, dtype=np.uint64))
            a3 = ((a[2] & m) | (v & ~m)) & M
            W[r] = (a3 - T2r - base) & M
        elif r in (4, 5):                    # force a_r = v
            W[r] = (v - T2r - base) & M
        elif r == 8:                         # force e8 = -1  (e8 = a4 + T1_8)
            W[r] = ((M - a[4]) - base) & M
        elif r == 9:                         # force e9 = -1
            W[r] = ((M - a[5]) - base) & M
        T1 = (base + W[r]) & M
        a[r] = (T1 + T2r) & M
        e[r] = (a[r - 4] + T1) & M
    assert a[4] == v and a[5] == v and e[8] == M and e[9] == M
    assert Maj(v, a[3], a[2]) == a[3], 'submask condition not planted'
    return W, a


def in_tables(u):
    v = (s0(u) - u) & M
    return any(int(t[v]) & M == u for t in tbls)


rng = np.random.default_rng(2026)
found = missing_root = wrong = 0
N = 16
for i in range(N):
    W, a_true = plant(rng)
    h = digest(W, R)
    ctx = {r: a_true[r] for r in range(4, 12)}
    cc = G.context_constants(h, ctx, R)
    # are the true W1, W2, W3 recoverable from the stored roots?
    ok_roots = all(in_tables(W[r]) for r in (1, 2, 3))
    a0 = a_true[0]
    st, kc, hits = G.sweep_context(tbls, cc, h, R, dev, 4096, 8, True, lambda m: None,
                                   a0_start=(a0 - 2048) & M)
    got = any(Wm == W for Wm in hits)
    status = ("FOUND" if got else "not found")
    if got and not ok_roots: status += "  (?? found despite a missing root)"
    if got: found += 1
    elif ok_roots: wrong += 1; status += "  <-- roots present but NOT found: pipeline error"
    else: missing_root += 1; status += "  (true root not stored; expected)"
    print(f"plant {i:>2}: v=0x{a_true[4]:08x} a0=0x{a0:08x} roots_stored={ok_roots} ver={st['ver']} -> {status}")
print(f"\n{found} found, {missing_root} unfindable (missing root), {wrong} pipeline failures, of {N}")
print("expected found fraction with two roots ~ (2/e)^3 = 0.40")
print("PASS" if wrong == 0 and found > 0 else "FAIL")
