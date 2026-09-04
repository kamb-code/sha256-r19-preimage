#!/usr/bin/env python3
"""Run submask_triton.sweep under TRITON_INTERPRET=1 (CPU) against the on-disk
tables, and compare with gpu_submask.sweep_context and submask_family on
planted preimages at R=19 and R=20; also exercise the wrap at 2^32 and the
out_cap overflow path."""
import os, sys, time
os.environ["TRITON_INTERPRET"] = "1"
import numpy as np
import torch
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import M, K, IV, S0, S1, s0, Ch, Maj, T2, digest, recover_W, make_context
import gpu_submask as G
import submask_triton as T

TDIR = "/nvme0n1-disk/Kamvid"
NROOTS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
tbls = G.load_tables_cpu([f"{TDIR}/sigma0_u_table.npy", f"{TDIR}/sigma0_u_table_root2.npy"][:NROOTS])
T.rebuild.tbl = tbls
dev = torch.device("cpu")
G.CHUNK = 1 << 13


def in_tables(u):
    v = (s0(u) - u) & M
    return any(int(t[v]) & M == u for t in tbls)


def ref_candidates(cc, R, lo, n):
    """Reference (gpu_submask path) candidate a0 set + per-candidate verification."""
    st, _, hits = G.sweep_context(tbls, cc, None if False else h_global[0], R, dev, n, 8, False,
                                  lambda m: None, a0_start=lo)
    return st, hits


h_global = [None]
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 7)
bad = 0
for R in (19, 20):
    print(f"=== R={R}, {NROOTS} root table(s) ===")
    for i in range(6):
        W, a_true = T._plant(rng, R)
        h = digest(W, R); h_global[0] = h
        ctx = {r: a_true[r] for r in range(4, 12)}
        cc = G.context_constants(h, ctx, R)
        a0 = a_true[0]
        ok_roots = all(in_tables(W[r]) for r in (1, 2, 3))
        n = 4096
        lo = (a0 - 1500) & M
        t0 = time.time()
        hits, nsub = T.run_context(tbls, cc, R, n, a0_start=lo, block=256)
        el = time.time() - t0
        st, _, ref_hits = G.sweep_context(tbls, cc, h, R, dev, n, 8, False, lambda m: None, a0_start=lo)
        ver = sum(len(T.rebuild(cc, h, R, int(x))) for x in hits)
        got = a0 in {int(x) for x in hits}
        ok = (ver == st['ver']) and (nsub == st['sub']) and (got == ok_roots) and (ver == len(ref_hits))
        # every kernel-reported a0 must rebuild to >= 1 verified preimage
        for x in hits:
            if len(T.rebuild(cc, h, R, int(x))) == 0:
                ok = False; print("   kernel a0 did not rebuild:", hex(int(x)))
        bad += (not ok)
        print(f"  plant {i}: a0=0x{a0:08x} roots_stored={ok_roots} kernel hits={len(hits)} sub={nsub} "
              f"ver={ver} | ref sub={st['sub']} ver={st['ver']} | found={got} [{el:.1f}s] {'OK' if ok else 'MISMATCH'}")

# --- wrap around 2^32 and out_cap overflow, R=19 (hits plentiful) ---------
print("=== wrap at 2^32 + out_cap overflow, R=19 ===")
rng2 = np.random.default_rng(11)
msg = bytes(rng2.integers(0, 256, 55, dtype=np.uint8).tolist())
import struct
pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
h = digest([struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)], 19)
cc = G.context_constants(h, make_context(rng2, 19), 19)
n = 1 << 17
lo = (-(n // 2)) & M          # window straddles 0xFFFFFFFF -> 0
t0 = time.time()
hits, nsub = T.run_context(tbls, cc, 19, n, a0_start=lo, block=1024, chunk=1 << 15)
el = time.time() - t0
st, _, ref_hits = G.sweep_context(tbls, cc, h, 19, dev, n, 8, False, lambda m: None, a0_start=lo)
ref_a0 = sorted({int(struct_a0) for struct_a0 in []})
# reference a0 set: recompute from ref hits (W -> a0 via forward)
from submask_family import forward
ref_a0 = sorted({forward(Wm, 19)[0][0] for Wm in ref_hits})
ker_a0 = sorted({int(x) for x in hits})
print(f"  kernel: {len(hits)} hits, sub={nsub}, a0 set size {len(ker_a0)} [{el:.1f}s]")
print(f"  ref   : ver={st['ver']}, sub={st['sub']}, a0 set size {len(ref_a0)}")
print(f"  sets equal: {ker_a0 == ref_a0}; sub equal: {nsub == st['sub']}")
print(f"  hits below/above wrap: {sum(1 for x in ker_a0 if x < n)} / {sum(1 for x in ker_a0 if x >= lo)}")
bad += (ker_a0 != ref_a0) or (nsub != st['sub'])
# out_cap overflow: cap=3
cap = 3
hits_c, nsub_c = T.run_context(tbls, cc, 19, n, a0_start=lo, block=1024, chunk=1 << 15, out_cap=cap)
print(f"  out_cap={cap}: returned {len(hits_c)} (expect {min(cap, len(ker_a0))}), all in ref set: "
      f"{set(int(x) for x in hits_c) <= set(ref_a0)}, sub={nsub_c}")
bad += not (len(hits_c) == min(cap, len(ker_a0)) and set(int(x) for x in hits_c) <= set(ref_a0))
print("\nRESULT:", "ALL OK" if bad == 0 else f"{bad} MISMATCHES")
