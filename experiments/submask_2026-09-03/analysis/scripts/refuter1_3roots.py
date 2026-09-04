import os, sys, time, struct
os.environ["TRITON_INTERPRET"] = "1"
import numpy as np, torch
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import M, s0, digest, make_context, forward
import gpu_submask as G, submask_triton as T
D="/nvme0n1-disk/Kamvid/"
tbls = G.load_tables_cpu([D+"sigma0_u_table.npy", D+"sigma0_u_table_root2.npy", D+"sigma0_u_table_root3_refuter1.npy"])
T.rebuild.tbl = tbls; dev = torch.device("cpu"); G.CHUNK = 1 << 14
def in_tables(u):
    v = (s0(u) - u) & M
    return any(int(t[v]) & M == u for t in tbls)
rng = np.random.default_rng(31337); bad = 0; found = need = 0
for R in (19, 20):
    for i in range(8):
        W, a = T._plant(rng, R); h = digest(W, R)
        cc = G.context_constants(h, {r: a[r] for r in range(4, 12)}, R); a0 = a[0]
        ok_roots = all(in_tables(W[r]) for r in (1, 2, 3)); lo = (a0 - 700) & M; n = 2048
        hits, nsub = T.run_context(tbls, cc, R, n, a0_start=lo, block=256)
        st, _, ref = G.sweep_context(tbls, cc, h, R, dev, n, 8, False, lambda m: None, a0_start=lo)
        ver = sum(len(T.rebuild(cc, h, R, int(x))) for x in hits); got = a0 in {int(x) for x in hits}
        ok = ver == st['ver'] == len(ref) and nsub == st['sub'] and got == ok_roots
        bad += not ok; found += got; need += 1
        print(f"R={R} plant {i}: roots_stored={ok_roots} kernel hits={len(hits)} sub={nsub} ver={ver} | ref sub={st['sub']} ver={st['ver']} | found={got} {'OK' if ok else 'MISMATCH'}", flush=True)
print(f"planted found {found}/{need} (3-root expectation 0.93)")
# set-level window at R=19, all 27 paths
msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist()); pad = msg + b"\x80" + struct.pack(">Q", 440)
h = digest([struct.unpack(">I", pad[4*i:4*i+4])[0] for i in range(16)], 19)
cc = G.context_constants(h, make_context(rng, 19), 19); n = 1 << 19; lo = int(rng.integers(0, 1 << 32))
t0 = time.time(); hits, nsub = T.run_context(tbls, cc, 19, n, a0_start=lo, block=1024, chunk=1 << 16, out_cap=1 << 16); el = time.time() - t0
st, _, ref_hits = G.sweep_context(tbls, cc, h, 19, dev, n, 8, False, lambda m: None, a0_start=lo)
ker = sorted(int(x) for x in hits); ref = sorted(forward(Wm, 19)[0][0] for Wm in ref_hits)
reb = sorted(tuple(W) for a0 in set(ker) for W in T.rebuild(cc, h, 19, a0)); refW = sorted(tuple(W) for W in ref_hits)
print(f"window 2^19 (3 tables): kernel {len(ker)} hits sub={nsub} [{el:.0f}s]; ref ver={st['ver']} sub={st['sub']} sol={st['sol']} sol/a0={st['sol']/n:.4f}")
print("a0 multiset equal:", ker == ref, "| sub equal:", nsub == st['sub'], "| rebuilt W == ref W:", reb == refW)
bad += not (ker == ref and nsub == st['sub'] and reb == refW)
print("RESULT:", "ALL OK" if bad == 0 else f"{bad} MISMATCHES")
