import os, sys, time, struct
os.environ["TRITON_INTERPRET"] = "1"
import numpy as np, torch
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import M, digest, make_context, forward
import gpu_submask as G, submask_triton as T
tbls = G.load_tables_cpu(["/nvme0n1-disk/Kamvid/sigma0_u_table.npy", "/nvme0n1-disk/Kamvid/sigma0_u_table_root2.npy"])
T.rebuild.tbl = tbls; dev = torch.device("cpu"); G.CHUNK = 1 << 16
rng = np.random.default_rng(20260904)
msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
pad = msg + b"\x80" + b"\x00" * 0 + struct.pack(">Q", 55 * 8)
h = digest([struct.unpack(">I", pad[4*i:4*i+4])[0] for i in range(16)], 19)
cc = G.context_constants(h, make_context(rng, 19), 19)
n = 1 << 20; lo = int(rng.integers(0, 1 << 32))
t0 = time.time(); hits, nsub = T.run_context(tbls, cc, 19, n, a0_start=lo, block=1024, chunk=1 << 16, out_cap=1 << 16); el = time.time() - t0
st, _, ref_hits = G.sweep_context(tbls, cc, h, 19, dev, n, 8, False, lambda m: None, a0_start=lo)
ker = sorted(int(x) for x in hits); ref = sorted(forward(Wm, 19)[0][0] for Wm in ref_hits)
print(f"window 2^20 from 0x{lo:08x}: kernel {len(ker)} hits (unique {len(set(ker))}) sub={nsub} [{el:.0f}s]; ref ver={st['ver']} sub={st['sub']} sol={st['sol']}")
print("multiset equal:", ker == ref, "| sub equal:", nsub == st['sub'])
# every kernel a0 rebuilds to exactly the reference preimages
reb = sorted(tuple(W) for a0 in set(ker) for W in T.rebuild(cc, h, 19, a0)); refW = sorted(tuple(W) for W in ref_hits)
print("rebuilt W multiset == reference W multiset:", reb == refW, len(reb), len(refW))
