"""Third-table branch check under TRITON_INTERPRET: tables [T1,T2,T1] vs [T1,T2].
For each a0 hit of the 2-table kernel, enumerate the passing root chains and
predict the multiplicity the 3-table kernel must record: 2^(levels using table 0).
Compare multisets exactly.  R=19 (no C3) so hits are plentiful."""
import os, sys, time, struct, collections
os.environ["TRITON_INTERPRET"] = "1"
import numpy as np, torch
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import M, K, S0, S1, Ch, Maj, digest, make_context
import gpu_submask as G
import submask_triton as T
TDIR = "/nvme0n1-disk/Kamvid"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 14
tbls2 = G.load_tables_cpu([f"{TDIR}/sigma0_u_table.npy", f"{TDIR}/sigma0_u_table_root2.npy"])
tbls3 = [tbls2[0], tbls2[1], tbls2[0]]
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
h = digest([struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)], 19)
cc = G.context_constants(h, make_context(rng, 19), 19)
lo = int(rng.integers(0, 1 << 32, dtype=np.uint64))
t0 = time.time()
h2, s2 = T.run_context(tbls2, cc, 19, n, a0_start=lo, block=256, chunk=1 << 13, out_cap=1 << 16)
t1 = time.time()
h3, s3 = T.run_context(tbls3, cc, 19, n, a0_start=lo, block=256, chunk=1 << 13, out_cap=1 << 16)
t2 = time.time()
print(f"2 tables: {len(h2)} hits sub={s2} [{t1-t0:.0f}s]; 3 tables [T1,T2,T1]: {len(h3)} hits sub={s3} [{t2-t1:.0f}s]")
c2 = collections.Counter(int(x) for x in h2); c3 = collections.Counter(int(x) for x in h3)
# predict multiplicities from an independent host enumeration over the 2 tables
def chains(a0):
    c = {k: T.u32(v) for k, v in cc.items() if k not in ('a', 'e')}
    E0 = (a0 + c['Ce0']) & M; W0 = (a0 + c['C0c']) & M
    Gv = (-(S0(a0) + Maj(a0, c['am1'], c['am2'])) - c['em3'] - S1(E0) - Ch(E0, c['em1'], c['em2']) - K[1]) & M
    out = []
    def roots(v):
        return [(i, int(t[v]) & M) for i, t in enumerate(tbls2) if int(t[v]) & M != M]
    for i, W1 in roots((c['KC0'] - W0 - Gv) & M):
        A1 = (W1 - Gv) & M
        E1 = (c['am3'] + A1 - (S0(a0) + Maj(a0, c['am1'], c['am2']))) & M
        F12 = (-(S0(A1) + Maj(A1, a0, c['am1'])) - c['em2'] - S1(E1) - Ch(E1, E0, c['em1']) - K[2]) & M
        for j, W2 in roots((c['KC1'] - W1 - F12) & M):
            A2 = (W2 - F12) & M
            e2 = (c['am2'] + A2 - (S0(A1) + Maj(A1, a0, c['am1']))) & M
            F23 = (-(S0(A2) + Maj(A2, A1, a0)) - c['em1'] - S1(e2) - Ch(e2, E1, E0) - K[3]) & M
            for k, W3 in roots((c['KC2'] - W2 - F23) & M):
                A3 = (W3 - F23) & M
                if Maj(c['a4'], A3, A2) == A3:
                    out.append((i, j, k))
    return out
pred2 = {}; pred3 = {}
for a0 in set(c2) | set(c3):
    ch = chains(a0)
    pred2[a0] = len(ch)
    pred3[a0] = sum(2 ** sum(1 for t in tup if t == 0) for tup in ch)
pred2 = {k: v for k, v in pred2.items() if v}; pred3 = {k: v for k, v in pred3.items() if v}
ok2 = dict(c2) == pred2; ok3 = dict(c3) == pred3
print(f"2-table multiset == host enumeration: {ok2} ({sum(pred2.values())} chains over {len(pred2)} a0)")
print(f"3-table multiset == predicted 2^(#T1 levels): {ok3} ({sum(pred3.values())} chains over {len(pred3)} a0)")
print("sub relation: s3 == predicted", s3 == sum(pred3.values()), "; s2 == predicted", s2 == sum(pred2.values()))
hist = collections.Counter(pred3.values()); print("multiplicity histogram (3-table):", sorted(hist.items()))
if not ok3:
    for a0 in sorted(set(c3) | set(pred3)):
        if c3.get(a0, 0) != pred3.get(a0, 0): print("  mismatch a0=0x%08x kernel=%d pred=%d" % (a0, c3.get(a0, 0), pred3.get(a0, 0)))
print("RESULT:", "ALL OK" if ok2 and ok3 else "MISMATCH")
