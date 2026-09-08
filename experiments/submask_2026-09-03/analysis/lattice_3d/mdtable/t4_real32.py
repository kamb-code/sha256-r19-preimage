"""Full-width checks.

(1) The width-w model at w=32 reproduces the published attack's per-context
    constants (KC0,KC1,KC2,v,kappa3) exactly.
(2) With the real 16 GB table: the list L of solvable kappa3 values -- the set
    a single sweep would hand to every other context in the same fibre.
"""
import sys, os, numpy as np
sys.path.insert(0, '/home/administrator/sha/publish/code')
sys.path.insert(0, '/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/lattice/mdtable')
import submask_family as sf
from wmodel import W

M = 0xFFFFFFFF
w = W(32)


def published_signature(h, ctx, R=20):
    """The constants attack_context() actually uses, lifted verbatim."""
    ab, eb = sf.backward_chain(h, R)
    a = dict(ab); a.update(ctx)
    a.update({-1: sf.IV[0], -2: sf.IV[1], -3: sf.IV[2], -4: sf.IV[3]})
    e = {-1: sf.IV[4], -2: sf.IV[5], -3: sf.IV[6], -4: sf.IV[7]}
    for r in range(8, R):
        e[r] = (a[r - 4] + a[r] - sf.T2(a[r - 1], a[r - 2], a[r - 3])) & M
    a4, a5, a6, a7, a8, a9, a10 = (a[i] for i in range(4, 11))
    e8, e9, e10 = e[8], e[9], e[10]
    T1_7 = (a7 - sf.T2(a6, a5, a4)) & M
    c6 = (a6 - sf.S0(a5)) & M
    W9base = ((a9 - sf.T2(a8, a7, a6)) - sf.K[9]) & M
    W10base = ((a10 - sf.T2(a9, a8, a7)) - sf.S1(e9) - sf.K[10]) & M
    W11base = ((a[11] - sf.T2(a10, a9, a8)) - sf.S1(e10) - sf.K[11]) & M
    Wr = {r: sf.recover_W(a, e, r) for r in range(12, R)}
    K0p = (Wr[16] - sf.s1(Wr[14])) & M
    K1p = (Wr[17] - sf.s1(Wr[15])) & M
    K2p = (Wr[18] - sf.s1(Wr[16])) & M
    K3p = (Wr[19] - sf.s1(Wr[17]) - Wr[12]) & M
    W9hat = (W9base - (a5 - sf.S0(a4) - sf.Maj(a4, 0, 0)) - sf.S1(e8)
             - sf.Ch(e8, T1_7, (c6 - sf.Maj(a5, a4, 0)) & M)) & M
    D = ((a6 - sf.S0(a5) - sf.Maj(a5, a4, 0)) + sf.Ch(e9, e8, T1_7)) & M
    return ((K0p - W9hat) & M, (K1p - W10base + D) & M,
            (K2p - W11base + T1_7 + sf.Ch(e10, e9, e8)) & M, a4, K3p)


rng = np.random.default_rng(11)
bad = 0
for _ in range(200):
    msg = [int(x) for x in rng.integers(0, 1 << 32, 16, dtype=np.uint64)]
    h = sf.digest(msg, 20)
    ctx = sf.make_context(rng, 20)
    hh = tuple(int.from_bytes(h[4 * i:4 * i + 4], 'big') for i in range(8))
    if published_signature(h, ctx) != w.signature(hh, ctx, 20):
        bad += 1
print(f"(1) w=32 model signature == published attack constants: {200-bad}/200 "
      f"{'OK' if bad == 0 else 'FAIL'}")

if len(sys.argv) > 1 and sys.argv[1] == '--table':
    path = "/nvme0n1-disk/Kamvid/sigma0_u_table.npy"
    tbl = np.load(path, mmap_mode="r").view(np.uint32)
    print(f"(2) real table loaded: {tbl.size:,} entries")
    n = 1 << int(sys.argv[2]) if len(sys.argv) > 2 else 1 << 24
    msg = [int(x) for x in rng.integers(0, 1 << 32, 16, dtype=np.uint64)]
    h = sf.digest(msg, 20)
    ctx = sf.make_context(rng, 20)
    st, out = sf.attack_context(tbl, h, ctx, 20, n, rng)
    print(f"    swept {st['a0']:,}  C0 surv {st['surv']:,} ({st['surv']/st['a0']:.4f})"
          f"  triangular {st['sol']:,} ({st['sol']/max(st['surv'],1):.4f} per surv)"
          f"  collapse {st['eps0']:,} ({st['eps0']/max(st['sol'],1):.3e})")
    print(f"    solvable-kappa3 list size per 2^32 swept a0 (extrapolated): "
          f"2^{np.log2(max(st['eps0'],1) * (2**32 / st['a0'])):.2f}")
