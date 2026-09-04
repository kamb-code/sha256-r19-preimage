#!/usr/bin/env python3
"""GPU sweep of the submask context family at R = 19 or R = 20.

Per context (a4 = a5 = v, e8 = e9 = 0xFFFFFFFF, a6 a7 a10 [a11] random) the
whole a0 range is swept once.  Each a0 goes through the triangular solve
C0 -> a1, C1 -> a2, C2 -> a3 with up to --roots roots per table lookup, then the
collapsed consistency test Maj(a4, a3, a2) == a3, and at R = 20 the exact fourth
constraint C3.  Every full hit is rebuilt into sixteen message words and
re-hashed on the CPU by an implementation independent of the solver before it
is reported.

The tables u -> sigma0(u) - u (first, second, third root) are built on the
device at start-up, about a few seconds each on an H100; they need 16 GB each.

    python3 gpu_submask.py --rounds 20 --roots 3 --stop-on-hit --hours 60
    python3 gpu_submask.py --rounds 19 --device cpu --smoke   # CPU self-check

Writes status.json after every context and HIT.txt on a verified preimage.
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from submask_family import (M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, digest,   # noqa: E402
                            recover_W, backward_chain, make_context, forward)

MASK = (1 << 32) - 1
SEN = -1                      # int32 sentinel, reads back as 0xFFFFFFFF
CHUNK = 1 << 24               # a0 per device batch


# ---- torch versions of the primitives (int64 tensors holding 32-bit values) --
def rotr_t(x, n):
    return ((x >> n) | (x << (32 - n))) & MASK


def S0_t(x): return rotr_t(x, 2) ^ rotr_t(x, 13) ^ rotr_t(x, 22)
def S1_t(x): return rotr_t(x, 6) ^ rotr_t(x, 11) ^ rotr_t(x, 25)
def s0_t(x): return rotr_t(x, 7) ^ rotr_t(x, 18) ^ (x >> 3)
def Ch_t(e, f, g): return ((e & f) ^ (~e & g)) & MASK
def Maj_t(a, b, c): return (a & b) ^ (a & c) ^ (b & c)


# ---- tables ----------------------------------------------------------------
def build_tables(dev, nroots, log):
    tbls = []
    t0 = time.time()
    step = 1 << 25 if dev.type == "cuda" else 1 << 22
    for r in range(nroots):
        t = torch.full((1 << 32,), SEN, dtype=torch.int32, device=dev)
        for s in range(0, 1 << 32, step):
            u = torch.arange(s, s + step, dtype=torch.int64, device=dev)
            v = (s0_t(u) - u) & MASK
            if r == 0:
                t[v] = u.to(torch.int32)
            else:
                m = t[v] == SEN
                for prev in tbls:
                    m &= (prev[v].to(torch.int64) & MASK) != u
                t[v[m]] = u[m].to(torch.int32)
            del u, v
        cov = (t[::1024] != SEN).double().mean().item()
        log(f"table root {r+1}: coverage {cov*100:.2f}%  [{time.time()-t0:.0f}s]")
        tbls.append(t)
    return tbls


def load_tables_cpu(paths):
    tbls = []
    for p in paths:
        arr = np.load(p, mmap_mode="r")
        tbls.append(torch.from_numpy(np.asarray(arr).view(np.int32)))
    return tbls


def lookup(tbls, tgt):
    """Gather every stored root of each target.  Returns (source index, root)."""
    idxs, vals = [], []
    for t in tbls:
        r = t[tgt].to(torch.int64) & MASK
        ok = r != MASK
        idxs.append(torch.nonzero(ok, as_tuple=True)[0])
        vals.append(r[ok])
    if len(tbls) == 1:
        return idxs[0], vals[0]
    return torch.cat(idxs), torch.cat(vals)


# ---- per-context constants, python ints, mirroring submask_family.attack_context
def context_constants(h, ctx, R):
    ab, eb = backward_chain(h, R)
    a = dict(ab); a.update(ctx); a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
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
    K3p = ((Wr[19] - s1(Wr[17]) - Wr[12]) & M) if R >= 20 else 0
    T2iv = T2(am1, am2, am3)
    C0c = (-T2iv - em4 - S1(em1) - Ch(em1, em2, em3) - K[0]) & M
    Ce0 = (am4 - T2iv) & M
    W9hat = (W9base - (a5 - S0(a4) - Maj(a4, 0, 0)) - S1(e8)
             - Ch(e8, T1_7, (c6 - Maj(a5, a4, 0)) & M)) & M
    D = ((a6 - S0(a5) - Maj(a5, a4, 0)) + Ch(e9, e8, T1_7)) & M
    return dict(a=a, e=e, a4=a4, am1=am1, am2=am2, am3=am3, am4=am4,
                em1=em1, em2=em2, em3=em3, em4=em4,
                KC0=(K0p - W9hat) & M, KC1=(K1p - W10base + D) & M,
                KC2=(K2p - W11base + T1_7 + Ch(e10, e9, e8)) & M,
                K3p=K3p, C0c=C0c, Ce0=Ce0)


def verify_hit(cc, h, R, a0, a1, a2, a3):
    aa = dict(cc['a']); aa.update({0: a0, 1: a1, 2: a2, 3: a3})
    ee = dict(cc['e'])
    for rr in range(R):
        ee[rr] = (aa[rr - 4] + aa[rr] - T2(aa[rr - 1], aa[rr - 2], aa[rr - 3])) & M
    Wm = [recover_W(aa, ee, rr) for rr in range(16)]
    return Wm, digest(Wm, R) == h


# ---- one context ------------------------------------------------------------
def sweep_context(tbls, cc, h, R, dev, n_a0, kmax, stop_on_hit, log, a0_start=0):
    c = {k: int(v) for k, v in cc.items() if k not in ('a', 'e')}
    st = dict(a0=0, sol=0, sub=0, c3=0, ver=0)
    kcounts = np.zeros(kmax + 1, dtype=np.int64)
    hits = []
    for s in range(a0_start, a0_start + n_a0, CHUNK):
        b = min(CHUNK, a0_start + n_a0 - s)
        A0 = torch.arange(s, s + b, dtype=torch.int64, device=dev) & MASK
        st['a0'] += b
        E0 = (A0 + c['Ce0']) & MASK
        W0 = (A0 + c['C0c']) & MASK
        G = (-(S0_t(A0) + Maj_t(A0, c['am1'], c['am2'])) - c['em3'] - S1_t(E0)
             - Ch_t(E0, c['em1'], c['em2']) - K[1]) & MASK
        i, W1 = lookup(tbls, (c['KC0'] - W0 - G) & MASK)
        A0, E0, G = A0[i], E0[i], G[i]
        A1 = (W1 - G) & MASK
        E1 = (c['am3'] + A1 - (S0_t(A0) + Maj_t(A0, c['am1'], c['am2']))) & MASK
        F12 = (-(S0_t(A1) + Maj_t(A1, A0, c['am1'])) - c['em2'] - S1_t(E1)
               - Ch_t(E1, E0, c['em1']) - K[2]) & MASK
        i, W2 = lookup(tbls, (c['KC1'] - W1 - F12) & MASK)
        A0, A1, E0, E1, F12 = A0[i], A1[i], E0[i], E1[i], F12[i]
        A2 = (W2 - F12) & MASK
        e2 = (c['am2'] + A2 - (S0_t(A1) + Maj_t(A1, A0, c['am1']))) & MASK
        F23 = (-(S0_t(A2) + Maj_t(A2, A1, A0)) - c['em1'] - S1_t(e2)
               - Ch_t(e2, E1, E0) - K[3]) & MASK
        i, W3 = lookup(tbls, (c['KC2'] - W2 - F23) & MASK)
        A0, A1, A2, E0, E1, e2, F23 = (x[i] for x in (A0, A1, A2, E0, E1, e2, F23))
        A3 = (W3 - F23) & MASK
        st['sol'] += A3.numel()
        hit = Maj_t(c['a4'], A3, A2) == A3
        idx = torch.nonzero(hit, as_tuple=True)[0]
        st['sub'] += idx.numel()
        if idx.numel() == 0:
            continue
        if R >= 20:
            A0h, A1h, A2h, A3h = A0[idx], A1[idx], A2[idx], A3[idx]
            E0h, E1h, e2h, W3h = E0[idx], E1[idx], e2[idx], W3[idx]
            e3 = (c['am1'] + A3h - (S0_t(A2h) + Maj_t(A2h, A1h, A0h))) & MASK
            W4 = (c['a4'] - (S0_t(A3h) + Maj_t(A3h, A2h, A1h)) - E0h - S1_t(e3)
                  - Ch_t(e3, e2h, E1h) - K[4]) & MASK
            c3 = (s0_t(W4) + W3h - c['K3p']) & MASK
            for k in range(kmax + 1):
                kcounts[k] += int(((c3 & ((1 << k) - 1)) == 0).sum().item())
            full = torch.nonzero(c3 == 0, as_tuple=True)[0]
            st['c3'] += full.numel()
            cand = [(int(A0h[j]), int(A1h[j]), int(A2h[j]), int(A3h[j])) for j in full]
        else:
            cand = [(int(A0[j]), int(A1[j]), int(A2[j]), int(A3[j])) for j in idx]
        for (a0, a1, a2, a3) in cand:
            Wm, ok = verify_hit(cc, h, R, a0, a1, a2, a3)
            if ok:
                st['ver'] += 1
                hits.append(Wm)
                log(f"*** VERIFIED {R}-ROUND PREIMAGE  a0=0x{a0:08x}  "
                    f"W = {' '.join(f'{w:08x}' for w in Wm)}")
                if stop_on_hit:
                    return st, kcounts, hits
            else:
                log(f"!!! candidate failed forward verification a0=0x{a0:08x} (should not happen)")
    return st, kcounts, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--roots", type=int, default=3)
    ap.add_argument("--hash", default=None, help="64-hex target; random message digest if omitted")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hours", type=float, default=60.0, help="wall-clock cap")
    ap.add_argument("--contexts", type=int, default=10**9)
    ap.add_argument("--a0", type=int, default=1 << 32, help="a0 per context")
    ap.add_argument("--stop-on-hit", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="CPU: on-disk tables, small sweep")
    ap.add_argument("--table-dir", default="/nvme0n1-disk/Kamvid")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--kmax", type=int, default=28)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    R = args.rounds
    dev = torch.device(args.device)
    logf = open(os.path.join(args.out, "run.log"), "a")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "big")
    rng = np.random.default_rng(seed)
    if args.hash:
        h = bytes.fromhex(args.hash.strip()); assert len(h) == 32
    else:
        msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
        pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
        h = digest([struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)], R)
    log(f"R={R} roots={args.roots} device={dev} seed={seed} target={h.hex()}")
    log(f"torch {torch.__version__}; " + (torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"))

    if args.smoke:
        paths = [os.path.join(args.table_dir, "sigma0_u_table.npy"),
                 os.path.join(args.table_dir, "sigma0_u_table_root2.npy")][:args.roots]
        tbls = load_tables_cpu(paths)
        log(f"loaded {len(tbls)} on-disk table(s)")
    else:
        tbls = build_tables(dev, args.roots, log)

    tot = dict(a0=0, sol=0, sub=0, c3=0, ver=0)
    kcounts = np.zeros(args.kmax + 1, dtype=np.int64)
    t_start = time.time()
    all_hits = []
    for ci in range(args.contexts):
        if (time.time() - t_start) / 3600 > args.hours:
            log(f"wall-clock cap of {args.hours} h reached"); break
        ctx = make_context(rng, R)
        cc = context_constants(h, ctx, R)
        t0 = time.time()
        st, kc, hits = sweep_context(tbls, cc, h, R, dev, args.a0, args.kmax,
                                     args.stop_on_hit, log)
        for k in tot: tot[k] += st[k]
        kcounts += kc
        all_hits += hits
        el = time.time() - t_start
        rate = tot['a0'] / el
        exp_hits = tot['sub'] * (2.0 ** -32) if R >= 20 else tot['sub']
        proj = (el / exp_hits / 3600) if exp_hits > 0 else float('inf')
        log(f"ctx {ci}: {st['a0']:,} a0 in {time.time()-t0:.1f}s | cumulative a0 {tot['a0']:.3e} "
            f"sol {tot['sol']:.3e} sub {tot['sub']:,} c3 {tot['c3']} ver {tot['ver']} | "
            f"{rate:.2e} a0/s | " +
            (f"expected preimages so far {exp_hits:.3f}, projected hours per preimage {proj:.1f}"
             if R >= 20 else f"{tot['a0']/max(tot['ver'],1):,.0f} a0 per preimage"))
        status = dict(R=R, roots=args.roots, seed=seed, target=h.hex(), contexts=ci + 1,
                      elapsed_s=el, a0_per_s=rate, **tot,
                      expected_preimages=exp_hits, projected_hours=proj,
                      c3_lowk_zero=kcounts.tolist(),
                      hits=[" ".join(f"{w:08x}" for w in Wm) for Wm in all_hits])
        with open(os.path.join(args.out, "status.json"), "w") as f:
            json.dump(status, f)
        if hits and args.stop_on_hit:
            with open(os.path.join(args.out, "HIT.txt"), "w") as f:
                for Wm in hits:
                    f.write(f"rounds={R} hash={h.hex()} words={' '.join(f'{w:08x}' for w in Wm)}\n")
            log("stopping on verified hit")
            break
    log("DONE")


if __name__ == "__main__":
    main()
