#!/usr/bin/env python3
"""Fused Triton kernel for the submask-family sweep.

The PyTorch version of the sweep spends almost all of its memory bandwidth on
the wrong thing.  Every rotate, xor and add is a separate kernel that reads its
inputs from HBM and writes its output back, so roughly a hundred round trips per
swept a0 stream on the order of 300 GB/s of intermediates, while the three table
lookups that genuinely have to touch memory account for under 40 GB/s.  The
values are also carried in int64 tensors although they are 32-bit, doubling
every one of those transfers.

This kernel does the whole chain per thread in uint32 registers:

    a0 -> C0 lookup -> a1 -> C1 lookup -> a2 -> C2 lookup -> a3
       -> collapsed consistency test Maj(v,a3,a2) == a3
       -> (R = 20) the fourth constraint C3

Only the table gathers reach memory, and masked lanes do not fetch.  Several
root tables are handled by nested loops whose bounds are compile-time
constants, so lanes never diverge on control flow, only on masks.  Hits are
rare, so the kernel records just the winning a0 and the host rebuilds and
re-verifies the full state.

    python3 submask_triton.py --verify     # against gpu_submask.py, planted preimages
    python3 submask_triton.py --bench      # throughput, both implementations
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import time

import numpy as np
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from submask_family import (M, K, IV, S0, S1, s0, Ch, Maj, T2, digest,  # noqa: E402
                            recover_W, backward_chain, make_context, forward)
import gpu_submask as G  # noqa: E402

MISS32 = tl.constexpr(-1)


@triton.jit
def _shr(x, n: tl.constexpr):
    """Logical right shift on int32 (Triton's >> is arithmetic)."""
    return (x >> n) & ((1 << (32 - n)) - 1)


@triton.jit
def _rotr(x, n: tl.constexpr):
    return _shr(x, n) | (x << (32 - n))


@triton.jit
def _S0(x):
    return _rotr(x, 2) ^ _rotr(x, 13) ^ _rotr(x, 22)


@triton.jit
def _S1(x):
    return _rotr(x, 6) ^ _rotr(x, 11) ^ _rotr(x, 25)


@triton.jit
def _s0(x):
    return _rotr(x, 7) ^ _rotr(x, 18) ^ _shr(x, 3)


@triton.jit
def _Ch(e, f, g):
    return (e & f) ^ ((~e) & g)


@triton.jit
def _Maj(a, b, c):
    return (a & b) ^ (a & c) ^ (b & c)


@triton.jit
def _idx(x):
    """int32 value -> non-negative int64 table index."""
    return x.to(tl.int64) & 0xFFFFFFFF


@triton.jit
def sweep(t0_ptr, t1_ptr, t2_ptr,
          a0_lo, n_a0,
          Ce0, C0c, KC0, KC1, KC2, K3p, a4v,
          am1, am2, am3, em1, em2, em3,
          k1, k2, k3, k4,
          out_ptr, cnt_ptr, out_cap, sub_ptr,
          NROOTS: tl.constexpr, R20: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_a0
    A0 = a0_lo + idx
    subtot = 0

    E0 = A0 + Ce0
    W0 = A0 + C0c
    S0a0 = _S0(A0)
    Mj0 = _Maj(A0, am1, am2)
    Gv = -(S0a0 + Mj0) - em3 - _S1(E0) - _Ch(E0, em1, em2) - k1
    T0 = KC0 - W0 - Gv

    for i in tl.static_range(NROOTS):
        p0 = t0_ptr if i == 0 else (t1_ptr if i == 1 else t2_ptr)
        W1 = tl.load(p0 + _idx(T0), mask=live, other=-1)
        m1 = live & (W1 != MISS32)
        A1 = W1 - Gv
        E1 = am3 + A1 - (S0a0 + Mj0)
        S0a1 = _S0(A1)
        Mj1 = _Maj(A1, A0, am1)
        F12 = -(S0a1 + Mj1) - em2 - _S1(E1) - _Ch(E1, E0, em1) - k2
        T1 = KC1 - W1 - F12

        for j in tl.static_range(NROOTS):
            p1 = t0_ptr if j == 0 else (t1_ptr if j == 1 else t2_ptr)
            W2 = tl.load(p1 + _idx(T1), mask=m1, other=-1)
            m2 = m1 & (W2 != MISS32)
            A2 = W2 - F12
            e2 = am2 + A2 - (S0a1 + Mj1)
            S0a2 = _S0(A2)
            Mj2 = _Maj(A2, A1, A0)
            F23 = -(S0a2 + Mj2) - em1 - _S1(e2) - _Ch(e2, E1, E0) - k3
            T2v = KC2 - W2 - F23

            for kk in tl.static_range(NROOTS):
                p2 = t0_ptr if kk == 0 else (t1_ptr if kk == 1 else t2_ptr)
                W3 = tl.load(p2 + _idx(T2v), mask=m2, other=-1)
                m3 = m2 & (W3 != MISS32)
                A3 = W3 - F23
                hit = m3 & (_Maj(a4v, A3, A2) == A3)
                subtot += tl.sum(hit.to(tl.int32), axis=0)
                if R20:
                    e3 = am1 + A3 - (S0a2 + Mj2)
                    W4 = a4v - (_S0(A3) + _Maj(A3, A2, A1)) - E0 - _S1(e3) \
                        - _Ch(e3, e2, E1) - k4
                    hit = hit & ((_s0(W4) + W3 - K3p) == 0)
                n = tl.sum(hit.to(tl.int32), axis=0)
                slot = tl.atomic_add(cnt_ptr, n)
                pos = tl.cumsum(hit.to(tl.int32), axis=0) - 1
                keep = hit & (slot + pos < out_cap)
                tl.store(out_ptr + slot + pos, A0, mask=keep)
    tl.atomic_add(sub_ptr, subtot)


def u32(x):
    return int(x) & M


def i32(x):
    """uint32 value -> the signed int32 with the same bit pattern."""
    v = int(x) & M
    return v - (1 << 32) if v >= (1 << 31) else v


def run_context(tbls, cc, R, n_a0, a0_start=0, block=1024, out_cap=1 << 14,
                chunk=1 << 27):
    """Sweep one context with the fused kernel.  Returns the hit a0 values."""
    dev = tbls[0].device
    c = {k: u32(v) for k, v in cc.items() if k not in ('a', 'e')}
    t = list(tbls) + [tbls[0]] * (3 - len(tbls))
    out = torch.zeros(out_cap, dtype=torch.int32, device=dev)
    cnt = torch.zeros(1, dtype=torch.int32, device=dev)
    sub = torch.zeros(1, dtype=torch.int64, device=dev)
    done = 0
    while done < n_a0:
        n = min(chunk, n_a0 - done)
        sweep[(triton.cdiv(n, block),)](
            t[0], t[1], t[2], i32((a0_start + done) & M), n,
            i32(c['Ce0']), i32(c['C0c']), i32(c['KC0']), i32(c['KC1']), i32(c['KC2']),
            i32(c.get('K3p', 0)), i32(c['a4']),
            i32(c['am1']), i32(c['am2']), i32(c['am3']),
            i32(c['em1']), i32(c['em2']), i32(c['em3']),
            i32(K[1]), i32(K[2]), i32(K[3]), i32(K[4]),
            out, cnt, out_cap, sub,
            NROOTS=len(tbls), R20=(R >= 20), BLOCK=block)
        done += n
    nh = min(int(cnt.item()), out_cap)
    return out[:nh].cpu().numpy().astype(np.uint32), int(sub.item())


def rebuild(cc, h, R, a0):
    """Host-side: from a hit a0, redo the chain, rebuild W and verify."""
    a, e = cc['a'], cc['e']
    c = {k: u32(v) for k, v in cc.items() if k not in ('a', 'e')}
    tbl = rebuild.tbl
    E0 = (a0 + c['Ce0']) & M
    W0 = (a0 + c['C0c']) & M
    Gv = (-(S0(a0) + Maj(a0, c['am1'], c['am2'])) - c['em3'] - S1(E0)
          - Ch(E0, c['em1'], c['em2']) - K[1]) & M
    out = []
    for W1 in roots_of(tbl, (c['KC0'] - W0 - Gv) & M):
        A1 = (W1 - Gv) & M
        E1 = (c['am3'] + A1 - (S0(a0) + Maj(a0, c['am1'], c['am2']))) & M
        F12 = (-(S0(A1) + Maj(A1, a0, c['am1'])) - c['em2'] - S1(E1)
               - Ch(E1, E0, c['em1']) - K[2]) & M
        for W2 in roots_of(tbl, (c['KC1'] - W1 - F12) & M):
            A2 = (W2 - F12) & M
            e2 = (c['am2'] + A2 - (S0(A1) + Maj(A1, a0, c['am1']))) & M
            F23 = (-(S0(A2) + Maj(A2, A1, a0)) - c['em1'] - S1(e2)
                   - Ch(e2, E1, E0) - K[3]) & M
            for W3 in roots_of(tbl, (c['KC2'] - W2 - F23) & M):
                A3 = (W3 - F23) & M
                if Maj(c['a4'], A3, A2) != A3:
                    continue
                aa = dict(a); aa.update({0: a0, 1: A1, 2: A2, 3: A3})
                ee = dict(e)
                for rr in range(R):
                    ee[rr] = (aa[rr - 4] + aa[rr] - T2(aa[rr - 1], aa[rr - 2],
                                                       aa[rr - 3])) & M
                Wm = [recover_W(aa, ee, rr) for rr in range(16)]
                if digest(Wm, R) == h:
                    out.append(Wm)
    return out


def roots_of(tbls, v):
    seen = []
    for t in tbls:
        u = int(t[v]) & M
        if u != M and u not in seen:
            seen.append(u)
    return seen


def _plant(rng, R):
    """A block whose state is in the family AND satisfies the collapsed test."""
    v = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    W = [int(x) for x in rng.integers(0, 1 << 32, size=16, dtype=np.uint64)]
    a = {-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]}
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(16):
        base = (e[r - 4] + S1(e[r - 1]) + Ch(e[r - 1], e[r - 2], e[r - 3]) + K[r]) & M
        T2r = T2(a[r - 1], a[r - 2], a[r - 3])
        if r == 3:
            msk = int(rng.integers(0, 1 << 32, dtype=np.uint64))
            a3 = ((a[2] & msk) | (v & ~msk)) & M
            W[r] = (a3 - T2r - base) & M
        elif r in (4, 5):
            W[r] = (v - T2r - base) & M
        elif r == 8:
            W[r] = ((M - a[4]) - base) & M
        elif r == 9:
            W[r] = ((M - a[5]) - base) & M
        T1 = (base + W[r]) & M
        a[r] = (T1 + T2r) & M
        e[r] = (a[r - 4] + T1) & M
    assert a[4] == v and a[5] == v and e[8] == M and e[9] == M
    assert Maj(v, a[3], a[2]) == a[3]
    return W, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--roots", type=int, default=3)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--plants", type=int, default=12)
    ap.add_argument("--a0", type=int, default=1 << 30)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--run", action="store_true", help="production sweep, stops on a verified hit")
    ap.add_argument("--hash", default=None)
    ap.add_argument("--hours", type=float, default=60.0)
    ap.add_argument("--contexts", type=int, default=10**9)
    ap.add_argument("--out", default=".")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    dev = torch.device("cuda")
    R = args.rounds
    print(f"device {torch.cuda.get_device_name(0)}, triton {triton.__version__}", flush=True)
    tbls = G.build_tables(dev, args.roots, lambda m: print("  " + m, flush=True))
    cpu_tbls = [t.cpu() for t in tbls]
    rebuild.tbl = cpu_tbls

    if args.verify:
        rng = np.random.default_rng(99)
        ok_all = True
        for i in range(args.plants):
            W, a_true = _plant(rng, R)
            h = digest(W, R)
            ctx = {r: a_true[r] for r in range(4, 12)}
            cc = G.context_constants(h, ctx, R)
            a0 = a_true[0]
            lo = (a0 - 4096) & M
            hits, _ = run_context(tbls, cc, R, 8192, a0_start=lo, block=256)
            st, _, ref = G.sweep_context(tbls, cc, h, R, dev, 8192, 8, False,
                                         lambda m: None, a0_start=lo)
            ver = 0
            for x in hits:
                ver += len(rebuild(cc, h, R, int(x)))
            same = (ver == st['ver'])
            got = a0 in {int(x) for x in hits}
            ok_all &= same
            print(f"  plant {i:>2}: kernel a0-hits {len(hits)}, verified {ver}; "
                  f"reference verified {st['ver']}; planted a0 found {got}  "
                  f"{'OK' if same else 'MISMATCH'}", flush=True)
        print("\n" + ("VERIFY PASS" if ok_all else "VERIFY FAIL"))

    if args.run:
        production(args, tbls, dev, R)
        return

    if args.bench:
        rng = np.random.default_rng(5)
        msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
        pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
        h = digest([struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)], R)
        cc = G.context_constants(h, make_context(rng, R), R)
        run_context(tbls, cc, R, 1 << 22, block=args.block)      # warm up / compile
        res = {}
        for name in ("triton", "reference"):
            torch.cuda.synchronize(); t0 = time.time()
            if name == "triton":
                run_context(tbls, cc, R, args.a0, block=args.block)
            else:
                G.sweep_context(tbls, cc, h, R, dev, args.a0, 8, False, lambda m: None)
            torch.cuda.synchronize(); el = time.time() - t0
            res[name] = args.a0 / el
            print(f"{name:<10} {args.a0:,} a0 in {el:6.2f}s = {res[name]:.3e} a0/s", flush=True)
        print(f"speedup {res['triton']/res['reference']:.2f}x")


def production(args, tbls, dev, R):
    import json
    logf = open(os.path.join(args.out, "run.log"), "a")

    def log(m):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "big")
    rng = np.random.default_rng(seed)
    if args.hash:
        h = bytes.fromhex(args.hash.strip()); assert len(h) == 32
    else:
        msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
        pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
        h = digest([struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)], R)
    log(f"TRITON R={R} roots={len(tbls)} seed={seed} target={h.hex()}")

    tot = dict(a0=0, sub=0, ver=0)
    t0 = time.time()
    found = []
    for ci in range(args.contexts):
        if (time.time() - t0) / 3600 > args.hours:
            log(f"wall-clock cap of {args.hours} h reached"); break
        cc = context_constants_cached(h, make_context(rng, R), R)
        tc = time.time()
        hits, nsub = run_context(tbls, cc, R, 1 << 32, block=args.block)
        tot['a0'] += 1 << 32; tot['sub'] += nsub
        for a0 in hits:
            for Wm in rebuild(cc, h, R, int(a0)):
                tot['ver'] += 1; found.append(Wm)
                log("*** VERIFIED %d-ROUND PREIMAGE  a0=0x%08x  W = %s"
                    % (R, int(a0), " ".join(f"{w:08x}" for w in Wm)))
        el = time.time() - t0
        exp = tot['sub'] * (2.0 ** -32) if R >= 20 else tot['sub']
        log(f"ctx {ci}: {1 << 32:,} a0 in {time.time()-tc:.1f}s | cumulative a0 {tot['a0']:.3e} "
            f"sub {tot['sub']:,} ver {tot['ver']} | {tot['a0']/el:.2e} a0/s | "
            f"expected preimages so far {exp:.3f}")
        json.dump(dict(R=R, roots=len(tbls), seed=seed, target=h.hex(), contexts=ci + 1,
                       elapsed_s=el, a0_per_s=tot['a0'] / el, expected_preimages=exp,
                       kernel="triton", **tot,
                       hits=[" ".join(f"{w:08x}" for w in W) for W in found]),
                  open(os.path.join(args.out, "status.json"), "w"))
        if found:
            with open(os.path.join(args.out, "HIT.txt"), "w") as f:
                for Wm in found:
                    f.write(f"rounds={R} hash={h.hex()} words={' '.join(f'{w:08x}' for w in Wm)}\n")
            log("stopping on verified hit"); break
    log("DONE")


_CC = {}


def context_constants_cached(h, ctx, R):
    return G.context_constants(h, ctx, R)


if __name__ == "__main__":
    main()
