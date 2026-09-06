#!/usr/bin/env python3
"""Part (c): symmetric contexts at R=21, measured with the frame_search2 method
(boolean dependency of the constraint targets on the unknowns) plus edge weights
(bits moved per flipped bit) and the collapse rate of the W9 consistency residual.

Context types (a4..a12 at R=21; the digest fixes a13..a20):
  random, family (a4=a5=v, e8=e9=-1), quad (a4..a7=v, e8..e11=-1),
  allequal (a4..a12=v), zero (v=0), ones (v=-1),
  rot_s (a_{4+i} = ROTR^{s i}(v), s=1,2,4,8,16),
  aesym (e_k = a_k for k=8..12, i.e. a_{k-4} = T2(a_{k-1},a_{k-2},a_{k-3})).
Unknown window: a0 swept, a1..a5 absorbed by C0..C4 (frame B style at R=21)
and also the family frame (a1..a3 absorbed; C3, C4 filters).
"""
import sys
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, rotr, U32)

np.seterr(over="ignore")
R = 21


def popcnt(x):
    x = x.astype(np.uint64); cnt = np.zeros_like(x)
    for _ in range(32):
        cnt += x & 1; x >>= 1
    return cnt


def c(v): return U32(v & M)
def rw(rng, n): return rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(U32)


def make_ctx(kind, rng, n):
    """vectorised context a4..a12 (arrays of length n)."""
    ctx = {i: rw(rng, n) for i in range(4, 13)}
    if kind == "random":
        return ctx
    v = rw(rng, n)
    if kind == "zero": v = np.zeros(n, U32)
    if kind == "ones": v = np.full(n, M, U32)
    if kind in ("family", "quad"):
        m = 5 if kind == "family" else 7
        for i in range(4, m + 1): ctx[i] = v
        for i in range(8, 8 + (m - 3)):     # e_{i} = -1 via a_i, i = m+1 .. 2m-2
            ctx[i] = (c(M) - ctx[i - 4] + S0(ctx[i - 1]) + Maj(ctx[i - 1], ctx[i - 2], ctx[i - 3]))
        return ctx
    if kind in ("allequal", "zero", "ones"):
        for i in range(4, 13): ctx[i] = v
        return ctx
    if kind.startswith("rot_"):
        s = int(kind[4:])
        for i in range(4, 13): ctx[i] = rotr(v, (s * (i - 4)) % 32) if (s * (i - 4)) % 32 else v
        return ctx
    if kind == "aesym":
        for k in (8, 7, 6, 5, 4):        # a_k = T2(a_{k+3}, a_{k+2}, a_{k+1})
            ctx[k] = T2(ctx[k + 3], ctx[k + 2], ctx[k + 1])
        return ctx
    raise ValueError(kind)


def trajectory(a):
    e = {-1: c(IV[4]), -2: c(IV[5]), -3: c(IV[6]), -4: c(IV[7])}
    for r in range(R):
        e[r] = a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])
    W = {r: (a[r] - T2(a[r - 1], a[r - 2], a[r - 3]) - e[r - 4] - S1(e[r - 1])
             - Ch(e[r - 1], e[r - 2], e[r - 3]) - c(K[r])) for r in range(R)}
    F = [(W[16 + j] - s1(W[14 + j]) - W[9 + j] - s0(W[1 + j]) - W[j]) for j in range(R - 16)]
    return e, W, F


def state(ctx, unk, bc):
    a = {-1: c(IV[0]), -2: c(IV[1]), -3: c(IV[2]), -4: c(IV[3])}
    a.update(ctx); a.update(unk); a.update(bc)
    return a


def measure(kind, rng, n=200):
    ctx = make_ctx(kind, rng, n)
    bc = {i: rw(rng, n) for i in range(13, 21)}
    unk = {i: rw(rng, n) for i in range(0, 4)}
    # (i) frame B: a1..a5 absorbed; here a4, a5 are unknowns -> context must not depend on them.
    # For kinds that tie a4/a5 to the rest, frame B is not legal; we report the a-frame with
    # a4..a12 context (unknowns a1..a3) AND the boolean/weight matrix treating a4,a5 as if
    # perturbed with the context re-realised (the frame_search2 convention: cond(base)).
    dep = np.zeros((5, 5)); boolean = np.zeros((5, 5), bool)
    for k in range(1, 6):
        base = dict(ctx); base_unk = dict(unk)
        a0 = state(base, base_unk, bc)
        _, _, F0 = trajectory(a0)
        flip = U32(1) << rng.integers(0, 32, n, dtype=np.uint64).astype(U32)
        if k <= 3:
            u2 = dict(unk); u2[k] = unk[k] ^ flip
            a1 = state(ctx, u2, bc)
        else:
            # perturb a_k as a context word and re-realise the context conditions:
            # rebuild the context from a perturbed 'base' draw
            ctx2 = dict(ctx)
            ctx2[k] = ctx[k] ^ flip
            # re-impose ties that are functions of a4/a5
            if kind in ("family", "quad"):
                m = 5 if kind == "family" else 7
                if k == 4:
                    for i in range(4, m + 1): ctx2[i] = ctx2[4]
                if k == 5 and kind == "family":
                    ctx2[4] = ctx2[5]
                if k == 5 and kind == "quad":
                    for i in range(4, m + 1): ctx2[i] = ctx2[5]
                for i in range(8, 8 + (m - 3)):
                    ctx2[i] = (c(M) - ctx2[i - 4] + S0(ctx2[i - 1]) + Maj(ctx2[i - 1], ctx2[i - 2], ctx2[i - 3]))
            elif kind in ("allequal", "zero", "ones"):
                for i in range(4, 13): ctx2[i] = ctx2[k]
            elif kind.startswith("rot_"):
                s = int(kind[4:]); v = rotr(ctx2[k], (32 - (s * (k - 4)) % 32) % 32) if (s * (k - 4)) % 32 else ctx2[k]
                for i in range(4, 13): ctx2[i] = rotr(v, (s * (i - 4)) % 32) if (s * (i - 4)) % 32 else v
            elif kind == "aesym":
                pass   # a4, a5 are determined by a6..a12; perturbing them breaks the tie (illegal), measured anyway
            a1 = state(ctx2, unk, bc)
        _, _, F1 = trajectory(a1)
        for j in range(5):
            d = popcnt(F0[j] ^ F1[j])
            dep[k - 1, j] = d.mean(); boolean[k - 1, j] = bool((d > 0).any())
    # (ii) collapse rate of the W9 residual, per the family formula (needs a4..a9, e8, e9)
    m_ = 1 << 16
    A2 = rw(rng, m_); A3 = rw(rng, m_)
    hits = 0
    for t in range(8):     # 8 contexts x 2^16 pairs
        a4, a5, a6, a7, a8, a9 = (int(ctx[i][t]) for i in range(4, 10))
        e8 = (a4 + a8 - T2(a7, a6, a5)) & M
        T1_7 = (a7 - T2(a6, a5, a4)) & M
        def W9c(x2, x3):
            T15 = (c(a5) - c(S0(a4)) - Maj(c(a4), x3, x2))
            e7 = x3 + c(T1_7)
            e6 = x2 + c((a6 - S0(a5)) & M) - Maj(c(a5), c(a4), x3)
            return (-T15 - c(S1(e8)) - Ch(c(e8), e7, e6))
        eps = W9c(A2, A3) - W9c(U32(0), U32(0))
        hits += int((eps == 0).sum())
    rate = hits / (8 * m_)
    return dep, boolean, rate, ctx


def main():
    rng = np.random.default_rng(20260906)
    kinds = ["random", "family", "quad", "allequal", "zero", "ones", "rot_1", "rot_2", "rot_4",
             "rot_8", "rot_16", "aesym"]
    print("R=21: bits of F_j moved per flipped bit of a_k (200 trials/cell); a1..a3 absorbed by C0..C2,")
    print("a4,a5 candidates for C3,C4.  Heavy feedback = a_k -> C_j with j < k-1 and > 4 bits.")
    print("Also: C1 free of a3 / a4 / a5 and C2 free of a4 / a5 and C3 free of a5 (booleans, the")
    print("frame_search2 test), and the collapse rate P(eps==0) of the W9 residual ((3/4)^32 = 1.0e-4).\n")
    hdr = f"{'context':<10}{'#heavy fb':>10}{'C1!a3':>7}{'C1!a4':>7}{'C1!a5':>7}{'C2!a4':>7}{'C2!a5':>7}{'C3!a5':>7}{'P(eps=0)':>10}   heavy feedback edges"
    print(hdr); print("-" * len(hdr))
    for kind in kinds:
        dep, bo, rate, ctx = measure(kind, rng)
        heavy = [(k, j, dep[k - 1, j]) for k in range(1, 6) for j in range(5) if j < k - 1 and dep[k - 1, j] > 4]
        f = lambda k, j: 'yes' if not bo[k - 1, j] else 'no'
        print(f"{kind:<10}{len(heavy):>10}{f(3,1):>7}{f(4,1):>7}{f(5,1):>7}{f(4,2):>7}{f(5,2):>7}{f(5,3):>7}{rate:>10.2e}   "
              + ", ".join(f"a{k}->C{j} {w:.1f}" for k, j, w in heavy))
    print("\nFull matrices (rows a1..a5, columns C0..C4):")
    rng = np.random.default_rng(7)
    for kind in ("random", "family", "quad", "allequal", "zero", "rot_1", "aesym"):
        dep, bo, rate, ctx = measure(kind, rng)
        print(f"  {kind}:")
        for k in range(5):
            print(f"    a{k+1}: " + " ".join(f"{dep[k, j]:6.1f}" for j in range(5)))
    # sanity: the symmetric contexts really are what they claim
    rng = np.random.default_rng(3)
    for kind in ("zero", "ones", "allequal", "aesym", "quad"):
        ctx = make_ctx(kind, rng, 4)
        bc = {i: rw(rng, 4) for i in range(13, 21)}
        a = state(ctx, {i: rw(rng, 4) for i in range(4)}, bc)
        e, W, F = trajectory(a)
        print(f"  check {kind}: e8..e12 =", [hex(int(e[i][0])) for i in range(8, 13)],
              " a4..a8 =", [hex(int(a[i][0])) for i in range(4, 9)])


if __name__ == "__main__":
    main()
