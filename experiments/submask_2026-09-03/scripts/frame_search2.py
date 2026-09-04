#!/usr/bin/env python3
"""Can the family be pushed past 20 rounds?

At R rounds the backward chain fixes a_{R-8}..a_{R-1}, the context is
a_4..a_{R-9}, and the schedule gives constraints C_0..C_{R-17}.  In the submask
family C0,C1,C2 solve triangularly for a1,a2,a3 and the W9 consistency residual
collapses; every FURTHER constraint (C3 at R=20, C3 and C4 at R=21, ...) is an
extra exact 32-bit check.  The only way to avoid paying 2^32 per extra round is
to let the extra constraint ABSORB one more unknown, i.e. promote a4 (and a5,
...) from context to unknown and solve C3 for it.

That collides with the family, whose conditions are statements ABOUT a4 and a5.
This script settles it by measuring, for each candidate set of context
conditions, which constraint targets still depend on which unknowns.  An
acyclic dependency matrix (target of C_j free of a_k for every k > j+1) means a
triangular solve exists and the round is cheap; a cyclic one means it is not.

No table needed; runs in seconds.
"""
import itertools, struct, sys
import numpy as np

sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, digest,
                            recover_W, backward_chain, forward)

R = 20


def build_state(ctx, unk, R):
    """Full a/e maps from context + unknowns + a planted backward chain."""
    a = dict(ctx); a.update(unk)
    a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(0, R):
        e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M
    return a, e


def targets(a, e, R):
    """The four constraint targets, each written so the absorbed unknown cancels.

    C_j : sigma0(W_{1+j}) - W_{1+j} = W_{16+j} - s1(W_{14+j}) - W_{9+j} - W_j
          - (the part of W_{1+j} free of a_{j+1})
    We only need to know WHICH unknowns the right-hand side moves with, so we
    return the right-hand sides.
    """
    W = {r: recover_W(a, e, r) for r in range(0, R)}
    out = []
    for j in range(R - 16):
        rhs = (W[16 + j] - s1(W[14 + j]) - W[9 + j] - W[j]) & M
        out.append(rhs)
    return out


def dependency_matrix(cond, unknowns, trials=40, seed=0):
    """For each constraint j and unknown a_k, does the target of C_j move with a_k?

    `cond` builds a context dict given a random generator; it may itself depend
    on the unknowns only through values it is allowed to see (none).
    """
    rng = np.random.default_rng(seed)
    n = R - 16
    dep = np.zeros((n, len(unknowns)), dtype=bool)
    for _ in range(trials):
        base = {i: int(rng.integers(0, 1 << 32, dtype=np.uint64)) for i in range(R - 8)}
        ctx = cond(base, rng)
        unk = {k: ctx.pop(k) if k in ctx else int(rng.integers(0, 1 << 32, dtype=np.uint64))
               for k in unknowns}
        # planted backward chain: give a_{R-8}..a_{R-1} random values
        bc = {i: int(rng.integers(0, 1 << 32, dtype=np.uint64)) for i in range(R - 8, R)}
        full_ctx = {**ctx, **bc}
        a, e = build_state(full_ctx, unk, R)
        t0 = targets(a, e, R)
        for ki, k in enumerate(unknowns):
            u2 = dict(unk)
            u2[k] = (u2[k] + int(rng.integers(1, 1 << 32, dtype=np.uint64))) & M
            ctx2 = cond({**base, **{k: u2[k]}}, rng) if k in base else full_ctx
            a2, e2 = build_state(full_ctx, u2, R)
            t1 = targets(a2, e2, R)
            for j in range(n):
                if t0[j] != t1[j]:
                    dep[j, ki] = True
    return dep


def show(name, dep, unknowns):
    n = dep.shape[0]
    print(f"\n{name}")
    print("        " + "".join(f"  a{k:<3}" for k in unknowns))
    for j in range(n):
        row = "".join(("   X  " if dep[j, ki] else "   .  ") for ki in range(len(unknowns)))
        print(f"  C{j}    {row}")
    # triangular if the target of C_j is free of every a_k absorbed later
    ok = all(not dep[j, ki] for j in range(n) for ki, k in enumerate(unknowns) if ki > j)
    print(f"  -> {'TRIANGULAR (cheap)' if ok else 'cyclic (needs a guess or an iteration)'}")
    return ok


# ---- candidate context-condition sets -------------------------------------
def c_random(base, rng):
    return dict(base)


def c_family(base, rng):
    """a4 = a5 = v, e8 = e9 = 0xFFFFFFFF."""
    c = dict(base)
    c[5] = c[4]
    c[8] = (M - c[4] + S0(c[7]) + Maj(c[7], c[6], c[5])) & M
    c[9] = (M - c[5] + S0(c[8]) + Maj(c[8], c[7], c[6])) & M
    return c


def c_family_a6(base, rng):
    """the family, plus a6 = a5 (kills a4 out of T1_7)."""
    c = c_family(base, rng)
    c[6] = c[5]
    c[8] = (M - c[4] + S0(c[7]) + Maj(c[7], c[6], c[5])) & M
    c[9] = (M - c[5] + S0(c[8]) + Maj(c[8], c[7], c[6])) & M
    return c


def c_shift(base, rng):
    """the family shifted one round later: a5 = a6, e9 = e10 = 0xFFFFFFFF.

    This is the shape that would be needed if a4 became an unknown.
    """
    c = dict(base)
    c[6] = c[5]
    c[9] = (M - c[5] + S0(c[8]) + Maj(c[8], c[7], c[6])) & M
    c[10] = (M - c[6] + S0(c[9]) + Maj(c[9], c[8], c[7])) & M
    return c



def c_triple(base, rng):
    """a4 = a5 = a6 = v, e8 = e9 = e10 = 0xFFFFFFFF (a8, a9, a10 all determined)."""
    c = dict(base)
    c[5] = c[4]; c[6] = c[4]
    c[8] = (M - c[4] + S0(c[7]) + Maj(c[7], c[6], c[5])) & M
    c[9] = (M - c[5] + S0(c[8]) + Maj(c[8], c[7], c[6])) & M
    c[10] = (M - c[6] + S0(c[9]) + Maj(c[9], c[8], c[7])) & M
    return c


def c_quad(base, rng):
    """a4 = a5 = a6 = a7 = v, e8..e11 = 0xFFFFFFFF."""
    c = dict(base)
    for i in (5, 6, 7):
        c[i] = c[4]
    c[8] = (M - c[4] + S0(c[7]) + Maj(c[7], c[6], c[5])) & M
    c[9] = (M - c[5] + S0(c[8]) + Maj(c[8], c[7], c[6])) & M
    c[10] = (M - c[6] + S0(c[9]) + Maj(c[9], c[8], c[7])) & M
    c[11] = (M - c[7] + S0(c[10]) + Maj(c[10], c[9], c[8])) & M
    return c


CONDS = [("random context", c_random), ("family a4=a5, e8=e9=-1", c_family),
         ("family + a6=a5", c_family_a6), ("shifted a5=a6, e9=e10=-1", c_shift),
         ("TRIPLE a4=a5=a6, e8=e9=e10=-1", c_triple),
         ("QUAD a4..a7 equal, e8..e11=-1", c_quad)]

print("Which ABOVE-DIAGONAL dependencies can context choice clear?")
print("C_j absorbs a_{j+1} (the diagonal, always present and wanted).")
print("C0's row is special: its dependence on a2,a3,.. is frozen provisionally")
print("and paid for by the consistency residual, which the family collapses.")
print("For an extra round to be free, C3 must absorb a4, which needs")
print("C1 free of BOTH a3 and a4, and C2 free of a4.")
print()
hdr = f"{'context conditions':<34}{'C1 free of a3':>15}{'C1 free of a4':>15}{'C2 free of a4':>15}   verdict"
print(hdr); print("-" * len(hdr))
for name, cond in CONDS:
    dep = dependency_matrix(cond, [1, 2, 3, 4], trials=40)
    c1a3 = not dep[1, 2]; c1a4 = not dep[1, 3]; c2a4 = not dep[2, 3]
    v = "C3 CAN absorb a4" if (c1a3 and c1a4 and c2a4) else (
        "iteration removed only" if c1a3 else "nothing cleared")
    print(f"{name:<34}{str(c1a3):>15}{str(c1a4):>15}{str(c2a4):>15}   {v}")
print()
print("Also: does the consistency residual still collapse under each?")
import numpy as np
rng = np.random.default_rng(5)
for name, cond in CONDS:
    hits = 0; tot = 0
    for _ in range(30):
        base = {i: int(rng.integers(0, 1 << 32, dtype=np.uint64)) for i in range(12)}
        c = cond(base, rng)
        a4, a5, a6, a7, a8, a9 = (c[i] for i in range(4, 10))
        e8 = (a4 + a8 - T2(a7, a6, a5)) & M
        T1_7 = (a7 - T2(a6, a5, a4)) & M
        n = 1 << 18
        A2 = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
        A3 = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
        def W9c(x2, x3):
            T15 = (np.uint32(a5) - np.uint32(S0(a4)) - Maj(np.uint32(a4), x3, x2))
            e7 = (x3 + np.uint32(T1_7))
            e6 = (x2 + np.uint32((a6 - S0(a5)) & M) - Maj(np.uint32(a5), np.uint32(a4), x3))
            return (-T15 - np.uint32(S1(e8)) - Ch(np.uint32(e8), e7, e6))
        eps = (W9c(A2, A3) - W9c(np.uint32(0), np.uint32(0)))
        hits += int((eps == np.uint32(0)).sum()); tot += n
    print(f"  {name:<34} P(eps==0) = {hits/tot:.3e}   "
          f"{'COLLAPSED' if hits/tot > 1e-5 else 'full 32-bit'}")

