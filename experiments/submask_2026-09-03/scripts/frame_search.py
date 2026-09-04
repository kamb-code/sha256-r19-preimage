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


print("R = 20.  Which constraint targets move with which unknowns?")
print("X = the target of C_j changes when a_k changes; a triangular solve needs")
print("    the region above the diagonal to be clear.")

print("\n" + "=" * 72)
print("FRAME 1: unknowns a1,a2,a3 (a0 swept, a4..a11 context) -- the current attack")
for name, cond in (("random context", c_random), ("submask family", c_family)):
    show(name, dependency_matrix(cond, [1, 2, 3]), [1, 2, 3])

print("\n" + "=" * 72)
print("FRAME 2: unknowns a1,a2,a3,a4 (a0 swept, a5..a11 context)")
print("         -- C3 would absorb a4, removing the extra 2^32 at R=20")
for name, cond in (("random context", c_random), ("submask family", c_family),
                   ("family + a6=a5", c_family_a6), ("shifted family a5=a6, e9=e10=-1", c_shift)):
    show(name, dependency_matrix(cond, [1, 2, 3, 4]), [1, 2, 3, 4])

print("\n" + "=" * 72)
print("Note: the family's own conditions (a4 = a5, e8 = a4 + c8 = -1) are")
print("statements about a4, so they cannot be imposed when a4 is an unknown.")
print("c_family above therefore pins a4 even in frame 2, which is why it is")
print("listed only for comparison.")
