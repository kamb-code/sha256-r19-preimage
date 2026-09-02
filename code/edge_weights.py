#!/usr/bin/env python3
"""Linear entries and feedback-edge weights of the schedule constraints.

Reproduces the heavy-cycle section of the companion paper (the 20-round
barrier).  Three measurements, all CPU, no table, about three minutes:

  1. LINEAR ENTRIES.  Each free state word a_k enters exactly two recovered
     message words linearly: W_k with coefficient +1 and W_{k+8} with
     coefficient -1.  It enters W_{k+1..k+7} nonlinearly and nothing else.
     Consequence: constraint C_j (schedule consistency at W_{16+j}) absorbs
     a_{j+1} through the single global sigma0(u)-u table, for EVERY j --
     including C3 -> a4, which the earlier absorber inventory omitted.

  2. INVARIANCE.  W_k + W_{k+8} does not depend on a_k (the +1/-1 pair).

  3. EDGE WEIGHTS.  With the unknown window a_1..a_{R-16} absorbed by
     C0..C_{R-17} and a_0 swept, flip one random bit of a_k and record the
     Hamming weight of the change in C_j.  Full avalanche (~16) is a 'heavy'
     edge; a couple of bits is a 'mild' edge that passes only through Maj/Ch.
     At R=19 every FEEDBACK edge (k > j+1) is mild.  At R=20 exactly one
     feedback edge is heavy: a4 -> C0, through W9 at offset 5, where
     Sigma0(a4) is subtracted directly.  At R=21 there are three.

The R=19 fixed-point iteration works because the heavy dependency graph is
acyclic there and the loop can be cut at a mild edge.  A heavy cycle has no
mild cut; any 32-bit cut of it is a random map.

Usage:
    python3 edge_weights.py                 # defaults: trials=200, rounds 19,20,21
    python3 edge_weights.py --trials 50     # quicker, noisier
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sha256_core import sha256_full_trace              # noqa: E402
from extended_solver import backward_chain, compute_e_from_a, recover_W  # noqa: E402

M = 0xFFFFFFFF


def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & M


def s0(x):
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)


def s1(x):
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)


def words(ka, R):
    return recover_W(ka, compute_e_from_a(ka, R), R)


def constraints(ka, R):
    """C_j = W_{16+j} - s1(W_{14+j}) - W_{9+j} - s0(W_{1+j}) - W_j, all recovered."""
    W = words(ka, R)
    return [(W[16 + j] - s1(W[14 + j]) - W[9 + j] - s0(W[1 + j]) - W[j]) & M
            for j in range(R - 16)]


def context(rng, R):
    """Random target, backward chain, random values for every free word."""
    msg = bytes(rng.randrange(256) for _ in range(55))
    hb = sha256_full_trace(msg, num_rounds=R).final_hash
    ka, _ = backward_chain(hb, R)
    ka = dict(ka)
    for r in range(0, R - 8):
        ka[r] = rng.getrandbits(32)
    return ka


def linear_entries(rng, R=20, kmax=5, trials=4):
    print(f"1. how a_k enters W_(k+m)   (+ / - exactly linear, N nonlinear, . none)   R={R}")
    print("   m:    " + " ".join(f"{m:>2}" for m in range(10)))
    ok = True
    for k in range(kmax):
        row = []
        for m in range(10):
            r = k + m
            if r >= R:
                row.append(" ."); continue
            kinds = set()
            for _ in range(trials):
                ka = context(rng, R); base = words(ka, R)
                d = rng.getrandbits(32) | 1
                if d == 0x80000000:          # +d == -d, would be ambiguous
                    d = 3
                t = dict(ka); t[k] = (t[k] + d) & M
                dw = (words(t, R)[r] - base[r]) & M
                kinds.add('+' if dw == d else '-' if dw == (-d) & M
                          else '.' if dw == 0 else 'N')
            c = ''.join(sorted(kinds)) if len(kinds) == 1 else 'N'
            row.append(' ' + c)
            expect = '+' if m == 0 else '-' if m == 8 else '.' if m > 8 else 'N'
            ok &= (c == expect)
        print(f"   a{k}:   " + " ".join(row))
    print(f"   pattern (+ at m=0, - at m=8, N between): {'OK' if ok else 'FAIL'}")
    return ok


def invariance(rng, R=20, ks=(1, 2, 3, 4), trials=8):
    print(f"\n2. W_k + W_(k+8) is independent of a_k   R={R}")
    ok = True
    for k in ks:
        ka = context(rng, R)
        vals = set()
        for _ in range(trials):
            t = dict(ka); t[k] = rng.getrandbits(32)
            W = words(t, R)
            vals.add((W[k] + W[k + 8]) & M)
        good = len(vals) == 1
        ok &= good
        print(f"   k={k}: W{k}+W{k+8} took {len(vals)} distinct value(s) over "
              f"{trials} random a{k}   {'OK' if good else 'FAIL'}")
    return ok


def edge_weights(rng, R, trials, rows=None, absorbed=None, label=None):
    """rows: state words to flip (default a0..a_{R-16}); absorbed[j]: the word
    C_j absorbs, so that (k, j) with k == absorbed[j] is the self-edge and any
    other k is either a forward edge or a feedback edge."""
    n = R - 16
    if rows is None:
        rows = list(range(0, n + 1))
    if absorbed is None:
        absorbed = list(range(1, n + 1))          # C_j -> a_{j+1}
    if label is None:
        label = f"unknown window a1..a{n} absorbed by C0..C{n-1}; a0 swept; rest context"
    print(f"\n3. R={R}: mean Hamming weight of dC_j for a single-bit flip in a_k")
    print(f"   ({label}; {trials} trials per cell)")
    print("          " + " ".join(f"   C{j}" for j in range(n)))
    heavy_feedback = []
    for k in rows:
        row = []
        for j in range(n):
            tot = 0
            for _ in range(trials):
                ka = context(rng, R); base = constraints(ka, R)
                t = dict(ka); t[k] ^= 1 << rng.randrange(32)
                tot += bin(constraints(t, R)[j] ^ base[j]).count('1')
            w = tot / trials
            row.append(f"{w:6.1f}")
            # feedback edge: k is absorbed by a LATER constraint than j
            later = k in absorbed and absorbed.index(k) > j
            if later and w > 4.0:
                heavy_feedback.append((k, j, w))
        print(f"   a{k:<2}:    " + " ".join(row))
    if heavy_feedback:
        print("   heavy feedback edges: " +
              ", ".join(f"a{k}->C{j} ({w:.1f})" for k, j, w in heavy_feedback))
    else:
        print("   heavy feedback edges: none (all feedback is Maj/Ch only)")
    return heavy_feedback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--rounds", default="19,20,21")
    ap.add_argument("--seed", type=int, default=20260902)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    ok = linear_entries(rng)
    ok &= invariance(rng)
    summary = {}
    for R in (int(x) for x in a.rounds.split(",")):
        summary[R] = edge_weights(rng, R, a.trials)
    # The other admissible R=20 frame: context a4..a10, a11 unknown and
    # absorbed (per-context) by C3, a0 swept.
    summary["20A"] = edge_weights(
        rng, 20, a.trials, rows=[0, 1, 2, 3, 11], absorbed=[1, 2, 3, 11],
        label="frame A: unknowns a1,a2,a3,a11 absorbed by C0..C3; a0 swept; a4..a10 context")

    print("\nSummary: heavy feedback edges per round count")
    for R, hf in summary.items():
        print(f"   R={R}: {len(hf)}")
    print("Structural checks:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
