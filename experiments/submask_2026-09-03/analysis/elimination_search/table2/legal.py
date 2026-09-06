#!/usr/bin/env python3
"""Class-based legality of a context configuration, and a constructor.

A configuration is (w, eqmap, sat).  Context words a4..a11 minus w are grouped
into equality classes (one free 32-bit value each).  A saturation e_r = c is
solved through one class X and one member x in X with x in {r-4, r} (the two
words entering e_r with coefficient +1), provided X has no member in
{r-1, r-2, r-3} (those enter e_r through Sigma0/Maj) and not both r-4 and r
are in X (coefficient 2 is not invertible).  The classes are solved in an
order such that the class solving e_r enters no earlier-solved e_{r'}
(a word x enters e_x .. e_{x+4}).  A plan is a list (r, X, x) in solving order.
"""
import itertools
import numpy as np

M = 0xFFFFFFFF

def classes_of(w, eqmap):
    """classes with a free value; a class tied to a digest word (rep >= 12) has none."""
    cls = {}
    for i in range(4, 12):
        if i == w: continue
        rep = eqmap.get(i, i)
        if rep >= 12: continue
        cls.setdefault(rep, set()).add(i)
    return {rep: frozenset(m) for rep, m in cls.items()}

def plan(w, eqmap, sat):
    cls = classes_of(w, eqmap)
    sats = sorted(sat)
    if not sats: return []
    opts = {}
    for r in sats:
        o = []
        for rep, X in cls.items():
            if X & {r - 1, r - 2, r - 3}: continue
            if (r - 4) in X and r in X: continue
            for x in (r - 4, r):
                if x in X: o.append((rep, x))
        opts[r] = o
        if not o: return None
    for order in itertools.permutations(sats):
        for choice in itertools.product(*[opts[r] for r in order]):
            reps = [c[0] for c in choice]
            if len(set(reps)) != len(reps): continue
            ok = True
            for k, (r, (rep, x)) in enumerate(zip(order, choice)):
                X = cls[rep]
                if any(any(y <= rp <= y + 4 for y in X) for rp in order[:k]):
                    ok = False; break
            if ok: return [(r, rep, x) for r, (rep, x) in zip(order, choice)]
    return None

def legal(w, eqmap, sat):
    return plan(w, eqmap, sat) is not None

def T2(a, b, c):
    def rotr(x, n): return ((x >> n) | (x << (32 - n))) & M
    S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
    return (S0 + ((a & b) ^ (a & c) ^ (b & c))) & M

def construct(rng, w, eqmap, sat, R=20, tries=20, pl=None):
    """random context a4..a11 (+ random a12..a19) satisfying the configuration."""
    if pl is None: pl = plan(w, eqmap, sat)
    if pl is None: return None
    cls = classes_of(w, eqmap)
    for _ in range(tries):
        a = {i: int(rng.integers(0, 1 << 32, dtype=np.uint64)) for i in range(4, R)}
        for i, rep in eqmap.items(): a[i] = a[rep]
        for r, rep, x in pl:
            other = a[r] if x == r - 4 else a[r - 4]
            v = (sat[r] - other + T2(a[r - 1], a[r - 2], a[r - 3])) & M
            for y in cls[rep]: a[y] = v
        good = all((a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M == sat[r] for r in sat) \
            and all(a[i] == a[rep] for i, rep in eqmap.items())
        if good: return a
    return None

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(plan(6, {7: 5, 8: 5}, {8: 0, 9: M, 11: M, 12: 0}))
    print(plan(9, {10: 8}, {14: M}), construct(rng, 9, {10: 8}, {14: M}) is not None)
    print(plan(None, {5: 4}, {8: M, 9: M}), construct(rng, None, {5: 4}, {8: M, 9: M}) is not None)
    print(plan(4, {}, {10: M, 11: 0, 13: 0}), construct(rng, 4, {}, {10: M, 11: 0, 13: 0}) is not None)
