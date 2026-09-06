#!/usr/bin/env python3
"""Extension of elim/a10/symeng.py with two new context-condition kinds, and an
a5-as-unknown symbolic audit of C1.

New definition kinds (same defs dict format as symeng.build):
    ('sat_off', t)   e_i = a_{i-4} + t, solved for a_i:  a_i = t + T2(a_{i-1},a_{i-2},a_{i-3}).
                     For i = 9 this fixes e9 - a5 = t WITHOUT referencing a5 (legal).
    ('rot', j, k)    a_i = rotr(a_j, k)  (rotational tie; opaque atom ROT for the algebra,
                     it collapses nothing symbolically, and is realised numerically).
    ('sat_r', t) with t any literal already works in symeng (arbitrary constants).

Legality for the a5-unknown frame: a definition is legal iff the resolved expression
of every context word a4, a6..a11 is free of the symbol a5 (and of a0..a3).
"""
import sys
sys.path.insert(0, "/home/administrator/sha/publish/experiments/submask_2026-09-03/analysis/elimination_search/a10")
import symeng as E
from symeng import (M, sym, lit, add, neg, sub, S0, S1, s0, s1, Maj, Ch, T2, op,
                    contains, occurrence, Illegal, IVa, IVe, K, R)

UNK = ('a0', 'a1', 'a2', 'a3', 'a5')


def rot_expr(e, k):
    if E.is_lit(e):
        return lit(E.rotr(E.lit_val(e), k))
    return op('ROT', e, lit(k))


def build_ext(defs, unknown_words=(0, 1, 2, 3, 5), strict=True):
    """symeng.build with the extra kinds.  Words in unknown_words are symbols a_i;
    every other word 4..11 is context and must resolve to an a5-free expression."""
    a = dict(IVa)
    for i in unknown_words:
        a[i] = sym(f'a{i}')
    for i in range(12, R):
        a[i] = sym(f'A{i}')
    ctx = [i for i in range(4, 12) if i not in unknown_words]
    pending = {i: defs.get(i, ('free',)) for i in ctx}
    resolved = set()
    for i, d in pending.items():
        if d[0] == 'free':
            a[i] = sym(f'a{i}'); resolved.add(i)

    def need(d, i):
        k = d[0]
        if k in ('eq', 'neq'): return [d[1]]
        if k == 'rot': return [d[1]]
        if k in ('sat_r', 'sat_off'): return [i - 4, i - 1, i - 2, i - 3]
        if k == 'sat_h': r = i + 4; return [r, r - 1, r - 2, r - 3]
        if k == 'sat_s0': r = i + 1; return [r - 4, r, r - 2, r - 3]
        raise ValueError(d)

    def known(j):
        return (j in resolved) or (j in unknown_words) or (j >= 12) or (j < 4)

    progress = True
    while progress:
        progress = False
        for i, d in pending.items():
            if i in resolved: continue
            if not all(known(j) for j in need(d, i)): continue
            k = d[0]
            if k == 'eq': a[i] = a[d[1]]
            elif k == 'neq': a[i] = sub(lit(M), a[d[1]])
            elif k == 'rot': a[i] = rot_expr(a[d[1]], d[2])
            elif k == 'sat_r':
                r = i
                a[i] = add(lit(d[1]), neg(a[r - 4]), T2(a[r - 1], a[r - 2], a[r - 3]))
            elif k == 'sat_off':
                r = i
                a[i] = add(lit(d[1]), T2(a[r - 1], a[r - 2], a[r - 3]))
            elif k == 'sat_h':
                r = i + 4
                a[i] = add(lit(d[1]), neg(a[r]), T2(a[r - 1], a[r - 2], a[r - 3]))
            elif k == 'sat_s0':
                r = i + 1
                a[i] = op('INV', add(a[r - 4], a[r], neg(lit(d[1]))), a[i - 1], a[i - 2])
            resolved.add(i); progress = True
    if len(resolved) < len(ctx):
        raise Illegal('cyclic definitions')
    if strict:
        for i in ctx:
            for u in [f'a{k}' for k in unknown_words]:
                if contains(a[i], u):
                    raise Illegal(f'context word a{i} references unknown {u}')
    e = dict(IVe)
    for r in range(0, R):
        e[r] = add(a[r - 4], a[r], neg(T2(a[r - 1], a[r - 2], a[r - 3])))
    W = {}
    for r in range(0, R):
        W[r] = add(a[r], neg(T2(a[r - 1], a[r - 2], a[r - 3])), neg(e[r - 4]),
                   neg(S1(e[r - 1])), neg(Ch(e[r - 1], e[r - 2], e[r - 3])), neg(lit(K[r])))
    # lookup targets T_j = W_{16+j} - s1(W_{14+j}) - W_{9+j} - W_j  (= s0(W_{j+1}) on a solution)
    T = [add(W[16 + j], neg(s1(W[14 + j])), neg(W[9 + j]), neg(W[j])) for j in range(4)]
    return dict(a=a, e=e, W=W, T=T)


def atoms_with(expr, name):
    """Top-level atoms of expr containing `name`, as (coeff, description)."""
    out = []
    for at, c in expr:
        if at == E.ONE: continue
        cc = c if c < (1 << 31) else c - (1 << 32)
        if at[0] == 'sym':
            if at[1] == name: out.append((cc, name))
        elif any(contains(arg, name) for arg in at[1:]):
            out.append((cc, E.describe_atom(at, name)))
    return out


def audit(defs, name='a5', verbose=True):
    S = build_ext(defs)
    rep = {}
    for j in range(4):
        rep[j] = atoms_with(S['T'][j], name)
    if verbose:
        print(f"defs={defs}")
        for j in range(4):
            print(f"  T{j} <- {name}: " + ("none" if not rep[j] else "; ".join(f"{c:+d}*{d}" for c, d in rep[j])))
    return S, rep


if __name__ == "__main__":
    base = {7: ('eq', 6), 8: ('sat_r', M)}
    print("== a5 as the unknown, base 'closest miss' context (a7 = a6, e8 = -1) ==")
    S, rep = audit(base)
    print("\n  which message words of C1 = {W1, W2, W10, W15, W17} carry a5:")
    for r in (1, 2, 10, 15, 17):
        print(f"    W{r}: {occurrence(S['W'][r], 'a5')[0]}")
    print("  W10 atoms containing a5:")
    for c, d in atoms_with(S['W'][10], 'a5'):
        print(f"    {c:+d} * {d}")
    print("\n== e9 - a5 fixed to a constant through a9 (legal, new kind sat_off) ==")
    audit({7: ('eq', 6), 8: ('sat_r', M), 9: ('sat_off', 0)})
    print("\n== plus a6 = a4 (kills Maj(a6,a5,a4) in e7) and e11 = 0 (C3 linear) ==")
    audit({6: ('eq', 4), 7: ('eq', 6), 8: ('sat_r', M), 9: ('sat_off', 0), 11: ('sat_r', 0)})
    print("\n== rotational tie a6 = rotr(a4, 13), a7 = rotr(a6, 2) ==")
    audit({6: ('rot', 4, 13), 7: ('rot', 6, 2), 9: ('sat_off', 0)})
    print("\n== ILLEGAL: e9 = -1 solved through a9 (a9 becomes a function of a5) ==")
    try:
        audit({7: ('eq', 6), 8: ('sat_r', M), 9: ('sat_r', M)})
    except Illegal as ex:
        print("  rejected:", ex)
    S = build_ext({7: ('eq', 6), 8: ('sat_r', M), 9: ('sat_r', M)}, strict=False)
    print("  non-strict view (a9 tracks a5): T1 atoms containing a5:")
    for c, d in atoms_with(S['T'][1], 'a5'):
        print(f"    {c:+d} * {d}")
    print("\n== ILLEGAL: a4 = a5 (a4 a function of the unknown) ==")
    try:
        audit({4: ('eq', 5)})
    except Illegal as ex:
        print("  rejected:", ex)
    S = build_ext({4: ('eq', 5), 7: ('eq', 6), 8: ('sat_r', M)}, strict=False)
    for j in range(4):
        print(f"  non-strict T{j} <- a5: " + "; ".join(f"{c:+d}*{d}" for c, d in atoms_with(S['T'][j], 'a5')))
