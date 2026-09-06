#!/usr/bin/env python3
"""Substitution-based symbolic dependency engine for the R=20 constraint system.

Differs from analysis/scripts/symdep/L5_50_symdep.py in one essential way: a
context word that is *solved for* (to saturate an e_r, or to equal / complement
another word) is SUBSTITUTED by its defining expression, so that every path by
which the absorbed word w reaches a constraint through a derived word is
tracked.  (L5_50 replaces e_r by a literal and keeps a_r as a free symbol,
which is fine for tracking unknowns a0..a3 but wrong for tracking context
words.)

Expressions: canonical sums frozenset{(atom, coeff)} with coeff mod 2^32.
atom = ('one',) literal | ('sym', name) | (op, args...) with op in
S0 S1 s0 s1 Maj Ch INV.  Collapse rules applied structurally:
  Maj(x,x,y)=x, Maj(x,~x,y)=y, Ch(-1,y,z)=y, Ch(0,y,z)=z, Ch(x,y,y)=y,
  Ch(x,-1,0)=x, Ch(x,0,-1)=~x, Sigma/sigma of a literal evaluated.
"""
from __future__ import annotations
import itertools
from functools import lru_cache

M = 0xFFFFFFFF
ONE = ('one',)


def rotr(x, n): return ((x >> n) | (x << (32 - n))) & M
def nS0(x): return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
def nS1(x): return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
def ns0(x): return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)
def ns1(x): return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)
NUM = {'S0': nS0, 'S1': nS1, 's0': ns0, 's1': ns1}

K = [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
     0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
     0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
     0x0fc19dc6, 0x240ca1cc]


def sym(n): return frozenset({(('sym', n), 1)})
def lit(v):
    v &= M
    return frozenset({(ONE, v)}) if v else frozenset()
ZERO = frozenset()


def add(*es):
    d = {}
    for e in es:
        for a, c in e:
            d[a] = (d.get(a, 0) + c) & M
    return frozenset((a, c) for a, c in d.items() if c)


def neg(e): return frozenset((a, (-c) & M) for a, c in e)
def sub(a, b): return add(a, neg(b))
def is_lit(e): return all(a == ONE for a, c in e)
def lit_val(e):
    for a, c in e:
        if a == ONE:
            return c
    return 0
def is_compl(x, y): return add(x, y) == lit(M)


def op(name, *args): return frozenset({((name,) + tuple(args), 1)})


def unary(name, e):
    if is_lit(e):
        return lit(NUM[name](lit_val(e)))
    return op(name, e)


def S0(e): return unary('S0', e)
def S1(e): return unary('S1', e)
def s0(e): return unary('s0', e)
def s1(e): return unary('s1', e)


def Maj(a, b, c):
    if a == b or a == c: return a
    if b == c: return b
    if is_compl(a, b): return c
    if is_compl(a, c): return b
    if is_compl(b, c): return a
    if is_lit(a) and is_lit(b) and is_lit(c):
        x, y, z = lit_val(a), lit_val(b), lit_val(c)
        return lit((x & y) ^ (x & z) ^ (y & z))
    return op('Maj', a, b, c)


def Ch(a, b, c):
    if is_lit(a):
        v = lit_val(a)
        if v == M: return b
        if v == 0: return c
    if b == c: return b
    if is_lit(b) and is_lit(c):
        y, z = lit_val(b), lit_val(c)
        if y == M and z == 0: return a
        if y == 0 and z == M: return sub(lit(M), a)          # ~a
        if is_lit(a):
            x = lit_val(a)
            return lit((x & y) ^ ((~x) & z))
    return op('Ch', a, b, c)


def T2(a, b, c): return add(S0(a), Maj(a, b, c))


# ---------------------------------------------------------------- occurrence
def contains(expr, name):
    for at, c in expr:
        if at == ONE: continue
        if at[0] == 'sym':
            if at[1] == name: return True
        else:
            if any(contains(arg, name) for arg in at[1:]): return True
    return False


def occurrence(expr, name):
    """('none') | ('lin', coeff) | ('heavy', [top-level heavy atoms as strings], lin coeff)."""
    lin = 0
    heavy = []
    for at, c in expr:
        if at == ONE: continue
        if at[0] == 'sym':
            if at[1] == name: lin = (lin + c) & M
        else:
            if any(contains(arg, name) for arg in at[1:]):
                heavy.append(describe_atom(at, name))
    if heavy: return ('heavy', heavy, lin)
    if lin: return ('lin', lin)
    return ('none',)


def describe_atom(at, name, depth=0):
    """Short description of how `name` sits inside atom `at`."""
    o = at[0]
    parts = []
    for i, arg in enumerate(at[1:]):
        if contains(arg, name):
            occ = occurrence(arg, name)
            if occ[0] == 'lin':
                parts.append(f"arg{i}: {name} linear (coeff {occ[1] if occ[1] < 8 else (occ[1]-(1<<32))})")
            else:
                parts.append(f"arg{i}: [" + "; ".join(occ[1]) + "]")
    return f"{o}(" + " | ".join(parts) + ")"


def classify(expr, name):
    o = occurrence(expr, name)
    return o[0]


# ---------------------------------------------------------------- the system
R = 20
IVa = {i: sym(f'IVa{i}') for i in (-4, -3, -2, -1)}
IVe = {i: sym(f'IVe{i}') for i in (-4, -3, -2, -1)}


class Illegal(Exception):
    pass


def build(defs, unknowns=('a0', 'a1', 'a2', 'a3')):
    """defs: dict i in 4..11 -> definition tuple:
         ('free',)                 a_i is a free context symbol
         ('eq', j) / ('neq', j)    a_i = a_j  /  a_i = ~a_j          (j in 4..19)
         ('sat_r', t)              e_i = t solved for a_i           (i >= 8)
         ('sat_h', t)              e_{i+4} = t solved for a_i        (i+4 <= 15)
         ('sat_s0', t)             e_{i+1} = t solved for a_i through Sigma0(a_i)+Maj(a_i,.,.)
                                   (an opaque per-context inversion INV of the other words)
         ('const', t)              a_i = t                                            [added 2026-09-06]
         ('sat_off', t)            e_i - a_{i-4} = a_i - T2(a_{i-1},a_{i-2},a_{i-3}) = t, solved for a_i
                                   (the legal form of fixing e_i when a_{i-4} is an unknown) [added 2026-09-06]
       Returns dict with a, e, W, C (list of 4 constraint expressions)."""
    a = dict(IVa)
    for i in range(4): a[i] = sym(f'a{i}')
    for i in range(12, R): a[i] = sym(f'A{i}')
    pending = {i: defs.get(i, ('free',)) for i in range(4, 12)}
    # resolve definitions in dependency order
    resolved = set()
    for i, d in pending.items():
        if d[0] == 'free':
            a[i] = sym(f'a{i}'); resolved.add(i)

    def need(d, i):
        k = d[0]
        if k in ('eq', 'neq'): return [d[1]]
        if k == 'sat_r': return [i - 4, i - 1, i - 2, i - 3]
        if k == 'sat_h': r = i + 4; return [r, r - 1, r - 2, r - 3]
        if k == 'sat_s0': r = i + 1; return [r - 4, r, r - 2, r - 3]
        if k == 'const': return []
        if k == 'sat_off': return [i - 1, i - 2, i - 3]
        raise ValueError(d)

    progress = True
    while progress:
        progress = False
        for i, d in pending.items():
            if i in resolved: continue
            deps = need(d, i)
            if all((j in resolved) or (j < 4) or (j >= 12) for j in deps):
                k = d[0]
                if k == 'eq': a[i] = a[d[1]]
                elif k == 'neq': a[i] = sub(lit(M), a[d[1]])
                elif k == 'sat_r':
                    r = i
                    if r < 8: raise Illegal('e_r with unknowns')
                    a[i] = add(lit(d[1]), neg(a[r - 4]), T2(a[r - 1], a[r - 2], a[r - 3]))
                elif k == 'sat_h':
                    r = i + 4
                    if r > 15 or r < 8: raise Illegal('bad r')
                    a[i] = add(lit(d[1]), neg(a[r]), T2(a[r - 1], a[r - 2], a[r - 3]))
                elif k == 'const':
                    a[i] = lit(d[1])
                elif k == 'sat_off':
                    a[i] = add(lit(d[1]), T2(a[i - 1], a[i - 2], a[i - 3]))
                elif k == 'sat_s0':
                    r = i + 1
                    if r < 8 or r > 15: raise Illegal('bad r')
                    # Sigma0(a_i) + Maj(a_i, a_{i-1}, a_{i-2}) = a_{r-4} + a_r - t : opaque inverse
                    a[i] = op('INV', add(a[r - 4], a[r], neg(lit(d[1]))), a[i - 1], a[i - 2])
                resolved.add(i); progress = True
    if len(resolved) < 8:
        raise Illegal('cyclic definitions')
    e = dict(IVe)
    for r in range(0, R):
        e[r] = add(a[r - 4], a[r], neg(T2(a[r - 1], a[r - 2], a[r - 3])))
    W = {}
    for r in range(0, R):
        W[r] = add(a[r], neg(T2(a[r - 1], a[r - 2], a[r - 3])), neg(e[r - 4]),
                   neg(S1(e[r - 1])), neg(Ch(e[r - 1], e[r - 2], e[r - 3])), neg(lit(K[r])))
    C = [add(s0(W[j + 1]), W[j], W[j + 9], s1(W[j + 14]), neg(W[j + 16])) for j in range(4)]
    return dict(a=a, e=e, W=W, C=C)


CWORDS = {0: (0, 1, 9, 14, 16), 1: (1, 2, 10, 15, 17), 2: (2, 3, 11, 16, 18), 3: (3, 4, 12, 17, 19)}


def report_paths(sysd, name, js=(0, 1, 2, 3)):
    """Per constraint and per message word, how `name` occurs."""
    out = []
    for j in js:
        occC = occurrence(sysd['C'][j], name)
        out.append(f"C{j}: {occC[0]}" + (f" (linear coeff {occC[1]})" if occC[0] == 'lin' else ""))
        for r in CWORDS[j]:
            o = occurrence(sysd['W'][r], name)
            if o[0] == 'none': continue
            tag = f"    W{r}"
            if r == j + 1: tag += " (via s0)"
            if r == j + 14: tag += " (via s1)"
            if o[0] == 'lin':
                cf = o[1] if o[1] < 8 else o[1] - (1 << 32)
                out.append(f"{tag}: linear, coeff {cf}")
            else:
                cf = o[2] if o[2] < 8 else o[2] - (1 << 32)
                out.append(f"{tag}: heavy" + (f", plus linear coeff {cf}" if o[2] else ""))
                for h in o[1]:
                    out.append(f"        {h}")
    return "\n".join(out)


if __name__ == "__main__":
    fam = {5: ('eq', 4), 8: ('sat_r', M), 9: ('sat_r', M)}
    S = build(fam)
    # sanity: the family's known facts
    assert S['e'][8] == lit(M) and S['e'][9] == lit(M)
    print("family: e8 = e9 = -1 by substitution: OK")
    print("C1 classification of a3:", classify(S['C'][1], 'a3'), "(Prop.1 wants none)")
    print("C0 classification of a4:", classify(S['C'][0], 'a4'))
    for w in ('a10', 'a11', 'a7', 'a6', 'a5', 'a4'):
        print(f"\n=== occurrences of {w} in the base family ===")
        print(report_paths(S, w))
