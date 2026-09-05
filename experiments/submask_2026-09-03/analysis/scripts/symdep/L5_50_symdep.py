#!/usr/bin/env python3
"""Symbolic dependency search: can ANY context family make a 21st-round constraint
solvable, i.e. make the constraint/unknown dependency graph acyclic?

Expressions are canonical sum-of-atoms; Maj/Ch simplify on structural equality and
on saturating constants, exactly the two levers the submask family uses.  For each
(constraint, unknown) pair we classify how the unknown occurs:

   none        : absent
   lin         : only inside sums              -> solve by subtraction
   s0lin       : only inside one s0(x + rest)  -> solve by a sigma0-inversion table
   s0lin+lin   : both                          -> solve by the sigma0(u)-u table (the
                                                  table the attack already builds)
   heavy       : under S0/S1/sigma1, under Maj/Ch that did not collapse, or under
                 two different sigma0's -> no known O(1) inversion
"""
import itertools, sys
from functools import lru_cache

# ---------- expression algebra ----------
# expr = frozenset of (atom, coeff) with coeff in Z (mod 2^32 conceptually)
# atom = ('sym', name) | ('lit', value) | (op, args...)
def sym(n): return frozenset({(('sym', n), 1)})
def lit(v): return frozenset({(('lit', v), 1)})
ZERO = frozenset()

def add(*es):
    d = {}
    for e in es:
        for a, c in e:
            d[a] = d.get(a, 0) + c
    return frozenset((a, c) for a, c in d.items() if c % (1 << 32))

def neg(e): return frozenset((a, -c) for a, c in e)
def sub(a, b): return add(a, neg(b))

def is_lit(e):
    return all(at[0] == 'lit' for at, c in e)

def lit_val(e):
    s = 0
    for at, c in e:
        s += c * at[1]
    return s & 0xFFFFFFFF

def op(name, *args):
    return frozenset({((name,) + tuple(args), 1)})

def S0(e): return op('S0', e)
def S1(e): return op('S1', e)
def s0(e): return op('s0', e)
def s1(e): return op('s1', e)

def Maj(a, b, c):
    if a == b or a == c: return a
    if b == c: return b
    return op('Maj', a, b, c)

def Ch(a, b, c):
    if is_lit(a):
        v = lit_val(a)
        if v == 0xFFFFFFFF: return b
        if v == 0: return c
    if b == c: return b
    return op('Ch', a, b, c)

def T2(a, b, c): return add(S0(a), Maj(a, b, c))

# ---------- occurrence classification ----------
def occ(expr, var, inside=None, tags=None):
    """tags: set of strings describing how `var` occurs."""
    if tags is None: tags = set()
    for at, c in expr:
        if at[0] == 'sym':
            if at[1] == var:
                tags.add('lin' if inside is None else inside)
        elif at[0] == 'lit':
            pass
        else:
            o = at[0]
            if o in ('S0', 'S1', 's1'):
                occ(at[1], var, 'heavy', tags)
            elif o == 's0':
                # s0 of (var + stuff-without-var)?  then s0lin, else heavy
                sub_tags = set(); occ(at[1], var, None, sub_tags)
                if sub_tags == {'lin'}:
                    tags.add('s0lin')
                elif sub_tags:
                    tags.add('heavy')
            elif o in ('Maj', 'Ch'):
                for arg in at[1:]:
                    occ(arg, var, 'heavy', tags)
            else:
                raise ValueError(o)
    return tags

def classify(expr, var):
    t = occ(expr, var)
    if not t: return 'none'
    if 'heavy' in t: return 'heavy'
    if t == {'lin'}: return 'lin'
    if t == {'s0lin'}: return 's0lin'
    if t == {'lin', 's0lin'}: return 's0lin+lin'
    return 'heavy'

SOLVABLE = {'lin', 's0lin', 's0lin+lin'}

# ---------- the compression function, symbolically ----------
R = 21
def build(promote, eqmap, sat):
    """promote: set of indices in {4,5,6} treated as UNKNOWNS.
       eqmap:   dict mapping a-index -> representative a-index (equalities among
                the still-context words of a4..a7).
       sat:     dict r -> 0xFFFFFFFF / 0 / None, saturation target for e_r.
       returns (constraints C0..C4, unknown list, legal flag)."""
    unk = ['a0', 'a1', 'a2', 'a3'] + [f'a{i}' for i in sorted(promote)]
    a = {}
    for i in (-4, -3, -2, -1): a[i] = sym(f'IV{i}')
    for i in range(0, 4): a[i] = sym(f'a{i}')
    for i in range(4, 13):
        nm = f'a{eqmap.get(i, i)}'
        a[i] = sym(nm)
    for i in range(13, R): a[i] = sym(f'A{i}')          # from the backward chain
    e = {}
    for i in (-4, -3, -2, -1): e[i] = sym(f'IVe{i}')
    for r in range(0, R):
        e[r] = add(a[r - 4], a[r], neg(T2(a[r - 1], a[r - 2], a[r - 3])))
    # apply saturations
    for r, tgt in sat.items():
        if tgt is None: continue
        if any(occ(e[r], u) for u in unk):
            return None, unk, False        # a_r would depend on an unknown: illegal
        e[r] = lit(tgt)
    W = {}
    for r in range(0, 16):
        W[r] = add(a[r], neg(T2(a[r - 1], a[r - 2], a[r - 3])), neg(e[r - 4]),
                   neg(S1(e[r - 1])), neg(Ch(e[r - 1], e[r - 2], e[r - 3])),
                   neg(sym(f'K{r}')))
    for r in range(16, R):
        W[r] = sym(f'Wk{r}')                            # known from the chain
    C = [add(s0(W[1 + t]), W[t], W[9 + t], s1(W[14 + t]), neg(W[16 + t]))
         for t in range(5)]
    return C, unk, True


def edge_matrix(promote, eqmap, sat):
    C, unk, legal = build(promote, eqmap, sat)
    if not legal: return None, unk
    return [[classify(C[t], u) for u in unk] for t in range(5)], unk


def schedulable(mat, unk):
    """Sweep one unknown, then greedily see how many constraints can be solved,
    each for a distinct still-unknown variable, in some order (exact search)."""
    best = (0, None)
    n = len(unk)
    for swept in range(n):
        rest = [i for i in range(n) if i != swept]
        # search over orderings: constraint t solves variable j if every OTHER
        # not-yet-solved variable is absent from C_t.
        def rec(known, used_c, depth, plan):
            nonlocal best
            if depth > best[0]:
                best = (depth, (unk[swept], list(plan)))
            for t in range(5):
                if t in used_c: continue
                for j in rest:
                    if j in known: continue
                    if mat[t][j] not in SOLVABLE: continue
                    unresolved = [k for k in rest if k not in known and k != j]
                    if any(mat[t][k] != 'none' for k in unresolved): continue
                    rec(known | {j}, used_c | {t}, depth + 1,
                        plan + [(f'C{t}', unk[j], mat[t][j])])
        rec(frozenset(), frozenset(), 0, [])
    return best


if __name__ == "__main__":
    # ---- 0. the published R=20 frame, as a control
    print("=== control: submask family, R=20/21 frame (unknowns a0..a3) ===")
    eq = {5: 4}                                  # a5 = a4 = v
    sat = {8: 0xFFFFFFFF, 9: 0xFFFFFFFF}
    mat, unk = edge_matrix(set(), eq, sat)
    print("      " + "  ".join(f"{u:>10}" for u in unk))
    for t in range(5): print(f"  C{t}: " + "  ".join(f"{x:>10}" for x in mat[t]))
    print("  best schedule:", schedulable(mat, unk))
