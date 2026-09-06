#!/usr/bin/env python3
"""Symbolic dependency engine at R=20 (adapted from L5_50_symdep.py, R param).

Question 1: in frame B (a4 promoted to an unknown, absorbed by C3), is there ANY
legal context configuration (equalities among context words a5..a7, saturations
e8..e11 in {-1, 0, none}) under which C0 is free of a4, or depends on it only
through Maj/Ch (mild)?  We also record, for every configuration, the exact
top-level atoms of C0 that carry a4, to name the blocking term.

Question 2: which functions carry each FREE CONTEXT WORD (a6,a7,a10,a11, and v)
into the constraint constants KC0..KC2 and kappa3 (the unknown-free parts of
C0..C3)?  A word that reached kappa3 only, or reached KC0..KC2 only linearly /
through saturable Maj/Ch, would be the invariance the multi-context
amortisation needs.
"""
import itertools, sys
sys.path.insert(0, "/home/administrator/sha/publish/experiments/submask_2026-09-03/analysis/scripts/symdep")
from L5_50_symdep import (sym, lit, ZERO, add, neg, sub, is_lit, lit_val, op, S0, S1,
                          s0, s1, Maj, Ch, T2, occ, classify, SOLVABLE)

R = 20
NC = R - 16          # constraints C0..C3


def build(promote, eqmap, sat, family_a8a9=False):
    """promote: set of indices in {4,5,6,7} treated as unknowns.
       eqmap: a-index -> representative a-index among the context words.
       sat: r -> 0xFFFFFFFF / 0 / None for e_r (r in 8..11).
       family_a8a9: model a8, a9 as the family's derived words (a8 = -1 - a4 +
         T2(a7,a6,a5) etc.), i.e. e8 = e9 = -1 EXACTLY, with a8 and a9 carrying
         the context symbols they are built from (used for question 2)."""
    unk = ['a0', 'a1', 'a2', 'a3'] + [f'a{i}' for i in sorted(promote)]
    a = {}
    for i in (-4, -3, -2, -1): a[i] = sym(f'IV{i}')
    for i in range(0, 4): a[i] = sym(f'a{i}')
    for i in range(4, 12):
        a[i] = sym(f'a{eqmap.get(i, i)}')
    for i in range(12, R): a[i] = sym(f'A{i}')            # backward chain
    e = {}
    for i in (-4, -3, -2, -1): e[i] = sym(f'IVe{i}')
    if family_a8a9:
        # a8 = -1 - a4 + S0(a7) + Maj(a7,a6,a5); a9 = -1 - a5 + S0(a8) + Maj(a8,a7,a6)
        a[8] = add(lit(0xFFFFFFFF), neg(a[4]), T2(a[7], a[6], a[5]))
        a[9] = add(lit(0xFFFFFFFF), neg(a[5]), T2(a[8], a[7], a[6]))
    for r in range(0, R):
        e[r] = add(a[r - 4], a[r], neg(T2(a[r - 1], a[r - 2], a[r - 3])))
    if family_a8a9:
        e[8] = lit(0xFFFFFFFF); e[9] = lit(0xFFFFFFFF)    # exact by construction
    for r, tgt in sat.items():
        if tgt is None: continue
        if any(occ(e[r], u) for u in unk):
            return None, None, unk, False                   # would tie context to an unknown
        e[r] = lit(tgt)
    W = {}
    for r in range(0, 16):
        W[r] = add(a[r], neg(T2(a[r - 1], a[r - 2], a[r - 3])), neg(e[r - 4]),
                   neg(S1(e[r - 1])), neg(Ch(e[r - 1], e[r - 2], e[r - 3])), neg(sym(f'K{r}')))
    for r in range(16, R):
        # the chain words are functions of a8..a19 -- keep them explicit so that
        # context words a8..a11 show their routes (question 2)
        W[r] = add(a[r], neg(T2(a[r - 1], a[r - 2], a[r - 3])), neg(e[r - 4]),
                   neg(S1(e[r - 1])), neg(Ch(e[r - 1], e[r - 2], e[r - 3])), neg(sym(f'K{r}')))
    C = [add(s0(W[1 + t]), W[t], W[9 + t], s1(W[14 + t]), neg(W[16 + t])) for t in range(NC)]
    return C, W, unk, True


def atoms_with(expr, var):
    """Top-level atoms of expr that contain var, rendered compactly."""
    out = []
    for at, c in expr:
        if occ(frozenset({(at, 1)}), var):
            out.append((c, render(at)))
    return out


def render(at, depth=0):
    if at[0] == 'sym': return at[1]
    if at[0] == 'lit': return hex(at[1])
    name = at[0]
    args = []
    for e in at[1:]:
        args.append(render_expr(e, depth + 1))
    return f"{name}({', '.join(args)})"


def render_expr(e, depth=0):
    if depth > 2: return "..."
    parts = []
    for at, c in sorted(e, key=lambda x: str(x)):
        s = render(at, depth)
        parts.append(("+" if c > 0 else "-") + (f"{abs(c)}*" if abs(c) != 1 else "") + s)
    return "(" + " ".join(parts) + ")"


def schedulable(mat, unk):
    best = (0, None)
    n = len(unk)
    for swept in range(n):
        rest = [i for i in range(n) if i != swept]
        def rec(known, used_c, depth, plan):
            nonlocal best
            if depth > best[0]:
                best = (depth, (unk[swept], list(plan)))
            for t in range(NC):
                if t in used_c: continue
                for j in rest:
                    if j in known: continue
                    if mat[t][j] not in SOLVABLE: continue
                    unresolved = [k for k in rest if k not in known and k != j]
                    if any(mat[t][k] != 'none' for k in unresolved): continue
                    rec(known | {j}, used_c | {t}, depth + 1, plan + [(f'C{t}', unk[j], mat[t][j])])
        rec(frozenset(), frozenset(), 0, [])
    return best


if __name__ == "__main__":
    # ---------------- Question 1: frame B, a4 promoted ----------------
    print("=== Q1: frame B (unknowns a0..a4), all legal context configurations at R=20 ===")
    ctx_words = [5, 6, 7]
    n_legal = n_total = 0
    c0_a4_classes = {}
    s0a4_always = True
    examples = {}
    # eqmaps: each of a5,a6,a7 either itself or equal to a smaller context word (>=5)
    def eqmaps():
        for choice in itertools.product(*[[i] + list(range(5, i)) for i in ctx_words]):
            yield dict(zip(ctx_words, choice))
    for eqm in eqmaps():
        for sat_vals in itertools.product([None, 0xFFFFFFFF, 0], repeat=4):
            sat = dict(zip(range(8, 12), sat_vals))
            n_total += 1
            C, W, unk, legal = build({4}, eqm, sat)
            if not legal: continue
            n_legal += 1
            cls = classify(C[0], 'a4')
            c0_a4_classes[cls] = c0_a4_classes.get(cls, 0) + 1
            # add() flattens sums, so -e5 inside W9 inside C0 puts S0(a4) at top level
            has_s0a4 = any(at[0] == 'S0' and at[1] == sym('a4') and c % (1 << 32) != 0
                           for at, c in C[0])
            if not has_s0a4:
                s0a4_always = False
            key = (cls, has_s0a4)
            examples.setdefault(key, (eqm, sat))
            # full matrix, schedulability
            mat = [[classify(C[t], u) for u in unk] for t in range(NC)]
            sched = schedulable(mat, unk)
            if sched[0] >= 4:
                print("  !!! fully schedulable configuration found:", eqm, sat, sched)
    print(f"  configurations: {n_total} total, {n_legal} legal")
    print(f"  classification of a4 in C0 over legal configs: {c0_a4_classes}")
    print(f"  atom S0(a4) present in C0 in EVERY legal config: {s0a4_always}")
    # show the a4-carrying atoms of C0 in the control (family-on-context) config
    C, W, unk, legal = build({4}, {5: 5, 6: 6, 7: 7}, {8: None, 9: 0xFFFFFFFF, 10: None, 11: None})
    print("\n  a4-carrying top-level atoms of C0 (a5..a7 free, e9 = -1; e8 cannot be saturated"
          " because e8 = a4 + a8 - T2(a7,a6,a5) contains the unknown a4):")
    for c, s in atoms_with(C[0], 'a4'):
        print(f"    {'+' if c>0 else '-'} {s}")
    print("  a4-carrying atoms of C1:")
    for c, s in atoms_with(C[1], 'a4'):
        print(f"    {'+' if c>0 else '-'} {s}")
    print("  a4-carrying atoms of C2:")
    for c, s in atoms_with(C[2], 'a4'):
        print(f"    {'+' if c>0 else '-'} {s}")
    mat = [[classify(C[t], u) for u in unk] for t in range(NC)]
    print("\n  dependency classes (frame B, e9=-1):")
    print("      " + "  ".join(f"{u:>10}" for u in unk))
    for t in range(NC): print(f"  C{t}: " + "  ".join(f"{x:>10}" for x in mat[t]))
    print("  best schedule:", schedulable(mat, unk))

    # control: the published frame (a4 context)
    C, W, unk, legal = build(set(), {5: 4, 6: 6, 7: 7}, {8: 0xFFFFFFFF, 9: 0xFFFFFFFF, 10: None, 11: None})
    mat = [[classify(C[t], u) for u in unk] for t in range(NC)]
    print("\n  control: published family (a5=a4 context, e8=e9=-1), unknowns a0..a3:")
    print("      " + "  ".join(f"{u:>10}" for u in unk))
    for t in range(NC): print(f"  C{t}: " + "  ".join(f"{x:>10}" for x in mat[t]))
    print("  best schedule:", schedulable(mat, unk))

    # ---------------- Question 2: context words -> constants ----------------
    print("\n=== Q2: how each free context word reaches the constraint constants (family) ===")
    C, W, unk, legal = build(set(), {5: 4, 6: 6, 7: 7}, {10: None, 11: None}, family_a8a9=True)
    for word in ['a4', 'a6', 'a7', 'a10', 'a11']:
        print(f"\n  word {word} (a4 = v):")
        for t in range(NC):
            tags = occ(C[t], word)
            print(f"    C{t}: occurrence tags {sorted(tags) if tags else 'ABSENT'}")
        # name the one-input-function routes into C0
        routes = []
        for c, s in atoms_with(C[0], word):
            if s.startswith(('S0(', 'S1(', 's0(', 's1(')):
                routes.append(s[:90])
        print(f"    one-input-function routes into C0 (truncated): {len(routes)}")
        for r_ in routes[:4]: print(f"      {r_}")
