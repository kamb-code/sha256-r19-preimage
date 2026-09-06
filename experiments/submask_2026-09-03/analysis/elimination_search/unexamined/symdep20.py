#!/usr/bin/env python3
"""R=20 symbolic dependency search, extended.

Question: can ANY context word a_k (k = 4..11) be promoted to a fifth unknown,
absorbed by C3, while being at most MILD (Maj/Ch-only) in C0, C1, C2?
Mild feedback can be paid at (3/4)^32-type cost (the collapse mechanism);
heavy feedback (under Sigma0/Sigma1/sigma1, or sigma0 of a non-linear
argument) has no known cheap treatment and is the barrier.

Search space per k:
  saturations   e_r in {free, 0, -1} for r in 8..15 (legal iff e_r is free of
                every unknown; generous: realizability by a free word not checked)
  equalities    none, or one pair a_i = a_j among the remaining context words,
                or one consecutive triple.
This is a superset of the 2,835-configuration search in L5_50_symdep.py.
"""
import itertools, sys, multiprocessing as mp
sys.path.insert(0, "/home/administrator/sha/publish/experiments/submask_2026-09-03/analysis/scripts/symdep")
from L5_50_symdep import (sym, lit, add, neg, op, S0, S1, s0, s1, Maj, Ch, T2,
                          is_lit, lit_val)

R = 20
CTX = list(range(4, 12))


def occ2(expr, var, mode, tags):
    """mode: None (top level, additive), 'mc' (inside Maj/Ch argument), 'heavy'."""
    for at, c in expr:
        if at[0] == 'sym':
            if at[1] == var:
                tags.add({None: 'lin', 'mc': 'mild', 'heavy': 'heavy'}[mode])
        elif at[0] == 'lit':
            pass
        else:
            o = at[0]
            if o in ('S0', 'S1', 's1'):
                occ2(at[1], var, 'heavy', tags)
            elif o == 's0':
                sub = set(); occ2(at[1], var, None, sub)
                if not sub:
                    pass
                elif sub == {'lin'} and mode is None:
                    tags.add('s0lin')
                else:
                    tags.add('heavy')
            elif o in ('Maj', 'Ch'):
                for arg in at[1:]:
                    sub = set(); occ2(arg, var, None, sub)
                    if not sub:
                        continue
                    if sub <= {'lin', 'mild'} and mode in (None, 'mc'):
                        tags.add('mild')
                    else:
                        tags.add('heavy')
            else:
                raise ValueError(o)
    return tags


def classify(expr, var):
    t = occ2(expr, var, None, set())
    if not t:
        return 'none'
    if 'heavy' in t:
        return 'heavy'
    if 'mild' in t:
        return 'mild' if 's0lin' not in t else 'heavy'
    if t == {'lin'}:
        return 'lin'
    if t == {'s0lin'}:
        return 's0lin'
    return 's0lin+lin'


def build(k, eqmap, sat):
    unk = ['a0', 'a1', 'a2', 'a3', f'a{k}']
    a = {}
    for i in (-4, -3, -2, -1):
        a[i] = sym(f'IV{i}')
    for i in range(0, 4):
        a[i] = sym(f'a{i}')
    for i in CTX:
        a[i] = sym(f'a{eqmap.get(i, i)}')
    for i in range(12, R):
        a[i] = sym(f'A{i}')
    e = {}
    for i in (-4, -3, -2, -1):
        e[i] = sym(f'IVe{i}')
    for r in range(0, R):
        e[r] = add(a[r - 4], a[r], neg(T2(a[r - 1], a[r - 2], a[r - 3])))
    for r, tgt in sat.items():
        if tgt is None:
            continue
        if any(occ2(e[r], u, None, set()) for u in unk):
            return None, unk
        e[r] = lit(tgt)
    W = {}
    for r in range(0, 16):
        W[r] = add(a[r], neg(T2(a[r - 1], a[r - 2], a[r - 3])), neg(e[r - 4]),
                   neg(S1(e[r - 1])), neg(Ch(e[r - 1], e[r - 2], e[r - 3])),
                   neg(sym(f'K{r}')))
    for r in range(16, R):
        W[r] = sym(f'Wk{r}')
    C = [add(s0(W[1 + t]), W[t], W[9 + t], s1(W[14 + t]), neg(W[16 + t]))
         for t in range(R - 16)]
    return C, unk


def configs_for(k):
    ctx = [i for i in CTX if i != k]
    eqs = [dict()]
    for i, j in itertools.combinations(ctx, 2):
        eqs.append({j: i})
    for i in ctx:
        if i + 1 in ctx and i + 2 in ctx:
            eqs.append({i + 1: i, i + 2: i})
    sats = []
    for pat in itertools.product((None, 0xFFFFFFFF, 0), repeat=8):
        sats.append({8 + i: pat[i] for i in range(8)})
    return [(k, eq, sat) for eq in eqs for sat in sats]


def run(cfg):
    k, eq, sat = cfg
    C, unk = build(k, eq, sat)
    if C is None:
        return None
    v = f'a{k}'
    row = tuple(classify(C[t], v) for t in range(4))
    # a3 in C1 (the iteration-killer) and a2,a3 in C0 for reference
    extra = (classify(C[1], 'a3'), classify(C[0], 'a2'), classify(C[0], 'a3'))
    return (k, tuple(sorted(eq.items())), tuple(sat[r] for r in range(8, 16)), row, extra)


RANK = {'none': 0, 'lin': 1, 'mild': 2, 's0lin': 3, 's0lin+lin': 3, 'heavy': 4}

if __name__ == "__main__":
    ks = [int(x) for x in sys.argv[1:]] or list(range(4, 12))
    with mp.Pool(24) as pool:
        for k in ks:
            cfgs = configs_for(k)
            res = [r for r in pool.imap_unordered(run, cfgs, chunksize=256) if r]
            n_heavy_min = min(sum(1 for x in r[3][:3] if x == 'heavy') for r in res)
            best = [r for r in res if sum(1 for x in r[3][:3] if x == 'heavy') == n_heavy_min]
            # among those, rank by total feedback weight into C0..C2
            best.sort(key=lambda r: (sum(RANK[x] for x in r[3][:3]), RANK[r[3][3]]))
            print(f"\n=== k={k}: {len(cfgs)} configs, {len(res)} legal; "
                  f"min #heavy(a{k} -> C0,C1,C2) = {n_heavy_min}; "
                  f"{len(best)} configs attain it")
            seen = set()
            shown = 0
            for r in best:
                key = r[3] + r[4]
                if key in seen:
                    continue
                seen.add(key); shown += 1
                print(f"  eq={r[1]} sat(e8..e15)={['-' if x is None else hex(x) for x in r[2]]}")
                print(f"     a{k} in C0..C3: {r[3]}   [a3 in C1: {r[4][0]}, a2 in C0: {r[4][1]}, a3 in C0: {r[4][2]}]")
                if shown >= 6:
                    break
            # which constraints are heavy in the best configs
            from collections import Counter
            cnt = Counter()
            for r in best:
                for t in range(3):
                    if r[3][t] == 'heavy':
                        cnt[f'C{t}'] += 1
            print(f"  heavy-constraint tally over the {len(best)} minimal configs: {dict(cnt)}")
            sys.stdout.flush()
