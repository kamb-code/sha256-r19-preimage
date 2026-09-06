#!/usr/bin/env python3
"""Exhaustive enumeration over a menu of context conditions: can ANY choice of
definitions for the other context words make the absorbed word w absent from
the lookup targets C0, C1, C2 (while present in C3 = kappa3)?

For w in {a10, a11, a7, a6, a5}: a4 stays free; every other word a_i, i in
5..11, i != w, takes one of:
   free | a_i = a_j | a_i = ~a_j  (|i-j| <= 2, 4 <= j <= 13)
   | e_i = 0/-1 solved for a_i (i >= 8)
   | e_{i+4} = 0/-1 solved for a_i
   | e_{i+1} = 0/-1 solved for a_i through Sigma0(a_i)+Maj (opaque INV)
Usage: python3 enum_ctx.py a10 [ncore]
"""
import sys, itertools, collections, multiprocessing as mp, time, pickle
import symeng as E

M = E.M


def menu(i, wi):
    opts = [('free',)]
    for j in range(max(4, i - 2), min(13, i + 2) + 1):
        if j == i: continue
        opts.append(('eq', j)); opts.append(('neq', j))
    if i >= 8:
        opts += [('sat_r', 0), ('sat_r', M)]
    if i + 4 <= 15:
        opts += [('sat_h', 0), ('sat_h', M)]
    if 8 <= i + 1 <= 15:
        opts += [('sat_s0', 0), ('sat_s0', M)]
    return opts


def evaluate(args):
    wi, first_opt, others = args
    w = f'a{wi}'
    words = [i for i in range(5, 12) if i != wi]
    res = collections.Counter()
    best = []
    n_legal = 0
    for rest in itertools.product(*[menu(i, wi) for i in words[1:]]):
        defs = {words[0]: first_opt}
        defs.update(zip(words[1:], rest))
        try:
            S = E.build(defs)
        except E.Illegal:
            continue
        n_legal += 1
        cls = tuple(E.classify(S['C'][j], w) for j in range(4))
        tri = E.classify(S['C'][1], 'a3') == 'none'
        res[(cls, tri)] += 1
        score = sum(c != 'none' for c in cls[:3])
        if score <= 1 or cls[3] == 'none' and score <= 2:
            best.append((score, cls, tri, dict(defs)))
    return n_legal, res, best


if __name__ == "__main__":
    wi = int(sys.argv[1].lstrip('a'))
    ncore = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    words = [i for i in range(5, 12) if i != wi]
    first = menu(words[0], wi)
    total = 1
    for i in words: total *= len(menu(i, wi))
    print(f"w = a{wi}: words {words}, menu sizes {[len(menu(i, wi)) for i in words]}, "
          f"{total:,} configurations", flush=True)
    t0 = time.time()
    agg = collections.Counter(); n_legal = 0; best = []
    with mp.Pool(ncore) as p:
        for nl, res, b in p.imap_unordered(evaluate, [(wi, o, None) for o in first]):
            n_legal += nl; agg.update(res); best += b
            print(f"  ... {n_legal:,} legal so far, {time.time()-t0:.0f}s", flush=True)
    print(f"\nlegal (acyclic) configurations: {n_legal:,} of {total:,}; {time.time()-t0:.0f}s")
    print("\n(C0, C1, C2, C3) occurrence class of w, triangular(a3 not in C1) -> count")
    for (cls, tri), n in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {cls}  tri={tri}: {n:,}")
    absorbable = [b for b in best if b[0] == 0]
    print(f"\nconfigurations with w absent from C0, C1 and C2: {len(absorbable)}")
    for b in absorbable[:20]: print("   ", b)
    print(f"configurations with w absent from two of C0,C1,C2: {sum(1 for b in best if b[0]==1)}")
    for b in [b for b in best if b[0] == 1][:20]: print("   ", b)
    with open(f"enum_a{wi}.pkl", "wb") as f:
        pickle.dump(dict(agg=agg, best=best, n_legal=n_legal, total=total), f)
