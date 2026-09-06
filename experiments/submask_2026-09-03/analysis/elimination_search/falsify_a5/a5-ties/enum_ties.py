#!/usr/bin/env python3
"""Enumerate the class of rotational / linear ties among context words (never a5),
combined with saturations of e8..e11 in {0, -1}, and measure a5 -> C0..C3 and the
a4 -> C0 control edge weights for every condition with tie_scan.measure.

Output: results.jsonl (one line per condition) and a summary on stdout.
"""
import itertools, json, sys, time, os
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tie_scan as T

M = 0xFFFFFFFF
CTX = [4, 6, 7, 8, 9, 10, 11]
rng = np.random.default_rng(20260906)


def rc(n):
    return [int(x) for x in rng.integers(0, 1 << 32, n, dtype=np.uint64)]


SPECIAL_XOR = [M, 1, 0x80000000, 0x55555555, 0xAAAAAAAA, 0x0000FFFF, 0xFFFF0000, 0x01010101]
SPECIAL_ADD = [1, M, 0x80000000, 0x7FFFFFFF, 0x10000]
SPECIAL_S = [0, M, 1, 0x80000000]


def ties():
    """Yield (class_name, {word: def, ...}) — one or two tie definitions."""
    # A/B/C rotational ties among a4, a6, a7
    for n in range(32):
        yield 'rot a7=ROTR^n(a6)', {7: ('rotr', 6, n)}
        yield 'rot a6=ROTR^n(a4)', {6: ('rotr', 4, n)}
        yield 'rot a7=ROTR^n(a4)', {7: ('rotr', 4, n)}
    # D xor with constant
    for c in rc(8) + SPECIAL_XOR:
        yield 'xor a7=a6^c', {7: ('xor', 6, c)}
        yield 'xor a6=a4^c', {6: ('xor', 4, c)}
    # E add constant
    for c in rc(8) + SPECIAL_ADD:
        yield 'add a7=a6+c', {7: ('add', 6, c)}
        yield 'add a6=a4+c', {6: ('add', 4, c)}
    # F Sigma0 ties
    for c in rc(8) + SPECIAL_S:
        yield 'S0 a6=S0(a4)+c', {6: ('S0add', 4, c)}
        yield 'S0 a7=S0(a6)+c', {7: ('S0add', 6, c)}
        yield 'S0 a7=S0(a4)+c', {7: ('S0add', 4, c)}
        yield 'S1 a7=S1(a6)+c', {7: ('S1add', 6, c)}
        yield 'S1 a6=S1(a4)+c', {6: ('S1add', 4, c)}
    # H Maj with constants
    majc = [tuple(rc(2)) for _ in range(8)] + [(0, M), (M, 0), (0x55555555, 0xAAAAAAAA)]
    for c in rc(4):
        majc += [(c, c), (0, c), (M, c)]
    for c1, c2 in majc:
        yield 'maj a6=Maj(a4,c1,c2)', {6: ('maj', 4, c1, c2)}
        yield 'maj a7=Maj(a6,c1,c2)', {7: ('maj', 6, c1, c2)}
        yield 'maj a7=Maj(a4,c1,c2)', {7: ('maj', 4, c1, c2)}
    # I GF(2)-linear weight-2 ties between pairs among a4, a6, a7
    for n1, n2 in itertools.combinations(range(32), 2):
        yield 'lin2 a7=L(a6)', {7: ('lin', 6, (n1, n2))}
        yield 'lin2 a6=L(a4)', {6: ('lin', 4, (n1, n2))}
        yield 'lin2 a7=L(a4)', {7: ('lin', 4, (n1, n2))}
    # J rotational ties between every other ordered pair of context words
    for x in CTX:
        for y in CTX:
            if x == y or {x, y} <= {4, 6, 7}:
                continue
            for n in range(32):
                yield f'rot a{y}=ROTR^n(a{x})', {y: ('rotr', x, n)}
    # K two ties at once: a7 = ROTR^m(a6) and a6 = ROTR^n(a4)
    for m in range(32):
        for n in range(32):
            yield 'rot2 a7=ROTR^m(a6),a6=ROTR^n(a4)', {7: ('rotr', 6, m), 6: ('rotr', 4, n)}


def ties_w3(sample=600):
    allt = list(itertools.combinations(range(32), 3))
    pick = rng.choice(len(allt), sample, replace=False)
    for i in sorted(pick):
        ns = allt[i]
        yield 'lin3 a7=L(a6)', {7: ('lin', 6, ns)}
        yield 'lin3 a6=L(a4)', {6: ('lin', 4, ns)}
        yield 'lin3 a7=L(a4)', {7: ('lin', 4, ns)}


def sat_grid(full=True, e9=True):
    """Saturation patterns for e8..e11: value in {none, 0, -1}; e8, e9 solved through
    a8, a9 (sat_r); e10, e11 through a10, a11 (sat_r) or a6, a7 (sat_h)."""
    opts8 = [None, (8, 'sat_r', 0), (8, 'sat_r', M)]
    opts9 = [None, (9, 'sat_r', 0), (9, 'sat_r', M)]
    opts10 = [None, (10, 'sat_r', 0), (10, 'sat_r', M)] + ([(6, 'sat_h', 0), (6, 'sat_h', M)] if full else [])
    opts11 = [None, (11, 'sat_r', 0), (11, 'sat_r', M)] + ([(7, 'sat_h', 0), (7, 'sat_h', M)] if full else [])
    if not e9:
        opts9 = [None]
    for combo in itertools.product(opts8, opts9, opts10, opts11):
        d = {}
        ok = True
        for c in combo:
            if c is None:
                continue
            if c[0] in d:
                ok = False
                break
            d[c[0]] = (c[1], c[2])
        if ok:
            yield d


def conditions():
    grid = list(sat_grid(True, True))
    grid_small = list(sat_grid(False, False))     # e8, e10, e11 in {none, 0, -1} via a_r; no e9
    for cls, tie in ties():
        g = grid_small if cls.split()[0] in ('lin2', 'rot2') else grid
        for sat in g:
            if set(tie) & set(sat):
                continue
            defs = dict(tie)
            defs.update(sat)
            if T.order_of(defs) is None:
                continue
            yield cls, defs
    grid3 = [d for d in sat_grid(False, False) if 10 not in d]   # e8, e11 only
    for cls, tie in ties_w3():
        for sat in grid3:
            if set(tie) & set(sat):
                continue
            defs = dict(tie)
            defs.update(sat)
            if T.order_of(defs) is None:
                continue
            yield cls, defs


def work(item):
    idx, cls, defs = item
    res = T.measure(defs, N=200, seed=1)
    if res is None:
        return None
    r5, r4 = res[5], res[4]
    return dict(idx=idx, cls=cls, defs={str(k): list(v) for k, v in defs.items()},
                a5=r5['mean'], a5_moved=r5['moved'], a5_min=r5['min'], strict=not r5['ctx_dep'],
                a4C0=(r4['mean'][0] if r4 else None), a4ctxdep=(r4['ctx_dep'] if r4 else None))


if __name__ == '__main__':
    conds = [(i, c, d) for i, (c, d) in enumerate(conditions())]
    print(f"{len(conds)} conditions", flush=True)
    t0 = time.time()
    out = open(sys.argv[1] if len(sys.argv) > 1 else 'results.jsonl', 'w')
    n = 0
    best = None
    with Pool(10) as p:
        for r in p.imap_unordered(work, conds, chunksize=64):
            if r is None:
                continue
            out.write(json.dumps(r) + '\n')
            n += 1
            if best is None or r['a5'][1] < best['a5'][1]:
                best = r
            if n % 10000 == 0:
                print(f"  {n} done, {time.time()-t0:.0f}s, best a5->C1 so far {best['a5'][1]:.2f} {best['cls']} {best['defs']}", flush=True)
    out.close()
    print(f"{n} measured in {time.time()-t0:.0f}s")
    print("best a5->C1:", best)
