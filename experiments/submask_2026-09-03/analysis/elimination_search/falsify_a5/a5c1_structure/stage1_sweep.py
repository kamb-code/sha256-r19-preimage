#!/usr/bin/env python3
"""Stage 1: extended-menu sweep of the a5 -> C1 edge (and C0, C2, C3), R=20.

Knobs that can reach the a5-dependence of the C1 target (which lives entirely
in W10 = ... - e6 - Sigma1(e9) - Ch(e9,e8,e7)):
  a6 vs a4 : none | eq | neq | rot n | rotneq n | xor c | add c   (Maj(a6,a5,a4) in e7)
  a7 vs a6 : none | eq | neq                                     (legality of e8 via a8)
  a8       : free | e8 = t  (t in V)                              (Ch(e9,e8,e7))
  a9       : free | e9 = a5 + t (t in V)                          (Sigma1(e9), Ch)
e10, e11 do not touch W10 and are swept in stage 2.
"""
import sys, json, itertools, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from a5edge import legal, edge_weights, M

V = [0, M, 0x80000000, 0x7fffffff, 0x55555555, 0xaaaaaaaa, 0x0000ffff, 0xffff0000,
     0x00ff00ff, 0xff00ff00, 0x0f0f0f0f, 0x33333333, 1, 0x6a09e667, 0x428a2f98,
     0x9e3779b9, 0x243f6a88]

a6rel = [None, ('eq', 4), ('neq', 4)]
a6rel += [('rot', 4, n) for n in range(1, 32)]
a6rel += [('rotneq', 4, n) for n in range(1, 32)]
a6rel += [('xor', 4, c) for c in V]
a6rel += [('add', 4, c) for c in V]
a7rel = [None, ('eq', 6), ('neq', 6)]
a8def = [None] + [('sat_r', t) for t in V]
a9def = [None] + [('hoff', t) for t in V]


def make_defs(r6, r7, d8, d9):
    defs = {}
    if r6: defs[6] = r6
    if r7: defs[7] = r7
    if d8: defs[8] = d8
    if d9: defs[9] = d9
    return defs


def work(args):
    idx, defs = args
    rng = np.random.default_rng(1000 + idx)
    if not legal(defs, rng):
        return None
    w, mv = edge_weights(defs, rng, n=200)
    return dict(defs={str(k): v for k, v in defs.items()}, w=w.tolist(), moved=mv.tolist())


if __name__ == "__main__":
    cfgs = [make_defs(*c) for c in itertools.product(a6rel, a7rel, a8def, a9def)]
    print(f"{len(cfgs)} configurations", flush=True)
    t0 = time.time()
    res = []
    with Pool(26) as p:
        for i, r in enumerate(p.imap_unordered(work, list(enumerate(cfgs)), chunksize=32)):
            if r is not None:
                res.append(r)
            if i % 5000 == 0:
                print(f"  {i} done, {len(res)} legal, {time.time()-t0:.0f}s", flush=True)
    print(f"{len(res)} legal of {len(cfgs)}; {time.time()-t0:.0f}s")
    W = np.array([r['w'] for r in res])
    names = ['C0', 'C1', 'C2', 'C3', 'idxC1', 'idxC0']
    for j, nm in enumerate(names):
        print(f"  a5 -> {nm}: min {W[:, j].min():.3f}  mean {W[:, j].mean():.3f}  max {W[:, j].max():.3f}")
    order = np.argsort(W[:, 4])
    print("\n20 lowest a5 -> C1 (exact table index) weights:")
    for k in order[:20]:
        print(f"  {W[k, 4]:.3f}  C0 {W[k,0]:.2f} C2 {W[k,2]:.2f} C3 {W[k,3]:.2f}  {res[k]['defs']}")
    print("\n5 highest:")
    for k in order[-5:]:
        print(f"  {W[k, 4]:.3f}  {res[k]['defs']}")
    below4 = [res[k] for k in range(len(res)) if W[k, 4] < 4]
    print(f"\nconfigurations with a5 -> C1 below 4 bits: {len(below4)}")
    with open("stage1_results.json", "w") as f:
        json.dump(res, f)
