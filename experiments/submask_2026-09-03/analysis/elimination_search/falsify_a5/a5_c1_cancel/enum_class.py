#!/usr/bin/env python3
"""Enumerate the 'cancel Sigma0(a5) in e6' condition class and measure edge weights.

Frame: a0 swept, a1<-C0, a2<-C1, a3<-C2, a5<-C3 (last).  a4 = v and a6..a11 are
context and may not depend on a5.  Menu per context word (only words whose
definition can touch an a5-atom of C0..C3 are varied; e10/e11 are swept
separately because they do not enter the a5 -> C1 edge at all):

  a6 : free | a6 = a4 | a6 = ~a4 | a6 = rotr(a4, k), k in ROTS
  a7 : free | a7 = a6 | a7 = ~a6 | a7 = a4 | a7 = ~a4 | a7 = rotr(a6, k)
  a8 : free | e8 = t (t in T8; legal only when Maj(a7,a6,a5) collapses)
  a9 : free | e9 - a5 = c9 (c9 in C9; the new 'sat_off' kind, always legal)
  a10: free | e10 = 0 | e10 = -1     (second pass only)
  a11: free | e11 = 0 | e11 = -1     (second pass only)

Per configuration: 200 random states, all 32 single-bit flips of a5, bits
changed in T0..T3; and the same for a4 (context held fixed) as the control.
"""
import itertools, json, sys, time
import numpy as np
from multiprocessing import Pool
import a5edge

M = 0xFFFFFFFF
ROTS = (1, 2, 6, 7, 11, 13, 16, 18, 22, 25, 31)
T8 = (0, M, 0x80000000, 0x7FFFFFFF, 0x55555555, 0xAAAAAAAA, 0x6A09E667, 0x243F6A88)
C9 = (0, M, 1, 0x80000000, 0x7FFFFFFF, 0x55555555, 0xAAAAAAAA, 0x6A09E667, 0x243F6A88, 0x9E3779B9)


def menu6():
    return [('free',), ('eq', 4), ('neq', 4)] + [('rot', 4, k) for k in ROTS]


def menu7():
    return [('free',), ('eq', 6), ('neq', 6), ('eq', 4), ('neq', 4)] + [('rot', 6, k) for k in ROTS]


def menu8():
    return [('free',)] + [('sat_r', t) for t in T8]


def menu9():
    return [('free',)] + [('sat_off', c) for c in C9]


def configs():
    for d6, d7, d8, d9 in itertools.product(menu6(), menu7(), menu8(), menu9()):
        defs = {}
        if d6[0] != 'free': defs[6] = d6
        if d7[0] != 'free': defs[7] = d7
        if d8[0] != 'free': defs[8] = d8
        if d9[0] != 'free': defs[9] = d9
        yield defs


def evaluate(defs, n=200, seed=1):
    np.seterr(over='ignore')
    try:
        w5, pb5 = a5edge.edge_weights(defs, n=n, seed=seed, flip_word=5)
    except a5edge.Illegal as ex:
        return dict(defs=defs, legal=False, why=str(ex))
    w4, _ = a5edge.edge_weights(defs, n=n, seed=seed, flip_word=4)
    return dict(defs=defs, legal=True,
                a5={j: [float(x) for x in w5[j]] for j in range(4)},
                a5_perbit_C1=[float(x) for x in pb5[1]],
                a4={j: [float(x) for x in w4[j]] for j in range(4)})


def keyf(d):
    return json.dumps({str(k): list(v) for k, v in sorted(d.items())})


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "enum_class.json"
    cfgs = list(configs())
    print(f"{len(cfgs)} configurations in the class", flush=True)
    t0 = time.time()
    with Pool(24) as p:
        res = list(p.imap_unordered(evaluate, cfgs, chunksize=8))
    legal = [r for r in res if r['legal']]
    print(f"{len(legal)} legal, {len(res) - len(legal)} illegal (context would reference a5); {time.time() - t0:.0f}s")
    with open(out, "w") as f:
        json.dump([dict(r, defs=keyf(r['defs'])) for r in res], f)
    c1 = np.array([r['a5'][1][0] for r in legal])
    c0 = np.array([r['a5'][0][0] for r in legal])
    c2 = np.array([r['a5'][2][0] for r in legal])
    c3 = np.array([r['a5'][3][0] for r in legal])
    k0 = np.array([r['a4'][0][0] for r in legal])
    print(f"a5 -> C1 : min {c1.min():.2f}  mean {c1.mean():.2f}  max {c1.max():.2f}")
    print(f"a5 -> C0 : min {c0.min():.2f}  mean {c0.mean():.2f}  max {c0.max():.2f}")
    print(f"a5 -> C2 : min {c2.min():.2f}  mean {c2.mean():.2f}  max {c2.max():.2f}")
    print(f"a5 -> C3 : min {c3.min():.2f}  mean {c3.mean():.2f}  max {c3.max():.2f}")
    print(f"a4 -> C0 (control): min {k0.min():.2f}  mean {k0.mean():.2f}  max {k0.max():.2f}")
    order = np.argsort(c1)
    print("\nten lightest a5 -> C1 configurations:")
    for i in order[:10]:
        r = legal[i]
        print(f"  C1 {r['a5'][1][0]:5.2f} [{r['a5'][1][1]:4.1f},{r['a5'][1][2]:4.1f}]  C0 {r['a5'][0][0]:4.2f}  "
              f"C2 {r['a5'][2][0]:4.2f}  C3 {r['a5'][3][0]:4.2f}  a4->C0 {r['a4'][0][0]:5.2f}   {r['defs']}")
    print("\nten heaviest:")
    for i in order[-5:]:
        r = legal[i]
        print(f"  C1 {r['a5'][1][0]:5.2f}  {r['defs']}")
    weak = [r for r in legal if r['a5'][1][0] < 4.0]
    print(f"\nWEAK (< 4 bits) a5 -> C1 configurations: {len(weak)}")
    for r in weak:
        print("  ", r)
