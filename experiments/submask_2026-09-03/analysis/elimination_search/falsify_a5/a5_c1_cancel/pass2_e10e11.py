#!/usr/bin/env python3
"""Second pass: take the lightest a5 -> C1 configurations of enum_class.json and
add e10 in {free, 0, -1} (via a10) and e11 in {free, 0, -1} (via a11), which do
not enter the a5 -> C1 edge but set the a5 -> C2 / C3 edges.  Also re-measures
with a fresh seed and 400 states as a stability check."""
import json, sys, itertools
import numpy as np
from multiprocessing import Pool
import a5edge
from enum_class import evaluate

M = 0xFFFFFFFF


def parse(s):
    d = json.loads(s)
    return {int(k): tuple(v) for k, v in d.items()}


def job(defs):
    np.seterr(over='ignore')
    r = evaluate(defs, n=400, seed=99)
    return r


if __name__ == "__main__":
    res = json.load(open("enum_class.json"))
    legal = [r for r in res if r['legal']]
    legal.sort(key=lambda r: r['a5']['1'][0])
    top = [parse(r['defs']) for r in legal[:40]]
    jobs = []
    for d in top:
        for e10, e11 in itertools.product((None, 0, M), (None, 0, M)):
            dd = dict(d)
            if e10 is not None: dd[10] = ('sat_r', e10)
            if e11 is not None: dd[11] = ('sat_r', e11)
            jobs.append(dd)
    with Pool(24) as p:
        out = list(p.imap_unordered(job, jobs, chunksize=4))
    out = [r for r in out if r['legal']]
    out.sort(key=lambda r: (r['a5'][1][0], r['a5'][0][0] + r['a5'][2][0] + r['a5'][3][0]))
    print(f"{len(out)} legal configurations (top-40 x e10/e11), 400 states, seed 99")
    c1 = np.array([r['a5'][1][0] for r in out])
    print(f"a5 -> C1: min {c1.min():.2f} mean {c1.mean():.2f} max {c1.max():.2f}")
    print("lightest by C1, then by C0+C2+C3:")
    for r in out[:12]:
        w = r['a5']
        print(f"  C1 {w[1][0]:5.2f}  C0 {w[0][0]:4.2f}  C2 {w[2][0]:4.2f}  C3 {w[3][0]:4.2f}  a4->C0 {r['a4'][0][0]:5.2f}  {r['defs']}")
    best_total = min(out, key=lambda r: r['a5'][0][0] + r['a5'][2][0] + r['a5'][3][0])
    w = best_total['a5']
    print(f"lightest C0+C2+C3 (the three edges that CAN be made mild):  C0 {w[0][0]:.2f} C1 {w[1][0]:.2f} C2 {w[2][0]:.2f} C3 {w[3][0]:.2f}  {best_total['defs']}")
    json.dump([dict(r, defs={str(k): list(v) for k, v in r['defs'].items()}) for r in out], open("pass2.json", "w"))
