#!/usr/bin/env python3
"""Read exhaust_*.json results; re-measure the lowest-weight contexts with the
full round function (lift.measure, no reduction); look for structure in the
parameters of the tail; report the fraction below thresholds."""
import sys, json, glob, collections
import numpy as np
import lift
from wmodel import Model

files = sys.argv[1:] or sorted(glob.glob("exhaust_*.json"))
for fn in files:
    d = json.load(open(fn))
    w = d['w']; m = Model(w)
    print(f"\n==== {fn}: mode {d['mode']} w={w}, {d['n']:,} contexts; weight min {d['min']:.4f} mean {d['mean']:.4f} sd {d['sd']:.4f} max {d['max']:.4f}")
    print("  fraction below:", {k: f"{v}/{d['n']} = {v/d['n']:.2e}" for k, v in d['below'].items()})
    low = d['lowest']
    print(f"  lowest {len(low)} contexts: weight range {low[0][0]:.4f} .. {low[-1][0]:.4f}; image sizes {sorted(set(r[1] for r in low))}")
    # structure in the tail: value frequencies per parameter slot
    for slot in range(len(low[0][3])):
        vals = collections.Counter(r[3][slot] for r in low)
        top = vals.most_common(4)
        print(f"    param slot {slot}: {len(vals)} distinct among {len(low)}; most common {[(hex(v), c) for v, c in top]}")
    img = d['smallest_image']
    print(f"  smallest images: {[(r[0], r[1], round(r[2], 3)) for r in img[:6]]}  (random-function expectation {(1 << w) * (1 - np.exp(-1)):.0f}/{1 << w})")
    if d['mode'] in ('kern', 'kernrand'):
        print("  (kernel-only mode: contexts are (c9, X, Y); full-engine re-measurement needs a4,a6,a7 realising X,Y -> see lift of a4=a6=a7 family)")
        continue
    if d['mode'] == 'famB':
        print("  (famB: params are (mask, -, -, c8, c9) with a6&a4 random; re-measure with a6=0, a4=mask, a7=a6)")
    print("  full-round-function re-measurement of the 4 lowest (N=48 states, all a5):")
    for r in low[:4]:
        p = r[3]
        if d['mode'] == 'famB':
            a4, a6, a7, c8, c9 = p[0], 0, 0, p[3], p[4]
        else:
            a4, a6, a7, c8, c9 = p
        out = lift.measure(w, a4, a6, a7, c8, c9, N=48, verbose=False)
        print(f"    reduced {r[0]:.4f} img {r[1]} | full engine C0..C3 {np.round(out['mean'], 3).tolist()} image {out['image_mean']:.1f} fibre max {out['fibre_max']} "
              f"sens min {out['sens'].min():.2f}   params {[hex(x) for x in p]}")
