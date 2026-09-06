#!/usr/bin/env python3
"""Stage 2: knobs that do NOT reach W10 (e10, e11 constants, a7 rotational /
xor / add ties, a4 tied to a digest word, a10/a11 ties) on top of the best
stage-1 configurations; confirms C1 invariance and records C0/C2/C3.
Also a control table: a4 -> C0 in the random context and in the best config."""
import sys, json, itertools, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from a5edge import legal, edge_weights, M
from stage1_sweep import V

BASES = [
    {6: ('neq', 4), 7: ('eq', 6), 8: ('sat_r', M)},                  # a5 absent from C0 (e7 -> Maj collapses to a5? no: to a5) see log
    {6: ('eq', 4), 7: ('eq', 6), 8: ('sat_r', M)},
    {7: ('eq', 6), 8: ('sat_r', M)},
    {6: ('neq', 4), 7: ('eq', 6), 8: ('sat_r', M), 9: ('hoff', 0)},
    {6: ('neq', 4), 7: ('eq', 6), 8: ('sat_r', 0), 9: ('hoff', 0)},
]
extra10 = [None] + [('sat_r', t) for t in V] + [('sat_h', t) for t in (0, M)] + [('eq', 8), ('neq', 8)]
extra11 = [None] + [('sat_r', t) for t in V] + [('sat_h', t) for t in (0, M)] + [('eq', 9), ('neq', 9), ('eq', 12), ('neq', 12)]
a7extra = [None] + [('rot', 6, n) for n in (1, 5, 13, 31)] + [('xor', 6, 0x55555555), ('add', 6, 1)]


def work(args):
    idx, defs = args
    rng = np.random.default_rng(5000 + idx)
    if not legal(defs, rng):
        return None
    w, mv = edge_weights(defs, rng, n=200)
    return dict(defs={str(k): v for k, v in defs.items()}, w=w.tolist(), moved=mv.tolist())


if __name__ == "__main__":
    cfgs = []
    for b in BASES:
        for d10, d11 in itertools.product(extra10, extra11):
            d = dict(b)
            if d10: d[10] = d10
            if d11: d[11] = d11
            cfgs.append(d)
    # a7 ties that break the Maj collapse (a8 must then be free)
    for r7 in a7extra[1:]:
        for r6 in (None, ('eq', 4), ('neq', 4)):
            for d9 in (None, ('hoff', 0)):
                d = {7: r7}
                if r6: d[6] = r6
                if d9: d[9] = d9
                cfgs.append(d)
    # a4 tied to digest words / a8..a11 tied among themselves
    for d4 in (('eq', 12), ('neq', 12), ('eq', 13), ('rot', 12, 7)):
        d = dict(BASES[0]); d[4] = d4; cfgs.append(d)
    print(f"{len(cfgs)} configurations", flush=True)
    t0 = time.time()
    with Pool(26) as p:
        res = [r for r in p.imap_unordered(work, list(enumerate(cfgs)), chunksize=8) if r is not None]
    print(f"{len(res)} legal; {time.time()-t0:.0f}s")
    W = np.array([r['w'] for r in res])
    for j, nm in enumerate(['C0', 'C1', 'C2', 'C3', 'idxC1', 'idxC0']):
        print(f"  a5 -> {nm}: min {W[:, j].min():.3f}  mean {W[:, j].mean():.3f}  max {W[:, j].max():.3f}")
    order = np.argsort(W[:, 4])
    print("\nlowest a5 -> C1:")
    for k in order[:8]:
        print(f"  C1 {W[k,4]:.3f}  C0 {W[k,0]:.2f} C2 {W[k,2]:.2f} C3 {W[k,3]:.2f}  {res[k]['defs']}")
    print("\nconfigs with a5 absent from C0 and C2 and C3 linear (C0=C2=0, C3 moved 100%):")
    n = 0
    for r in res:
        if r['w'][0] == 0 and r['w'][2] == 0 and r['moved'][3] == 1.0:
            n += 1
            if n <= 6:
                print(f"  C1 {r['w'][4]:.3f} C3 {r['w'][3]:.2f}  {r['defs']}")
    print(f"  ({n} such configurations)")
    # controls
    rng = np.random.default_rng(99)
    w, _ = edge_weights({}, rng, n=200, flip=4, unknown=4)
    print(f"\ncontrol: a4 -> [C0 C1 C2 C3 idxC1 idxC0], random context, a4 unknown: {np.round(w, 2)}")
    w, _ = edge_weights({7: ('eq', 6), 9: ('sat_r', M)}, rng, n=200, flip=4, unknown=4)
    print(f"control: a4 -> ..., a7=a6, e9=-1 (a4 unknown, legal={legal({7: ('eq', 6), 9: ('sat_r', M)}, rng, unknown=4)}): {np.round(w, 2)}")
    w, _ = edge_weights(BASES[0], rng, n=200, flip=4, unknown=5)
    print(f"a4 -> ... inside best a5 config (a4 context word, context re-realised): {np.round(w, 2)}")
    with open("stage2_results.json", "w") as f:
        json.dump(res, f)
