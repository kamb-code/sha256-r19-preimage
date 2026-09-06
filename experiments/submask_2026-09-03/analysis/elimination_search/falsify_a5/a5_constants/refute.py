#!/usr/bin/env python3
"""Refuter: re-measure the lowest-weight conditions found by edge_bulk.py with
(a) the bulk engine at a fresh seed and 1000 states, and (b) the independent
scalar implementation numcheck.py (extended with 'const' / 'sat_off'), which
re-realises the whole context from a definition dict through the symbolic
engine's condition format and flips all 32 bits of a5 on 40 planted-digest
states.  Any condition whose a5 -> C1 weight is below 4 bits in the bulk run is
flagged WEAK and must be confirmed here to count.

Usage: python3 refute.py [results_dir]
"""
import sys, os, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_bulk import measure
import numcheck as N

M = 0xFFFFFFFF


def to_defs(cond):
    """edge_bulk condition -> numcheck/symeng defs dict (None if not expressible)."""
    d = {}
    if cond.get('v') is not None: d[4] = ('const', cond['v'])
    a6 = cond.get('a6')
    if a6 == 'eq4': d[6] = ('eq', 4)
    elif a6 == 'neq4': d[6] = ('neq', 4)
    elif a6 is not None: d[6] = ('const', a6)
    a7 = cond.get('a7')
    if a7 == 'eq6': d[7] = ('eq', 6)
    elif a7 == 'neq6': d[7] = ('neq', 6)
    elif a7 is not None: d[7] = ('const', a7)
    if cond.get('e8') is not None:
        if cond.get('e8via', 'a8') == 'a8': d[8] = ('sat_r', cond['e8'])
        else: d[4] = ('sat_h', cond['e8'])
    if cond.get('c9') is not None: d[9] = ('sat_off', cond['c9'])
    if cond.get('e10') is not None: d[10] = ('sat_r', cond['e10'])
    if cond.get('e11') is not None: d[11] = ('sat_r', cond['e11'])
    return d


def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else 'results'
    rows = []
    for fn in glob.glob(os.path.join(rdir, '*.json')):
        if os.path.basename(fn) in ('summary.json', 'atoms.json', 'refute.json'): continue
        for r in json.load(open(fn)):
            if r.get('legal'): rows.append((os.path.basename(fn)[:-5], r))
    rows.sort(key=lambda x: x[1]['C1_mean'])
    print(f"{len(rows)} legal conditions loaded; global a5->C1 min {rows[0][1]['C1_mean']:.3f} "
          f"({rows[0][0]}: {rows[0][1]['cond']}), max {rows[-1][1]['C1_mean']:.3f}")
    weak = [x for x in rows if x[1]['C1_mean'] < 4.0]
    print(f"conditions below 4 bits (WEAK candidates): {len(weak)}")
    rng = np.random.default_rng(99)
    pick = rows[:20] + [rows[i] for i in rng.choice(len(rows), 20, replace=False)]
    out = []
    print(f"{'sweep':<24} {'bulk C1':>8} {'bulk2 C1':>9} {'numchk C1':>10} {'numchk C0':>10} {'C2':>6} {'C3':>6}  cond")
    for name, r in pick:
        cond = r['cond']
        r2 = measure(cond, 4242 + len(out), ns=1000)
        defs = to_defs(cond)
        try:
            _, _, ham = N.run(defs, 5, trials=40, seed=17, verbose=False, allbits=True)
            hs = [float(h) for h in ham]
        except RuntimeError as ex:
            hs = [None] * 4; print("   numcheck refused:", ex)
        out.append(dict(sweep=name, cond=cond, defs={str(k): v for k, v in defs.items()},
                        bulk_C1=r['C1_mean'], bulk2_C1=r2['C1_mean'], numcheck=hs))
        print(f"{name:<24} {r['C1_mean']:8.3f} {r2['C1_mean']:9.3f} {hs[1] if hs[1] is None else round(hs[1], 3)!s:>10} "
              f"{hs[0] if hs[0] is None else round(hs[0], 3)!s:>10} {hs[2] if hs[2] is None else round(hs[2], 2)!s:>6} "
              f"{hs[3] if hs[3] is None else round(hs[3], 2)!s:>6}  {cond}")
    confirmed_weak = [o for o in out if o['bulk2_C1'] < 4 and o['numcheck'][1] is not None and o['numcheck'][1] < 4]
    print(f"\nWEAK confirmed by refuter: {len(confirmed_weak)}")
    mins = [o['numcheck'][1] for o in out if o['numcheck'][1] is not None]
    print(f"refuter (numcheck) a5->C1 over the {len(out)} re-measured conditions: min {min(mins):.3f} max {max(mins):.3f}; "
          f"max |bulk - numcheck| = {max(abs(o['bulk_C1'] - o['numcheck'][1]) for o in out if o['numcheck'][1] is not None):.3f}")
    json.dump(dict(n_legal=len(rows), global_min=rows[0][1], n_weak=len(weak), rechecked=out,
                   confirmed_weak=confirmed_weak), open(os.path.join(rdir, 'refute.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
