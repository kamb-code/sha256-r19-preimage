#!/usr/bin/env python3
"""Per-constraint analysis over ALL legal configurations, for each promoted w:

 (a) forced homes: for each unknown u in {a1,a2,a3,w} and constraint C_j, is u
     ever deferrable (entries only lin/mild) or absent in C_j?  Is it ever
     absorbable?  (over all configs)
 (b) the persistent non-deferrable atoms of w in C0, C1, C2: the intersection
     over all configurations of the set of (function, argument-shape) atoms
     through which w enters non-deferrably.  These are the exact blocking
     dependencies: no equality / saturation removes them.
 (c) the chain argument: first constraint must be C0 (a1 is absorbable only
     there), so w must be deferrable in C0; then C1 (a2 only there) so w must
     be deferrable in C1; then C2; then C3 absorbs w.  Report at which step
     each w dies, in the BEST configuration for that step.
"""
import sys, itertools
from multiprocessing import Pool
import symdep_w as S

DEFER = S.DEFER

def atom_sig(k, c, info):
    if k == 'tab': return f'tab:{info[0]}({"+" if info[1] == 1 else "-"}u+c)'
    if k == 'tabmulti': return f'oneinput[{",".join(info[0])}]({info[1]})'
    if k == 'heavy': return f'heavy:{info[0]}:{info[-1]}'
    return f'{k}:{info[0]}' if info else k

def one(args):
    w, pairs_on, sat = args
    eqmap = S.closure(pairs_on, w)
    C, unk, legal, ef = S.build(w, eqmap, sat)
    if not legal: return None
    wn = f'a{w}'
    live_all = frozenset(unk - {'a0'})
    out = dict(eq=sorted(eqmap.items()), sat=sorted(sat.items()), strict=S.dof_legal(w, eqmap, sat))
    # (a) with all live
    tab = {}
    for t in range(4):
        for u in sorted(live_all):
            e = S.entries(C[t], u, live_all)
            ks = S.kinds(e)
            tab[(t, u)] = ('absent' if not ks else 'defer' if ks <= DEFER else
                           'absorb' if S.absorbable(e)[0] else 'block')
    out['tab'] = tab
    # (b) w's non-deferrable atoms, evaluated in the chain frame: at C0 live={a1,a2,a3,w},
    #     at C1 live={a2,a3,w}, at C2 live={a3,w}, at C3 live={w}
    lives = [live_all, live_all - {'a1'}, live_all - {'a1', 'a2'}, frozenset({wn})]
    nd = {}
    for t in range(4):
        e = S.entries(C[t], wn, lives[t])
        nd[t] = frozenset(atom_sig(k, c, i) for k, c, i in e if k not in DEFER)
        # also the a1/a2/a3 status at that step
    out['nd'] = nd
    out['w_defer_chain'] = [not nd[t] for t in range(3)]
    out['c3_absorb'] = S.absorbable(S.entries(C[3], wn, frozenset({wn})))
    # home forcing at each chain step
    out['a1_C0'] = tab[(0, 'a1')]
    return out

if __name__ == '__main__':
    ws = [int(x) for x in sys.argv[1:]] or list(range(4, 12))
    for w in ws:
        wn = f'a{w}'
        cfgs = list(S.configs(w))
        with Pool(24) as p:
            res = [r for r in p.imap_unordered(one, cfgs, chunksize=64) if r is not None]
        print(f"\n########## w = a{w}: {len(res)} legal configurations")
        # (a)
        unks = ['a1', 'a2', 'a3', wn]
        print("  (a) status of each unknown in each constraint, all four live; counts over configs "
              "[absent / defer / absorb / block]:")
        for t in range(4):
            row = []
            for u in unks:
                cnt = {k: 0 for k in ('absent', 'defer', 'absorb', 'block')}
                for r in res: cnt[r['tab'][(t, u)]] += 1
                row.append(f"{u}: {cnt['absent']}/{cnt['defer']}/{cnt['absorb']}/{cnt['block']}")
            print(f"     C{t}:  " + "   ".join(row))
        # (b) persistent atoms in the chain frame
        print("  (b) non-deferrable atoms of a%d that survive in EVERY configuration (chain frame):" % w)
        for t in range(3):
            inter = None
            for r in res:
                inter = r['nd'][t] if inter is None else inter & r['nd'][t]
            n_free = sum(1 for r in res if not r['nd'][t])
            n_free_strict = sum(1 for r in res if not r['nd'][t] and r['strict'])
            print(f"     C{t}: a{w} deferrable/absent in {n_free} configs ({n_free_strict} strict-DOF); "
                  f"persistent non-deferrable atoms: {sorted(inter) if inter else '{}'}")
            if n_free == 0:
                # show the smallest non-deferrable atom sets
                sets = {}
                for r in res: sets[r['nd'][t]] = sets.get(r['nd'][t], 0) + 1
                for s_, n in sorted(sets.items(), key=lambda x: (len(x[0]), -x[1]))[:3]:
                    print(f"          minimal set ({n} cfgs): {sorted(s_)}")
        n3 = sum(1 for r in res if r['c3_absorb'][0])
        print(f"     C3: a{w} absorbable in {n3} configs")
        # (c) chain
        best_depth = -1; best_cfg = None
        for r in res:
            d = 0
            for t in range(3):
                if r['nd'][t]: break
                d += 1
            else:
                d = 3 + (1 if r['c3_absorb'][0] else 0)
            if d > best_depth or (d == best_depth and best_cfg is not None and r['strict'] and not best_cfg['strict']):
                best_depth = d; best_cfg = r
        step = ['C0->a1 (w must be deferrable in C0)', 'C1->a2 (w deferrable in C1)',
                'C2->a3 (w deferrable in C2)', 'C3->w', 'COMPLETE'][best_depth]
        print(f"  (c) chain frame dies at step {best_depth}: {step}; best cfg eq={best_cfg['eq']} "
              f"sat={[(k, hex(v)) for k, v in best_cfg['sat']]} strict={best_cfg['strict']}")
        if best_depth < 3:
            print(f"       blocking atoms there: {sorted(best_cfg['nd'][best_depth])}")
        sys.stdout.flush()
