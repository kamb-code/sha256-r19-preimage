#!/usr/bin/env python3
"""Driver: for each promoted context word w in a4..a11

  1. enumerate every configuration (Maj-collapsing equalities x saturations of
     e8..e15), keep the legal ones (class-based legality, legal.py);
  2. forced homes: status of a1,a2,a3,w in each constraint over all configs;
  3. chain frame  sweep a0 -> C0:a1 -> C1:a2 -> C2:a3 -> C3:w : the
     non-deferrable atoms of w in C0, C1, C2 that persist in EVERY legal
     configuration, and the step at which the chain dies;
  4. full-width numeric confirmation on the best legal configuration:
     Hamming weight of dC_j per flipped bit of w (200 states), and exactness of
     the C3 form where C3 absorbs w (linear / global one-input / per-context).
"""
import sys, itertools, json
from multiprocessing import Pool
import numpy as np
import symdep_w as S
import legal as L
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, recover_W

R = 20
DEFER = S.DEFER

def atom_sig(k, c, info):
    if k == 'tab': return f'tab:{info[0]}({"+" if info[1] == 1 else "-"}u+c)'
    if k == 'tabmulti': return f'oneinput[{",".join(info[0])}]({info[1]})'
    if k == 'heavy': return f'heavy:{info[0]}:{info[-1]}'
    return f'{k}:{info[0]}' if info else k

def one(args):
    w, pairs_on, sat = args
    eqmap = S.closure(pairs_on, w)
    if eqmap is None: return None
    C, unk, ok, ef = S.build(w, eqmap, sat)
    if not ok: return None
    if not L.legal(w, eqmap, sat): return None
    wn = f'a{w}'
    live_all = frozenset(unk - {'a0'})
    out = dict(eq=tuple(sorted(eqmap.items())), sat=tuple(sorted(sat.items())), ncond=len(eqmap) + len(sat))
    tab = {}
    for t in range(4):
        for u in sorted(live_all):
            e = S.entries(C[t], u, live_all); ks = S.kinds(e)
            tab[(t, u)] = ('absent' if not ks else 'defer' if ks <= DEFER else
                           'absorb' if S.absorbable(e)[0] else 'block')
    out['tab'] = tab
    lives = [live_all, live_all - {'a1'}, live_all - {'a1', 'a2'}, frozenset({wn})]
    nd = {}; desc = {}
    for t in range(4):
        e = S.entries(C[t], wn, lives[t])
        nd[t] = frozenset(atom_sig(k, c, i) for k, c, i in e if k not in DEFER)
        desc[t] = S.describe(e)
    out['nd'] = nd; out['desc'] = desc
    e3 = S.entries(C[3], wn, frozenset({wn}))
    out['c3'] = S.absorbable(e3); out['c3kinds'] = tuple(sorted(S.kinds(e3)))
    return out

# ---------------- numerics ----------------
def full_state(ctx, unk):
    a = dict(ctx); a.update(unk)
    a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(0, R):
        e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M
    return a, e

def constraints(a, e):
    W = {r: recover_W(a, e, r) for r in range(0, R)}
    return [(W[16 + j] - s1(W[14 + j]) - W[9 + j] - s0(W[1 + j]) - W[j]) & M for j in range(4)]

def numeric(w, eqmap, sat, trials=200, seed=1):
    rng = np.random.default_rng(seed)
    pl = L.plan(w, eqmap, sat)
    hw = np.zeros((trials, 4)); nz = np.zeros(4, dtype=int)
    lin = 0; sep_unk = 0; sep_ctx = 0
    for t in range(trials):
        ctx = L.construct(rng, w, eqmap, sat, pl=pl)
        if ctx is None: return None
        unk = {i: int(rng.integers(0, 1 << 32, dtype=np.uint64)) for i in range(4)}
        a, e = full_state(ctx, unk); c = constraints(a, e)
        bit = int(rng.integers(0, 32))
        ctx2 = dict(ctx); ctx2[w] ^= (1 << bit)
        a2, e2 = full_state(ctx2, unk); c2 = constraints(a2, e2)
        for j in range(4):
            d = c[j] ^ c2[j]; hw[t, j] = bin(d).count('1'); nz[j] += (d != 0)
        # C3 form checks: linearity (dC3 == +-delta) and separability
        delta = int(rng.integers(1, 1 << 32, dtype=np.uint64))
        ctx3 = dict(ctx); ctx3[w] = (ctx3[w] + delta) & M
        a3, e3 = full_state(ctx3, unk); c3 = constraints(a3, e3)[3]
        d1 = (c3 - c[3]) & M
        lin += (d1 == delta) or (d1 == (-delta) & M)
        unk2 = {i: int(rng.integers(0, 1 << 32, dtype=np.uint64)) for i in range(4)}
        ab, eb = full_state(ctx, unk2); cb = constraints(ab, eb)[3]
        ab3, eb3 = full_state(ctx3, unk2); cb3 = constraints(ab3, eb3)[3]
        sep_unk += (((cb3 - cb) & M) != d1)
        ctxc = L.construct(rng, w, eqmap, sat, pl=pl); ctxc[w] = ctx[w]
        ctxc3 = dict(ctxc); ctxc3[w] = ctx3[w]
        ac, ec = full_state(ctxc, unk); cc = constraints(ac, ec)[3]
        ac3, ec3 = full_state(ctxc3, unk); cc3 = constraints(ac3, ec3)[3]
        sep_ctx += (((cc3 - cc) & M) != d1)
    return hw.mean(0), nz, lin, sep_unk, sep_ctx, trials

def show_numeric(w, r):
    eqmap = dict(r['eq']); sat = dict(r['sat'])
    print(f"      cfg eq={r['eq']} sat={[(k, hex(v)) for k, v in r['sat']]}")
    for t in range(4): print(f"        symbolic C{t} <- a{w}: {r['desc'][t]}")
    nm = numeric(w, eqmap, sat)
    if nm is None: print("        numeric: not constructible"); return
    hw, nz, lin, su, sc, n = nm
    print("        numeric dC_j per flipped bit of a%d (Hamming): " % w +
          "  ".join(f"C{j}={hw[j]:5.2f} (moved {nz[j]}/{n})" for j in range(4)))
    print(f"        C3 in a{w}: exactly linear (dC3 = +-delta) in {lin}/{n}; dC3 depends on a0..a3 in {su}/{n}; on the context in {sc}/{n}")

if __name__ == '__main__':
    ws = [int(x) for x in sys.argv[1:]] or list(range(4, 12))
    summary = {}
    for w in ws:
        wn = f'a{w}'
        cfgs = list(S.configs(w))
        with Pool(24) as p:
            res = [r for r in p.imap_unordered(one, cfgs, chunksize=64) if r is not None]
        print(f"\n########## w = a{w}: {len(cfgs)} configurations, {len(res)} legal (class-based DOF)")
        unks = ['a1', 'a2', 'a3', wn]
        print("  (a) status of each unknown in each constraint with all four live, counts over legal configs "
              "[absent/defer/absorb/block]:")
        for t in range(4):
            row = []
            for u in unks:
                cnt = {k: 0 for k in ('absent', 'defer', 'absorb', 'block')}
                for r in res: cnt[r['tab'][(t, u)]] += 1
                row.append(f"{u}: {cnt['absent']}/{cnt['defer']}/{cnt['absorb']}/{cnt['block']}")
            print(f"     C{t}:  " + "   ".join(row))
        print(f"  (b) chain frame (C0:a1, C1:a2, C2:a3, C3:a{w}); non-deferrable atoms of a{w}:")
        pers = {}
        for t in range(3):
            inter = None
            for r in res: inter = r['nd'][t] if inter is None else inter & r['nd'][t]
            n_free = sum(1 for r in res if not r['nd'][t])
            sets = {}
            for r in res: sets[r['nd'][t]] = sets.get(r['nd'][t], 0) + 1
            mins = sorted(sets.items(), key=lambda x: (len(x[0]), -x[1]))[:2]
            pers[t] = (n_free, sorted(inter) if inter else [], [(sorted(s_), n) for s_, n in mins])
            print(f"     C{t}: a{w} deferrable/absent in {n_free}/{len(res)} configs; persistent in all: {pers[t][1]}; "
                  f"minimal sets: {pers[t][2]}")
        n3 = sum(1 for r in res if r['c3'][0])
        forms = {}
        for r in res:
            if r['c3'][0]:
                forms[r['c3'][1] + ' :: ' + r['desc'][3]] = forms.get(r['c3'][1] + ' :: ' + r['desc'][3], 0) + 1
        print(f"     C3: a{w} absorbable in {n3}/{len(res)} configs; forms: ")
        for f, n in sorted(forms.items(), key=lambda x: -x[1])[:5]: print(f"          ({n}) {f}")
        # chain depth
        def depth(r):
            d = 0
            for t in range(3):
                if r['nd'][t]: return d
                d += 1
            return 3 + (1 if r['c3'][0] else 0)
        best = max(res, key=lambda r: (depth(r), -r['ncond']))
        dd = depth(best)
        step = ['C0->a1: a%d not deferrable in C0' % w, 'C1->a2: a%d not deferrable in C1' % w,
                'C2->a3: a%d not deferrable in C2' % w, 'C3 cannot absorb a%d' % w, 'COMPLETE'][dd]
        print(f"  (c) chain dies at step {dd}: {step}")
        print(f"      best configuration for the chain:")
        show_numeric(w, best)
        # numerics on the simplest config realising each interesting C3 form (fewest conditions)
        shown = set()
        for f, n in sorted(forms.items(), key=lambda x: -x[1])[:3]:
            cands = [r for r in res if r['c3'][0] and (r['c3'][1] + ' :: ' + r['desc'][3]) == f]
            r = min(cands, key=lambda r: r['ncond'])
            if (r['eq'], r['sat']) in shown or (r['eq'], r['sat']) == (best['eq'], best['sat']): continue
            shown.add((r['eq'], r['sat']))
            print(f"      C3 form {f}:")
            show_numeric(w, r)
        summary[w] = dict(n_cfg=len(cfgs), n_legal=len(res), chain_depth=dd, step=step,
                          persistent={t: pers[t][1] for t in range(3)}, minimal={t: pers[t][2] for t in range(3)},
                          n_c3=n3, forms=forms, best=dict(eq=best['eq'], sat=best['sat'], desc=best['desc']))
        sys.stdout.flush()
    json.dump(summary, open('run_all_summary.json', 'w'), indent=1, default=str)
