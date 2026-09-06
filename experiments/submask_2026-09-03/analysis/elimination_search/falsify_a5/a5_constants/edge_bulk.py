#!/usr/bin/env python3
"""Bulk edge-weight measurement for the a5 -> C1 edge at R = 20 under a widened
context-condition menu (arbitrary constants for the derived words and for the
a-words themselves).

Frame.  a5 is the unknown to be absorbed; a0..a3 are the other unknowns (held
fixed while a5 is perturbed); a4 = v, a6, a7, a8..a11 are context; a12..a19 are
digest words (uniform random = random digests, since the backward chain is a
bijection).  A condition is a dict of knobs:

    v   : None | int        a4 fixed to a constant
    a6  : None | int | 'eq4' | 'neq4'
    a7  : None | int | 'eq6' | 'neq6'
    e8  : None | int        e8  = a4 + a8  - T2(a7,a6,a5) = t, solved through a8
                            (context-only ONLY when a7 = a6; otherwise Maj(a7,a6,a5)
                            depends on a5 and the condition would reference the
                            unknown -> illegal, rejected by the legality test below)
    e8via: 'a8' | 'a4'      route: solve through a8 (default) or through a4
    c9  : None | int        c9  = a9 - T2(a8,a7,a6) = t, solved through a9
                            (e9 = a5 + c9; fixing e9 itself references a5)
    e10 : None | int        e10 = a6 + a10 - T2(a9,a8,a7) = t, through a10
    e11 : None | int        e11 = a7 + a11 - T2(a10,a9,a8) = t, through a11

Every condition is realised on NS random states; then every one of the 32 bits
of a5 is flipped (context held fixed: no condition references a5, and this is
verified numerically by recomputing the imposed quantities under the flip) and
the Hamming weight of the change of each lookup target
    T_j = W_{16+j} - s1(W_{14+j}) - W_{9+j} - W_j     (j = 0..3)
is recorded.  T_j differs from the actual table key (s0(W_{j+1}) - W_{j+1}) by a
word free of a5 and a4, so the differences are the same.  The a4 -> C0 edge is
measured the same way (flip a4, hold everything else) as a control.

All arithmetic is full-width uint32 on the real recovered-W formulas, no
symbolic shortcuts.
"""
from __future__ import annotations
import sys, os, json, time, itertools
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import M, K, IV, S0, S1, s0, s1, Ch, Maj, T2  # noqa: E402

U = np.uint32
R = 20
NS = 200            # random states per condition
KU = [U(k) for k in K]
IVA = {i: np.array([IV[-1 - i]], dtype=U) for i in (-1, -2, -3, -4)}   # a_{-1}=IV0 ... a_{-4}=IV3
IVE = {i: np.array([IV[3 - i]], dtype=U) for i in (-1, -2, -3, -4)}    # e_{-1}=IV4 ... e_{-4}=IV7
assert IVA[-1][0] == IV[0] and IVA[-4][0] == IV[3] and IVE[-1][0] == IV[4] and IVE[-4][0] == IV[7]


def u(x): return U(int(x) & M)


def rand32(rng, shape):
    return rng.integers(0, 1 << 32, size=shape, dtype=np.uint64).astype(U)


def targets(a):
    """a: dict r -> uint32 array (r in -4..19), all broadcastable.  Returns T[0..3]."""
    e = dict(IVE)
    for r in range(R):
        e[r] = a[r - 4] + a[r] - (S0(a[r - 1]) + Maj(a[r - 1], a[r - 2], a[r - 3]))
    W = {}
    for r in range(R):
        W[r] = (a[r] - (S0(a[r - 1]) + Maj(a[r - 1], a[r - 2], a[r - 3])) - e[r - 4]
                - S1(e[r - 1]) - Ch(e[r - 1], e[r - 2], e[r - 3]) - KU[r])
    T = [W[16 + j] - s1(W[14 + j]) - W[9 + j] - W[j] for j in range(4)]
    return T, e, W


# ---------------------------------------------------------------- realisation
def realise(cond, rng, ns=NS):
    """Random states satisfying `cond`.  Returns dict of arrays (shape (ns,)) for
    a_{-4}..a_{19}, plus the list of imposed quantities for the legality test."""
    a = dict(IVA)
    for i in range(4):
        a[i] = rand32(rng, ns)
    a[5] = rand32(rng, ns)
    for i in range(12, R):
        a[i] = rand32(rng, ns)
    v = cond.get('v')
    a[4] = np.full(ns, u(v)) if v is not None else rand32(rng, ns)
    a6 = cond.get('a6')
    if a6 is None: a[6] = rand32(rng, ns)
    elif a6 == 'eq4': a[6] = a[4].copy()
    elif a6 == 'neq4': a[6] = ~a[4]
    else: a[6] = np.full(ns, u(a6))
    a7 = cond.get('a7')
    if a7 is None: a[7] = rand32(rng, ns)
    elif a7 == 'eq6': a[7] = a[6].copy()
    elif a7 == 'neq6': a[7] = ~a[6]
    else: a[7] = np.full(ns, u(a7))
    a[8] = rand32(rng, ns)
    a[9] = rand32(rng, ns)
    a[10] = rand32(rng, ns)
    a[11] = rand32(rng, ns)
    e8 = cond.get('e8')
    if e8 is not None:
        if cond.get('e8via', 'a8') == 'a8':
            a[8] = u(e8) - a[4] + S0(a[7]) + Maj(a[7], a[6], a[5])
        else:   # solve through a4 (only meaningful when v is free)
            a[4] = u(e8) - a[8] + S0(a[7]) + Maj(a[7], a[6], a[5])
            if a6 == 'eq4': a[6] = a[4].copy()
            if a6 == 'neq4': a[6] = ~a[4]
    c9 = cond.get('c9')
    if c9 is not None:
        a[9] = u(c9) + S0(a[8]) + Maj(a[8], a[7], a[6])
    e10 = cond.get('e10')
    if e10 is not None:
        a[10] = u(e10) - a[6] + S0(a[9]) + Maj(a[9], a[8], a[7])
    e11 = cond.get('e11')
    if e11 is not None:
        a[11] = u(e11) - a[7] + S0(a[10]) + Maj(a[10], a[9], a[8])
    return a


def imposed_ok(cond, a, e):
    """Legality: every imposed quantity must hold in the (possibly a5-flipped) state
    with the context words held fixed.  Returns True iff all hold everywhere."""
    ok = True
    if cond.get('e8') is not None: ok &= bool(np.all(e[8] == u(cond['e8'])))
    if cond.get('c9') is not None: ok &= bool(np.all(a[9] - (S0(a[8]) + Maj(a[8], a[7], a[6])) == u(cond['c9'])))
    if cond.get('e10') is not None: ok &= bool(np.all(e[10] == u(cond['e10'])))
    if cond.get('e11') is not None: ok &= bool(np.all(e[11] == u(cond['e11'])))
    if cond.get('a6') == 'eq4': ok &= bool(np.all(a[6] == a[4]))
    if cond.get('a6') == 'neq4': ok &= bool(np.all(a[6] == ~a[4]))
    if cond.get('a7') == 'eq6': ok &= bool(np.all(a[7] == a[6]))
    if cond.get('a7') == 'neq6': ok &= bool(np.all(a[7] == ~a[6]))
    return ok


BITS = (U(1) << np.arange(32, dtype=U))


def measure(cond, seed, ns=NS, control=True):
    """Returns dict: legal flag, per-target mean/min/max weight of a5 flips,
    a4->C0 control mean, and moved fractions."""
    rng = np.random.default_rng(seed)
    a = realise(cond, rng, ns)
    T0, e0, _ = targets(a)
    # flipped copies: shape (ns, 32)
    af = {r: a[r][:, None] for r in a}
    af[5] = a[5][:, None] ^ BITS[None, :]
    Tf, ef, _ = targets(af)
    legal = imposed_ok(cond, af, ef) and imposed_ok(cond, a, e0)
    out = dict(legal=legal)
    for j in range(4):
        d = np.bitwise_count(Tf[j] ^ T0[j][:, None])
        out[f'C{j}_mean'] = float(d.mean())
        out[f'C{j}_min'] = int(d.min())
        out[f'C{j}_max'] = int(d.max())
        out[f'C{j}_moved'] = float((d > 0).mean())
        out[f'C{j}_perbit'] = d.mean(axis=0).round(2).tolist() if j == 1 else None
    if control:
        ag = {r: a[r][:, None] for r in a}
        ag[4] = a[4][:, None] ^ BITS[None, :]
        Tg, _, _ = targets(ag)
        d = np.bitwise_count(Tg[0] ^ T0[0][:, None])
        out['a4C0_mean'] = float(d.mean())
        out['a4C0_min'] = int(d.min())
        # a4 -> C0 with a8 re-solved to keep e8 fixed (the other reading of the control)
        if cond.get('e8') is not None and cond.get('e8via', 'a8') == 'a8':
            ag[8] = u(cond['e8']) - ag[4] + S0(ag[7]) + Maj(ag[7], ag[6], ag[5])
            if cond.get('c9') is not None: ag[9] = u(cond['c9']) + S0(ag[8]) + Maj(ag[8], ag[7], ag[6])
            if cond.get('e10') is not None: ag[10] = u(cond['e10']) - ag[6] + S0(ag[9]) + Maj(ag[9], ag[8], ag[7])
            if cond.get('e11') is not None: ag[11] = u(cond['e11']) - ag[7] + S0(ag[10]) + Maj(ag[10], ag[9], ag[8])
            Tg2, _, _ = targets(ag)
            d2 = np.bitwise_count(Tg2[0] ^ T0[0][:, None])
            out['a4C0_resolved_mean'] = float(d2.mean())
    return out


# ---------------------------------------------------------------- menus
def structured_constants():
    S = [0, M, 0x55555555, 0xAAAAAAAA, 0x33333333, 0xCCCCCCCC, 0x0F0F0F0F, 0xF0F0F0F0,
         0x00FF00FF, 0xFF00FF00, 0x0000FFFF, 0xFFFF0000, 0x80000000, 0x00000001,
         0x7FFFFFFF, 0xFFFFFFFE, 0x01010101, 0x80808080, 0x11111111, 0x88888888,
         0x00010001, 0x10001000, 0x0000FFFF, 0x0F0F0000, 0x6a09e667, 0x428a2f98]
    single = [1 << i for i in range(32)] + [(M ^ (1 << i)) for i in range(32)]
    rot4 = [0x11111111 * m for m in range(1, 15)]                 # period-4 (ROTR-4 invariant)
    rot8 = [0x01010101 * m for m in range(1, 255)]                # period-8 (ROTR-8 invariant)
    rot2 = [0x55555555, 0xAAAAAAAA]
    return dict(base=[0, M], struct=S, single=single, rot2=rot2, rot4=rot4, rot8=rot8)


def full_menu(rng, n_random=2000, n_rot16=200):
    m = structured_constants()
    rot16 = [0x00010001 * int(x) for x in rng.integers(1, 65535, n_rot16)]
    rnd = [int(x) for x in rng.integers(0, 1 << 32, n_random, dtype=np.uint64)]
    allv = []
    seen = set()
    for grp in ('base', 'struct', 'single', 'rot2', 'rot4', 'rot8'):
        for x in m[grp]:
            x &= M
            if x not in seen: seen.add(x); allv.append((grp, x))
    for x in rot16:
        if x not in seen: seen.add(x); allv.append(('rot16', x))
    for x in rnd:
        if x not in seen: seen.add(x); allv.append(('random', x))
    return allv


# ---------------------------------------------------------------- driver
def worker(args):
    idx, cond, seed = args
    r = measure(cond, seed)
    r['idx'] = idx
    r['cond'] = cond
    return r


def run_sweep(name, conds, seed0, procs=24, chunks=16):
    from multiprocessing import Pool
    t0 = time.time()
    jobs = [(i, c, seed0 + i) for i, c in enumerate(conds)]
    with Pool(procs) as p:
        res = list(p.imap(worker, jobs, chunksize=chunks))
    el = time.time() - t0
    legal = [r for r in res if r['legal']]
    ill = len(res) - len(legal)
    c1 = np.array([r['C1_mean'] for r in legal]) if legal else np.array([])
    print(f"[{name}] {len(conds)} conditions, {len(legal)} legal, {ill} illegal (reference a5); {el:.0f}s")
    if legal:
        best = min(legal, key=lambda r: r['C1_mean'])
        print(f"   a5->C1 weight: min {c1.min():.3f}  mean {c1.mean():.3f}  max {c1.max():.3f}   "
              f"(#below 4 bits: {(c1 < 4).sum()}, below 2: {(c1 < 2).sum()})")
        print(f"   best: {best['cond']}  C0 {best['C0_mean']:.2f} C1 {best['C1_mean']:.2f} "
              f"C2 {best['C2_mean']:.2f} C3 {best['C3_mean']:.2f}  a4->C0 {best['a4C0_mean']:.2f}")
        for j in (0, 2, 3):
            cj = np.array([r[f'C{j}_mean'] for r in legal])
            print(f"   a5->C{j}: min {cj.min():.3f} mean {cj.mean():.3f} max {cj.max():.3f}")
        ctl = np.array([r['a4C0_mean'] for r in legal])
        print(f"   control a4->C0: min {ctl.min():.3f} mean {ctl.mean():.3f} max {ctl.max():.3f}")
    sys.stdout.flush()
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    ap.add_argument('--procs', type=int, default=24)
    ap.add_argument('--sweeps', default='all')
    ap.add_argument('--nrand', type=int, default=2000)
    ap.add_argument('--pairn', type=int, default=400, help='random values per word in the e8 x c9 product')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(20260906)
    menu = full_menu(rng, args.nrand)
    print(f"menu size per word: {len(menu)}  (groups: "
          + ", ".join(f"{g}={sum(1 for gg, _ in menu if gg == g)}" for g in ('base', 'struct', 'single', 'rot2', 'rot4', 'rot8', 'rot16', 'random')) + ")")
    D0 = dict(a7='eq6', e8=M, e11=0)      # the closest-miss frame of FINDINGS.md
    want = args.sweeps.split(',') if args.sweeps != 'all' else None
    summary = {}

    def do(name, conds, seed):
        if want is not None and name not in want: return
        res = run_sweep(name, conds, seed, procs=args.procs)
        with open(os.path.join(args.out, f'{name}.json'), 'w') as f:
            json.dump(res, f)
        legal = [r for r in res if r['legal']]
        if legal:
            best = min(legal, key=lambda r: r['C1_mean'])
            summary[name] = dict(n=len(conds), legal=len(legal),
                                 C1_min=min(r['C1_mean'] for r in legal),
                                 C1_mean=float(np.mean([r['C1_mean'] for r in legal])),
                                 C1_max=max(r['C1_mean'] for r in legal),
                                 C0_min=min(r['C0_mean'] for r in legal),
                                 a4C0_mean=float(np.mean([r['a4C0_mean'] for r in legal])),
                                 best=best['cond'], best_C1=best['C1_mean'])
        with open(os.path.join(args.out, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=1)

    # 0. structural grid: a6, a7 relations x e8, c9, e10, e11 in {free, 0, -1}, both e8 routes
    conds = []
    for a6 in (None, 'eq4', 'neq4'):
        for a7 in (None, 'eq6', 'neq6'):
            for e8 in (None, 0, M):
                for c9 in (None, 0, M):
                    for e10 in (None, 0, M):
                        for e11 in (None, 0, M):
                            c = dict(a6=a6, a7=a7, e8=e8, c9=c9, e10=e10, e11=e11)
                            c = {k: x for k, x in c.items() if x is not None}
                            conds.append(c)
                            if e8 is not None:
                                conds.append(dict(c, e8via='a4'))
    do('grid01', conds, 1_000_000)

    # 1. single-word sweeps over the full menu, others at D0
    for word in ('e8', 'c9', 'e10', 'e11'):
        conds = [dict(D0, **{word: x}) for _, x in menu]
        do(f'sweep_{word}', conds, 2_000_000 + 100_000 * ord(word[-1]))
    # a-words to constants (a7 = a6 = const; a6 const with a7 free; a7 const with a6 free; v const)
    do('sweep_a6a7const', [dict(D0, a6=x) for _, x in menu], 3_000_000)          # a7 = a6 = const
    do('sweep_a6const_a7free', [dict(a6=x, e11=0) for _, x in menu], 3_100_000)   # e8 free (a7 != a6)
    do('sweep_a7const_a6free', [dict(a7=x, e11=0) for _, x in menu], 3_200_000)
    do('sweep_vconst', [dict(D0, v=x) for _, x in menu], 3_300_000)
    do('sweep_vconst_a6eq4', [dict(D0, v=x, a6='eq4') for _, x in menu], 3_400_000)

    # 2. all 16 combinations of {0,-1} for (e8, c9, e10, e11) + 26^4 structured grid
    S = structured_constants()['struct']
    conds = [dict(a7='eq6', e8=w, c9=x, e10=y, e11=z) for w, x, y, z in itertools.product([0, M], repeat=4)]
    do('combo16', conds, 4_000_000)
    conds = [dict(a7='eq6', e8=w, c9=x, e10=y, e11=z) for w, x, y, z in itertools.product(S, repeat=4)]
    do('struct4', conds, 4_100_000)

    # 3. pair products over the two words that carry a5 into C1: e8 x c9
    small = [x for g, x in menu if g in ('base', 'struct', 'single', 'rot2', 'rot4')]
    conds = [dict(a7='eq6', e8=w, c9=x, e11=0) for w in small for x in small]
    do('pair_e8c9_small', conds, 5_000_000)
    conds = [dict(a7='eq6', e8=w, c9=x, e11=0, v=y) for w in small[:26] for x in small[:26] for y in small[:26]]
    do('triple_e8c9v_struct', conds, 5_100_000)

    # 4. random joint samples
    rr = lambda n: [int(x) for x in rng.integers(0, 1 << 32, n, dtype=np.uint64)]
    n = args.nrand
    conds = [dict(a7='eq6', e8=w, c9=x, e10=y, e11=z) for w, x, y, z in zip(rr(n), rr(n), rr(n), rr(n))]
    do('rand4_e8c9e10e11', conds, 6_000_000)
    conds = [dict(v=p, a6=q, a7='eq6', e8=w, c9=x, e10=y, e11=z)
             for p, q, w, x, y, z in zip(rr(n), rr(n), rr(n), rr(n), rr(n), rr(n))]
    do('rand6_allconst', conds, 6_100_000)
    conds = [dict(v=p, a6=q, a7=s, c9=x, e10=y, e11=z)
             for p, q, s, x, y, z in zip(rr(n), rr(n), rr(n), rr(n), rr(n), rr(n))]
    do('rand6_a7free_const', conds, 6_200_000)
    conds = [dict(a6=q, a7=s, e10=y, e11=z) for q, s, y, z in zip(rr(n), rr(n), rr(n), rr(n))]
    do('rand4_a6a7e10e11', conds, 6_300_000)

    # 4b. big single-word random sweeps of the two words that carry a5 into C1, and
    #     random pair products c9 x a6const and c9 x v (the bitwise partners of a5)
    do('sweep_c9_big', [dict(D0, c9=x) for x in rr(100_000)], 6_400_000)
    do('sweep_e8_big', [dict(D0, e8=x) for x in rr(100_000)], 6_500_000)
    do('pair_c9_a6_random', [dict(D0, c9=x, a6=y) for x in rr(300) for y in rr(300)], 6_600_000)
    do('pair_c9_v_random', [dict(D0, c9=x, v=y) for x in rr(300) for y in rr(300)], 6_700_000)
    do('pair_c9_v_a6eq4_random', [dict(D0, c9=x, v=y, a6='eq4') for x in rr(300) for y in rr(300)], 6_800_000)

    # 5. the full random product e8 x c9 (the only two words that carry a5 into C1)
    if want is None or 'pair_e8c9_random' in want:
        rw = [x for g, x in menu if g == 'random'][:args.pairn]
        conds = [dict(a7='eq6', e8=w, c9=x, e11=0) for w in rw for x in rw]
        do('pair_e8c9_random', conds, 7_000_000)


if __name__ == '__main__':
    main()
