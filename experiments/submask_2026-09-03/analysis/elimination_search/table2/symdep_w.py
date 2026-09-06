#!/usr/bin/env python3
"""Second-table search at R=20.

Promote ONE context word w in {a4..a11} to a fifth unknown (with a0..a3), and
ask whether the fourth constraint C3 can absorb it through a one-input table
while every other place w enters (C0, C1, C2) is cut by context conditions
(equalities among context words, which collapse Maj; saturations of e8..e15 to
0 / 0xFFFFFFFF, which collapse Ch and kill Sigma1(e_r)).

Symbolic algebra as in L5_50_symdep.py (canonical sums of atoms).  New here:

  * W16..W19 are EXPANDED through the recovered-W formula (a12..a19 are digest
    symbols), so w in {a8..a11} is tracked into the schedule words.
  * Entry taxonomy of an unknown u in a constraint, RELATIVE to the set `live`
    of unknowns not yet known at that point (the swept a0 is known):
        lin      : top-level +-u
        tab      : inside exactly one one-input map F(+-u + c) (F in
                   S0,S1,s0,s1), c free of live unknowns.  Solvable by F^-1
                   (all four are bijections) or, with a linear term, by the
                   GLOBAL table x -> F(x) +- x.
        tabmulti : inside several one-input maps F_i(+-u + c_i), each c_i free
                   of live unknowns: a one-input map of u whose shape depends on
                   the differences c_i - c_j.  Table is PER-CONTEXT if those
                   differences carry no unknown at all (a0..a3 included),
                   otherwise PER-CANDIDATE, i.e. a 2^32 inversion each time.
        mild     : u linear inside an argument of Maj/Ch, other arguments free
                   of live unknowns.  Bitwise-affine; deferrable at a bitwise
                   residual (about 13 bits per term).
        mildmix  : as mild, other arguments contain live unknowns.
        mild2    : u inside Maj/Ch nested (through sums) inside Maj/Ch.
        heavy    : u inside a one-input map together with another live
                   unknown, or nonlinearly (Maj/Ch of u inside F, or F of u
                   inside Maj/Ch).  Not saturable, not deferrable below 32 bits.
  * Scheduler: sweep a0; any order of C0..C3; each constraint absorbs one
    unknown whose entries are lin + (tab | tabmulti-per-context) [own mild
    entries may be self-deferred]; every other live unknown must enter that
    constraint through {none, lin, mild, mildmix, mild2} (deferred); a
    tab/tabmulti/heavy entry of another live unknown BLOCKS.

Enumerates, per w, every combination of the Maj-collapsing equalities among
context words and every saturation pattern of e8..e15 in {-, 0, -1}.
"""
import itertools, sys, json, time
from multiprocessing import Pool

M32 = 0xFFFFFFFF
R = 20
UNK_ALL = frozenset()
ONEIN = ('S0', 'S1', 's0', 's1')

# ---------- expression algebra ----------
def sym(n): return frozenset({(('sym', n), 1)})
def lit(v): return frozenset({(('lit', v & M32), 1)})
ZERO = frozenset()

def add(*es):
    d = {}
    for e in es:
        for a, c in e:
            d[a] = d.get(a, 0) + c
    return frozenset((a, c % (1 << 32)) for a, c in d.items() if c % (1 << 32))

def neg(e): return frozenset((a, (-c) % (1 << 32)) for a, c in e)
def is_lit(e): return all(at[0] == 'lit' for at, c in e)
def lit_val(e):
    s = 0
    for at, c in e: s += c * at[1]
    return s & M32
def op(name, *args): return frozenset({((name,) + tuple(args), 1)})
def S0(e): return op('S0', e)
def S1(e): return op('S1', e)
def s0(e): return op('s0', e)
def s1(e): return op('s1', e)
def Maj(a, b, c):
    if a == b or a == c: return a
    if b == c: return b
    return op('Maj', a, b, c)
def Ch(a, b, c):
    if is_lit(a):
        v = lit_val(a)
        if v == M32: return b
        if v == 0: return c
    if b == c: return b
    return op('Ch', a, b, c)
def T2(a, b, c): return add(S0(a), Maj(a, b, c))

_symcache = {}
def symbols(e):
    r = _symcache.get(e)
    if r is not None: return r
    acc = set()
    for at, c in e:
        if at[0] == 'sym': acc.add(at[1])
        elif at[0] == 'lit': pass
        else:
            for arg in at[1:]: acc |= symbols(arg)
    acc = frozenset(acc)
    _symcache[e] = acc
    return acc

def strip(e, u):
    """e with the top-level symbol u removed (its shift c)."""
    return frozenset((a, c) for a, c in e if a != ('sym', u))

# ---------- entry taxonomy ----------
def entries(expr, u, live, depth=0):
    out = []
    others = live - {u}
    for at, c in expr:
        if at[0] == 'sym':
            if at[1] == u: out.append(('lin', c, None))
        elif at[0] == 'lit':
            continue
        elif at[0] in ONEIN:
            arg = at[1]
            syms = symbols(arg)
            if u not in syms: continue
            inner = entries(arg, u, live, depth + 1)
            only_lin = len(inner) == 1 and inner[0][0] == 'lin'
            if only_lin and not (syms & others):
                out.append(('tab', c, (at[0], inner[0][1], strip(arg, u))))
            else:
                out.append(('heavy', c, (at[0], 'with-live-unknowns' if syms & others else 'nonlinear-in-u')))
        elif at[0] in ('Maj', 'Ch'):
            args = at[1:]
            for i, arg in enumerate(args):
                syms = symbols(arg)
                if u not in syms: continue
                inner = entries(arg, u, live, depth + 1)
                ks = {k for k, _, _ in inner}
                if ks & {'tab', 'heavy', 'tabmulti'}:
                    out.append(('heavy', c, (at[0], i, 'F(u)-inside')))
                    continue
                other_syms = set()
                for j, a2 in enumerate(args):
                    if j != i: other_syms |= symbols(a2)
                if ks != {'lin'}:
                    out.append(('mild2', c, (at[0], i)))
                elif other_syms & live:
                    out.append(('mildmix', c, (at[0], i, tuple(sorted(other_syms & live)))))
                else:
                    out.append(('mild', c, (at[0], i)))
        else:
            raise ValueError(at[0])
    # merge several tab atoms into tabmulti
    tabs = [x for x in out if x[0] == 'tab']
    if len(tabs) > 1:
        rest = [x for x in out if x[0] != 'tab']
        # relative shifts: c_i - c_0 (with the sign of u normalised)
        base = tabs[0][2]
        diffs = set()
        for F, coef, cshift in (t[2] for t in tabs[1:]):
            # normalise: if u enters with -1, negate the shift so it reads (u + c')
            def norm(coef_, sh):
                return sh if coef_ == 1 else neg(sh)
            diffs |= symbols(add(norm(coef, cshift), neg(norm(base[1], base[2]))))
        scope = 'ctx' if not (diffs & UNK_ALL) else 'cand'
        out = rest + [('tabmulti', 0, (tuple(t[2][0] for t in tabs), scope))]
    return out

def kinds(ents): return {k for k, _, _ in ents}

# ---------- the compression function, symbolically ----------
def build(w, eqmap, sat):
    global UNK_ALL
    unk = {'a0', 'a1', 'a2', 'a3'} | ({f'a{w}'} if w is not None else set())
    UNK_ALL = frozenset(unk)
    a = {}
    for i in (-4, -3, -2, -1): a[i] = sym(f'IV{i}')
    for i in range(0, 4): a[i] = sym(f'a{i}')
    for i in range(4, 12):
        rep = eqmap.get(i, i)
        a[i] = sym(f'A{rep}') if rep >= 12 else sym(f'a{rep}')
    for i in range(12, R): a[i] = sym(f'A{i}')
    e = {}
    for i in (-4, -3, -2, -1): e[i] = sym(f'IVe{i}')
    for r in range(0, R):
        e[r] = add(a[r - 4], a[r], neg(T2(a[r - 1], a[r - 2], a[r - 3])))
    e_free = {r: not (symbols(e[r]) & unk) for r in range(8, 16)}
    for r, tgt in sat.items():
        if not e_free[r]:
            return None, unk, False, e_free
        e[r] = lit(tgt)
    W = {}
    for r in range(0, R):
        W[r] = add(a[r], neg(T2(a[r - 1], a[r - 2], a[r - 3])), neg(e[r - 4]),
                   neg(S1(e[r - 1])), neg(Ch(e[r - 1], e[r - 2], e[r - 3])),
                   neg(sym(f'K{r}')))
    C = [add(W[16 + t], neg(s1(W[14 + t])), neg(W[9 + t]), neg(s0(W[1 + t])), neg(W[t]))
         for t in range(4)]
    return C, unk, True, e_free

def ctx_words(w): return [i for i in range(4, 12) if i != w]

def maj_pairs(w):
    unk = {0, 1, 2, 3} | ({w} if w is not None else set())
    pairs = set()
    for r in range(0, 16):
        tri = (r - 1, r - 2, r - 3)
        if not any(t in unk for t in tri): continue
        ctx = [t for t in tri if 4 <= t <= 11 and t != w]
        for x, y in itertools.combinations(ctx, 2):
            pairs.add((min(x, y), max(x, y)))
        dig = [t for t in tri if 12 <= t <= 19]
        for x in ctx:
            for y in dig:
                pairs.add((x, y))
    return sorted(pairs)

def closure(pairs_on, w):
    parent = {i: i for i in ctx_words(w)}
    for x, y in pairs_on:
        for z in (x, y):
            if z >= 12: parent[z] = z
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for x, y in pairs_on:
        rx, ry = find(x), find(y)
        if rx == ry: continue
        # a digest word (>= 12) is always the representative; two different digest words cannot be equated
        if rx >= 12 and ry >= 12: return None
        if rx >= 12 or ry >= 12: parent[min(rx, ry)] = max(rx, ry)
        else: parent[max(rx, ry)] = min(rx, ry)
    return {i: find(i) for i in ctx_words(w) if find(i) != i}

def dof_legal(w, eqmap, sat):
    consumed = set(eqmap.keys())
    free = [i for i in ctx_words(w) if i not in consumed]
    sats = sorted(sat.keys())
    cand = {r: [x for x in (r - 4, r) if x in free] for r in sats}
    match = {}
    def try_assign(r, seen):
        for x in cand[r]:
            if x in seen: continue
            seen.add(x)
            if x not in match or try_assign(match[x], seen):
                match[x] = r; return True
        return False
    for r in sats:
        if not try_assign(r, set()): return False
    return True

# ---------- scheduling ----------
DEFER = {'lin', 'mild', 'mildmix', 'mild2'}

def absorbable(ents):
    ks = kinds(ents)
    if not ents: return False, 'absent'
    core = {k for k in ks if k in ('lin', 'tab', 'tabmulti')}
    selfdef = ks - core
    if 'heavy' in ks: return False, 'heavy'
    if not core: return False, 'only-bitwise'
    tabs = [x for x in ents if x[0] == 'tab']
    tm = [x for x in ents if x[0] == 'tabmulti']
    if tm:
        if tm[0][2][1] == 'cand': return False, 'tabmulti-per-candidate'
        lab = 'tabmulti-per-context(' + ','.join(tm[0][2][0]) + ')'
    elif tabs:
        lab = f'tab:{tabs[0][2][0]}' + ('+lin' if 'lin' in ks else '')
    else:
        lab = 'lin'
    if selfdef: lab += ' selfdefer:' + ','.join(sorted(selfdef))
    return True, lab

def schedule(C, unk):
    unk_list = sorted(unk - {'a0'})
    n = len(unk_list)
    cache = {}
    def ent(t, u, live):
        k = (t, u, live)
        if k not in cache: cache[k] = entries(C[t], u, live)
        return cache[k]
    results = []
    for perm in itertools.permutations(range(4), n):
        for homes in itertools.permutations(unk_list, n):
            live = frozenset(unk_list)
            plan = []; ok_all = True; deferred = []; steps = 0
            for t, u in zip(perm, homes):
                eu = ent(t, u, live)
                ok, lab = absorbable(eu)
                if not ok:
                    ok_all = False; plan.append((f'C{t}', u, 'NOT-ABSORBABLE:' + lab)); break
                blockers = []
                for v in unk_list:
                    if v == u or v not in live: continue
                    ks = kinds(ent(t, v, live))
                    if not ks: continue
                    if ks <= DEFER:
                        deferred.append((f'C{t}', v, tuple(sorted(ks))))
                    else:
                        blockers.append((v, tuple(sorted(ks))))
                if blockers:
                    ok_all = False; plan.append((f'C{t}', u, lab, 'BLOCKED-BY', blockers)); break
                plan.append((f'C{t}', u, lab)); live = live - {u}; steps += 1
            results.append((steps, ok_all, plan, deferred))
    results.sort(key=lambda x: (-x[0], not x[1]))
    return results, ent

def describe(ents):
    if not ents: return 'none'
    parts = []
    for k, c, info in ents:
        cc = c if c < (1 << 31) else c - (1 << 32)
        if k == 'lin': parts.append(f'{cc:+d}*u')
        elif k == 'tab': parts.append(f'{cc:+d}*{info[0]}({"+" if info[1] == 1 else "-"}u+c)')
        elif k == 'tabmulti': parts.append(f'ONEINPUT[{",".join(info[0])}]({info[1]})')
        elif k in ('mild', 'mild2'): parts.append(f'{cc:+d}*{info[0]}[arg{info[1]}:u]{"(nested)" if k == "mild2" else ""}')
        elif k == 'mildmix': parts.append(f'{cc:+d}*{info[0]}[arg{info[1]}:u;with {",".join(info[2])}]')
        else: parts.append(f'{cc:+d}*HEAVY:{info[0]}:{info[-1]}')
    return ' '.join(parts)

def one_config(args):
    w, pairs_on, sat = args
    eqmap = closure(pairs_on, w)
    if eqmap is None: return None
    C, unk, legal, e_free = build(w, eqmap, sat)
    if not legal: return None
    strict = dof_legal(w, eqmap, sat)
    results, ent = schedule(C, unk)
    best = results[0]
    wname = f'a{w}'
    live_all = frozenset(unk - {'a0'})
    went = {t: entries(C[t], wname, live_all) for t in range(4)}
    # w's entries with a1,a2,a3 already known (the C3-last frame): what remains of w in C0..C2
    went_alone = {t: entries(C[t], wname, frozenset({wname})) for t in range(4)}
    return dict(w=w, eq=sorted(eqmap.items()), sat=sorted(sat.items()), strict=strict,
                steps=best[0], complete=best[1], plan=best[2], deferred=best[3],
                w_entries={t: describe(went[t]) for t in range(4)},
                w_alone={t: describe(went_alone[t]) for t in range(4)},
                w_kinds={t: sorted(kinds(went_alone[t])) for t in range(4)},
                c3_absorb=absorbable(went_alone[3]))

def configs(w):
    pairs = maj_pairs(w)
    sat_vals = (None, 0, M32)
    for k in range(len(pairs) + 1):
        for on in itertools.combinations(pairs, k):
            for sv in itertools.product(sat_vals, repeat=8):
                sat = {8 + i: v for i, v in enumerate(sv) if v is not None}
                yield (w, on, sat)

def nblock(r):
    return sum(1 for t in range(3) for k in r['w_kinds'][t] if k in ('heavy', 'tab', 'tabmulti'))

if __name__ == '__main__':
    ws = [int(x) for x in sys.argv[1:]] or list(range(4, 12))
    out_all = {}
    for w in ws:
        t0 = time.time()
        pairs = maj_pairs(w)
        cfgs = list(configs(w))
        with Pool(24) as p:
            res = [r for r in p.imap_unordered(one_config, cfgs, chunksize=64) if r is not None]
        n_strict = sum(r['strict'] for r in res)
        c3ok = [r for r in res if r['c3_absorb'][0]]
        complete = [r for r in res if r['complete']]
        res.sort(key=lambda r: (nblock(r), -r['steps']))
        print(f"\n===== w = a{w}: {len(cfgs)} configurations ({len(pairs)} Maj-pairs {pairs}), "
              f"{len(res)} legal (saturations realisable), {n_strict} strictly legal (DOF); {time.time()-t0:.0f}s")
        print(f"  C3 can absorb a{w} (lin / one-input table, a0..a3 known) in {len(c3ok)} configurations; "
              f"complete 5-unknown schedules (any order, any homes): {len(complete)}")
        forms = {}
        for r in c3ok:
            key = r['c3_absorb'][1] + ' :: ' + r['w_alone'][3]
            forms[key] = forms.get(key, 0) + 1
        for f, n in sorted(forms.items(), key=lambda x: -x[1])[:6]:
            print(f"    C3 form ({n} cfgs): {f}")
        print(f"  minimum number of non-deferrable (tab/heavy) entries of a{w} into C0..C2 "
              f"(a1,a2,a3 known) over all configs: {nblock(res[0])}")
        shown = 0; seen = set()
        for r in res:
            key = tuple(tuple(r['w_kinds'][t]) for t in range(4))
            if key in seen: continue
            seen.add(key); shown += 1
            if shown > 5: break
            print(f"   cfg eq={r['eq']} sat={[(k, hex(v)) for k, v in r['sat']]} strictDOF={r['strict']}")
            for t in range(4):
                print(f"      C{t} <- a{w} (a0..a3 known): {r['w_alone'][t]}")
            print(f"      best schedule ({r['steps']} steps, complete={r['complete']}): {r['plan']}")
            if r['deferred']: print(f"      deferred: {r['deferred']}")
        for r in complete[:5]:
            print(f"   COMPLETE: eq={r['eq']} sat={[(k, hex(v)) for k, v in r['sat']]} strict={r['strict']} plan={r['plan']} deferred={r['deferred']}")
        out_all[w] = dict(n_cfg=len(cfgs), n_legal=len(res), n_strict=n_strict, n_c3=len(c3ok),
                          n_complete=len(complete), min_block=nblock(res[0]),
                          c3_forms=forms,
                          best=[dict(eq=r['eq'], sat=r['sat'], strict=r['strict'], entries=r['w_alone'],
                                     plan=[str(p) for p in r['plan']], deferred=[str(d) for d in r['deferred']])
                                for r in res[:3]],
                          complete_examples=[dict(eq=r['eq'], sat=r['sat'], strict=r['strict'],
                                                  plan=[str(p) for p in r['plan']],
                                                  deferred=[str(d) for d in r['deferred']]) for r in complete[:5]])
        sys.stdout.flush()
    with open('symdep_w_results.json', 'w') as f:
        json.dump(out_all, f, indent=1, default=str)
