#!/usr/bin/env python3
"""R=20 symbolic dependency tracer over ALL assignments (which words are unknown,
which is swept, which context conditions), with mild/heavy edge classes.

Extends symdep/L5_50_symdep.py: every W_r (r=0..19) is expanded in the a-words,
a12..a19 are chain symbols, e_r for r>=8 may be saturated when free of unknowns,
context words in 4..11 may be equated.

Edge classes of (C_j, u):
   none        absent
   lin         only additively                              -> subtraction
   s0lin       only inside one s0(u + rest)                 -> sigma0 inversion
   s0lin+lin   both                                         -> sigma0(u)-u table
   gamma       (C0, a0) special: a0 in W0 (lin) and W1 (g)  -> Gamma table (barrier note)
   mild        additively and/or as a DIRECT argument of an uncollapsed Maj/Ch, never
               under S0/S1/s1/s0 and never inside a nested nonlinear argument;
               provisional guess costs -log2 P(residual == guess) bits
   heavy       anything else
"""
import itertools, sys, json
from collections import Counter

def sym(n): return frozenset({(('sym', n), 1)})
def lit(v): return frozenset({(('lit', v & 0xFFFFFFFF), 1)})
ZERO = frozenset()

def add(*es):
    d = {}
    for e in es:
        for a, c in e:
            d[a] = d.get(a, 0) + c
    return frozenset((a, c) for a, c in d.items() if c % (1 << 32))

def neg(e): return frozenset((a, -c) for a, c in e)

def is_lit(e): return all(at[0] == 'lit' for at, c in e)
def lit_val(e):
    s = 0
    for at, c in e: s += c * at[1]
    return s & 0xFFFFFFFF

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
        if v == 0xFFFFFFFF: return b
        if v == 0: return c
    if b == c: return b
    return op('Ch', a, b, c)

def T2(a, b, c): return add(S0(a), Maj(a, b, c))

def syms_of(e, out=None):
    if out is None: out = set()
    for at, c in e:
        if at[0] == 'sym': out.add(at[1])
        elif at[0] == 'lit': pass
        else:
            for arg in at[1:]: syms_of(arg, out)
    return out

def occ(expr, var, mode, tags):
    """mode: None (top-level sum), 'majch' (direct arg of Maj/Ch), 'heavy'."""
    for at, c in expr:
        if at[0] == 'sym':
            if at[1] == var:
                tags.add('lin' if mode is None else mode)
        elif at[0] == 'lit':
            pass
        else:
            o = at[0]
            if o in ('S0', 'S1', 's1'):
                occ(at[1], var, 'heavy', tags)
            elif o == 's0':
                sub = set(); occ(at[1], var, None, sub)
                if sub == {'lin'}: tags.add('s0lin')
                elif sub: tags.add('heavy')
            elif o in ('Maj', 'Ch'):
                for arg in at[1:]:
                    sub = set(); occ(arg, var, None, sub)
                    if not sub: continue
                    if sub == {'lin'}:
                        # direct argument: var + (rest).  bitwise in var only if rest
                        # is free of everything?  we call it mild; cost measured later
                        tags.add('majch' if mode is None else 'heavy')
                    else:
                        tags.add('heavy')
            else:
                raise ValueError(o)
    return tags

def classify(expr, var):
    t = occ(expr, var, None, set())
    if not t: return 'none'
    if 'heavy' in t: return 'heavy'
    if t == {'lin'}: return 'lin'
    if t == {'s0lin'}: return 's0lin'
    if t == {'lin', 's0lin'}: return 's0lin+lin'
    if t <= {'lin', 'majch'}: return 'mild'
    return 'heavy'

R = 20
CHAIN = set(range(12, 20))

def build(unknown, eqmap, sat, swept=None):
    """unknown: set of indices in 0..11 that are unknowns.
       eqmap: {i: j} context word a_i := a_j (both context, in 4..11).
       sat: {r: 0 or 0xFFFFFFFF} for r in 8..15.
       Returns (C[0..3], legal, reason)."""
    a = {}
    for i in (-4, -3, -2, -1): a[i] = sym(f'IV{i}')
    for i in range(0, 12):
        if i in unknown: a[i] = sym(f'a{i}')
        else: a[i] = sym(f'a{eqmap.get(i, i)}')
    for i in range(12, R): a[i] = sym(f'A{i}')
    e = {}
    for i in (-4, -3, -2, -1): e[i] = sym(f'IVe{i}')
    for r in range(0, R):
        e[r] = add(a[r - 4], a[r], neg(T2(a[r - 1], a[r - 2], a[r - 3])))
    unk_names = {f'a{i}' for i in unknown if i != swept}
    if any(v in unknown and v != swept for v in eqmap.values()):
        return None, False, 'context tied to a non-swept unknown'
    # legality of saturations: each needs a context word (not chain, not unknown,
    # not the dependent side of an equality, not consumed by another saturation)
    consumed = set(eqmap.keys())
    for r in sorted(sat):
        if syms_of(e[r]) & unk_names:
            return None, False, f'e{r} depends on an unknown'
        reps = set(eqmap.values())
        cands = [k for k in (r, r - 4, r - 1, r - 2, r - 3)
                 if 0 <= k <= 11 and k not in unknown and k not in consumed and k not in reps]
        if not cands:
            return None, False, f'e{r} has no free context word'
        # prefer the linear entries a_r, a_{r-4}
        consumed.add(cands[0])
        e[r] = lit(sat[r])
    W = {}
    for r in range(0, R):
        W[r] = add(a[r], neg(T2(a[r - 1], a[r - 2], a[r - 3])), neg(e[r - 4]),
                   neg(S1(e[r - 1])), neg(Ch(e[r - 1], e[r - 2], e[r - 3])),
                   neg(sym(f'K{r}')))
    C = [add(s0(W[1 + t]), W[t], W[9 + t], s1(W[14 + t]), neg(W[16 + t])) for t in range(4)]
    return C, True, ''

ABSORB = {'lin', 's0lin', 's0lin+lin', 'gamma'}

def edge_matrix(unknown, eqmap, sat, swept=None):
    C, legal, why = build(unknown, eqmap, sat, swept)
    if not legal: return None, why
    unk = sorted(unknown)
    mat = {}
    for t in range(4):
        for k in unk:
            cl = classify(C[t], f'a{k}')
            if t == 0 and k == 0 and cl == 'heavy':
                # a0 reaches C0 only via W0 (lin) and W1 (g(a0)); Gamma table absorbs it
                cl = 'gamma'
            mat[(t, k)] = cl
    return mat, ''

MILD_COST = 13.3   # optimistic: (3/4)^32 collapse; real value measured separately

def best_schedule(mat, unk, allow_heavy=False, only_sweep=None):
    """Sweep one unknown, absorb the others in some order.  A constraint absorbing u
    may depend on later unknowns only through 'lin' or 'mild' edges (provisional,
    charged 32 or MILD_COST bits); any 'heavy' dependence on a later unknown is
    forbidden.  Constraints left over are 32-bit filters.  Returns (bits, plan)."""
    best = (1e9, None)
    n = len(unk)
    for swept in unk:
        if only_sweep is not None and swept != only_sweep: continue
        rest = [u for u in unk if u != swept]
        for order in itertools.permutations(rest):
            for cons in itertools.permutations(range(4), len(rest)):
                ok = True; bits = 0.0; plan = []
                for i, (u, t) in enumerate(zip(order, cons)):
                    cl = mat[(t, u)]
                    if cl not in ABSORB: ok = False; break
                    later = order[i + 1:]
                    has_lin = has_mild = False
                    for w in later:
                        cw = mat[(t, w)]
                        if cw in ('heavy', 'gamma', 's0lin', 's0lin+lin'):
                            if allow_heavy: has_lin = True
                            else: ok = False; break
                        elif cw == 'lin': has_lin = True
                        elif cw == 'mild': has_mild = True
                    if not ok: break
                    # one joint provisional residual per constraint: a linear later
                    # unknown makes it a full 32-bit guess; mild-only is charged the
                    # optimistic collapsed rate (measured separately for candidates)
                    if has_lin: bits += 32
                    elif has_mild: bits += MILD_COST
                    plan.append((f'C{t}', f'a{u}', cl))
                if not ok: continue
                bits += 32 * (4 - len(rest))
                if bits < best[0]:
                    best = (bits, (f'a{swept}', plan))
    return best


def enumerate_configs(unknown, swept=None):
    ctx = [i for i in range(4, 12) if i not in unknown]
    # equalities: partition-like maps, restrict to at most 3 equalities among ctx words 4..7
    # plus optional a_k = a_4 style for k up to 11 (keep it small)
    eq_opts = [{}]
    small = [i for i in ctx if i <= 7] + ([swept] if swept is not None and swept <= 7 else [])
    small = sorted(small)
    for i in small:
        for j in small:
            if i != j and j < i: eq_opts.append({i: j})
    for i, j, k in itertools.combinations(small, 3):
        eq_opts.append({j: i, k: i})
    if len(small) == 4:
        eq_opts.append({small[1]: small[0], small[2]: small[0], small[3]: small[0]})
    big = [i for i in ctx if i >= 8]
    for i in big:
        for j in big:
            if j < i: eq_opts.append({i: j})
    sat_opts = []
    for vals in itertools.product((None, 0, 0xFFFFFFFF), repeat=4):
        sat_opts.append({8 + i: v for i, v in enumerate(vals) if v is not None})
    # e12..e15 saturations (via a8..a11) as single additions
    base = list(sat_opts)
    for s in base:
        for r in (12, 13, 14, 15):
            for v in (0, 0xFFFFFFFF):
                d = dict(s); d[r] = v; sat_opts.append(d)
    for eq in eq_opts:
        for sat in sat_opts:
            yield eq, sat


def run_unknown_set(unknown, allow_heavy=False):
    unknown = set(unknown)
    unk = sorted(unknown)
    out = []
    seen = set()
    for swept in unk:
        for eq, sat in enumerate_configs(unknown, swept):
            mat, why = edge_matrix(unknown, eq, sat, swept)
            if mat is None: continue
            key = (swept,) + tuple(mat[(t, k)] for t in range(4) for k in unk)
            if key in seen: continue
            seen.add(key)
            bits, plan = best_schedule(mat, unk, allow_heavy, only_sweep=swept)
            out.append((bits, plan, eq, sat, key))
    out.sort(key=lambda x: x[0])
    return out


if __name__ == '__main__':
    import multiprocessing as mp
    sets = []
    for n in (4, 5):
        for U in itertools.combinations(range(12), n):
            if sum(1 for u in U if u >= 7) > 1: continue
            sets.append(U)
    print(f'{len(sets)} unknown sets')
    allow_heavy = '--heavy' in sys.argv
    with mp.Pool(24) as p:
        results = p.starmap(run_unknown_set, [(U, allow_heavy) for U in sets])
    summary = []
    for U, out in zip(sets, results):
        if not out: continue
        bits, plan, eq, sat, key = out[0]
        summary.append((bits, U, plan, eq, sat))
    summary.sort(key=lambda x: (x[0], x[1]))
    print('\nbest schedule per unknown set (filter bits; lower is better; MILD charged 13.3):')
    for bits, U, plan, eq, sat in summary[:80]:
        if plan is None: continue
        print(f'  {bits:6.1f}  U={U}  sweep {plan[0]}  {plan[1]}  eq={eq} sat={ {k: hex(v) for k, v in sat.items()} }')
    n_none = sum(1 for s in summary if s[2] is None)
    print(f'\n{n_none} unknown sets with NO legal schedule; {len(summary) - n_none} with one')
    json.dump([(b, list(U), p, {str(k): v for k, v in eq.items()}, {str(k): v for k, v in sat.items()})
               for b, U, p, eq, sat in summary], open('assign_results%s.json' % ('_heavy' if allow_heavy else ''), 'w'))
