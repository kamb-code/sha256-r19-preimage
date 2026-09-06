#!/usr/bin/env python3
"""Symbolic trace of every path from the free context word a11 into the four
per-context constants of the R=20 submask attack:

    KC0 = W16 - s1(W14) - W9hat            (C0 lookup constant)
    KC1 = W17 - s1(W15) - W10base + D      (C1 lookup constant)
    KC2 = W18 - s1(W16) - W11base + T1_7 + Ch(e10,e9,e8)
    collapse: Maj(v, a3, a2) == a3         (depends on v only)
    kappa3 = W19 - s1(W17) - W12           (fourth-constraint constant)

Expression algebra as in L5_50_symdep.py (sum of atoms, Maj/Ch collapse on
equal arguments and on all-ones / all-zero selectors).  The tracer walks each
constant, records every occurrence of a11 together with the chain of enclosing
operators, and for each Maj/Ch on the chain names the saturation that would
cut it and whether the words needed for that saturation are FREE context
words, family-determined words, chain (digest) words, or a11 itself.

A path is "uncuttable" when every operator on it is unary (S0, S1, s0, s1) or
is a Maj/Ch whose cut would require a digest word or a11 itself.

Then the same trace is repeated under every legal extra saturation that could
touch an a11 path (a10 pinned six ways, a7 pinned by e11), to show that no
configuration removes a11 from KC0 (or KC2): the atom S1(e15), with
e15 = a11 + A15 - S0(A14) - Maj(A14, A13, A12), survives everything.
"""
import itertools, sys

M = 0xFFFFFFFF

# ---------- expression algebra (sum of atoms) ----------
def sym(n): return frozenset({(('sym', n), 1)})
def lit(v): return frozenset({(('lit', v & M), 1)})
ZERO = frozenset()

def add(*es):
    d = {}
    for e in es:
        for a, c in e:
            d[a] = d.get(a, 0) + c
    return frozenset((a, c) for a, c in d.items() if c % (1 << 32))

def neg(e): return frozenset((a, -c) for a, c in e)
def sub(a, b): return add(a, neg(b))
def is_lit(e): return len(e) == 0 or all(at[0] == 'lit' for at, c in e)
def lit_val(e):
    return sum(c * at[1] for at, c in e) & M

def op(name, *args): return frozenset({((name,) + tuple(args), 1)})
def S0(e): return lit(0) if is_lit(e) and False else op('S0', e)
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
        if v == M: return b
        if v == 0: return c
    if b == c: return b
    return op('Ch', a, b, c)

def T2(a, b, c): return add(S0(a), Maj(a, b, c))

def contains(e, name):
    for at, c in e:
        if at[0] == 'sym':
            if at[1] == name: return True
        elif at[0] != 'lit':
            if any(contains(x, name) for x in at[1:]): return True
    return False

def symbols(e, out=None):
    if out is None: out = set()
    for at, c in e:
        if at[0] == 'sym': out.add(at[1])
        elif at[0] != 'lit':
            for x in at[1:]: symbols(x, out)
    return out

def short(e, depth=0):
    """compact printable form"""
    if is_lit(e):
        return f"0x{lit_val(e):x}" if e else "0"
    parts = []
    for at, c in sorted(e, key=lambda t: str(t)):
        if at[0] == 'sym': s = at[1]
        elif at[0] == 'lit': s = f"0x{at[1]:x}"
        else:
            s = at[0] + "(" + ",".join(short(x, depth + 1) for x in at[1:]) + ")"
        parts.append(("+" if c > 0 else "-") + (str(abs(c)) if abs(c) != 1 else "") + s)
    return "".join(parts)

# ---------- kinds of words ----------
def kind(name, free, fam, chain):
    if name == 'a11': return 'a11'
    if name in free: return 'FREE'
    if name in fam: return 'family'
    if name in chain: return 'DIGEST'
    return 'IV/K'

# ---------- path tracer ----------
def trace(e, var, stack, out, coef=1):
    """collect (coef, stack) for every occurrence of var in e.
    stack: list of (opname, argument index, sibling expressions)"""
    for at, c in e:
        if at[0] == 'sym':
            if at[1] == var: out.append((c * coef, list(stack)))
        elif at[0] == 'lit':
            continue
        else:
            o = at[0]
            for i, arg in enumerate(at[1:]):
                if contains(arg, var):
                    sibs = [x for j, x in enumerate(at[1:]) if j != i]
                    stack.append((o, i, sibs))
                    trace(arg, var, stack, out, coef * c)
                    stack.pop()

def cut_analysis(stack, free, fam, chain):
    """For a path (list of enclosing ops, outermost first), decide whether a
    saturation can cut it.  Returns (cuttable: bool, note)."""
    notes = []
    cuttable = False
    for o, i, sibs in stack:
        if o in ('S0', 'S1', 's0', 's1'):
            notes.append(f"{o}: unary, no saturation")
            continue
        if o == 'Maj':
            # Maj(x,y,z): the a11-carrying argument is cut iff the other two are equal
            s = [symbols(x) for x in sibs]
            k = [",".join(sorted(kind(n, free, fam, chain) for n in ss)) or 'const' for ss in s]
            if all(is_lit(x) for x in sibs):
                notes.append("Maj: siblings literal -> collapsed already")
            else:
                need = f"Maj: cut iff {short(sibs[0])} == {short(sibs[1])}  [siblings: {k[0]} | {k[1]}]"
                ok = not any('DIGEST' in kk for kk in k) or (
                    # two digest words equal: not choosable
                    False)
                # a sibling containing a11 cannot be used to cut an a11 path
                ok = ok and not any(contains(x, 'a11') for x in sibs)
                # both siblings must be movable: at least one FREE or family word among them
                ok = ok and any(('FREE' in kk or 'family' in kk) for kk in k)
                notes.append(need + ("  -> CUTTABLE" if ok else "  -> not cuttable"))
                cuttable = cuttable or ok
        if o == 'Ch':
            if i == 0:
                # a11 in the selector: cut only by fixing the selector (pins a11 unless
                # another word in the selector absorbs it) or by making the two data
                # inputs equal
                s = [symbols(x) for x in sibs]
                k = [",".join(sorted(kind(n, free, fam, chain) for n in ss)) or 'const' for ss in s]
                ok = (not any(contains(x, 'a11') for x in sibs)
                      and any(('FREE' in kk or 'family' in kk) for kk in k)
                      and not any('DIGEST' in kk for kk in k))
                notes.append(f"Ch(selector): cut iff {short(sibs[0])} == {short(sibs[1])}  "
                             f"[{k[0]} | {k[1]}]" + ("  -> CUTTABLE" if ok else "  -> not cuttable"))
                cuttable = cuttable or ok
            else:
                sel = sibs[0]
                sk = ",".join(sorted(kind(n, free, fam, chain) for n in symbols(sel))) or 'const'
                want = M if i == 2 else 0     # a11 in arg1 (i==1): need selector 0; in arg2: need -1
                ok = ('DIGEST' not in sk) and ('a11' not in sk) and (('FREE' in sk) or ('family' in sk))
                notes.append(f"Ch(data{i}): cut iff selector {short(sel)} == 0x{want:x}  [selector: {sk}]"
                             + ("  -> CUTTABLE" if ok else "  -> not cuttable"))
                cuttable = cuttable or ok
    return cuttable, notes

# ---------- the R=20 family, symbolically ----------
def build(extra=None):
    """extra: dict of substitutions applied to context words (a10, a7, a6 ...),
    each a function (a, e_known) -> expr, evaluated in order."""
    extra = extra or {}
    a = {}
    for i in (-4, -3, -2, -1): a[i] = sym(f'IV{i}')
    for i in range(0, 4): a[i] = sym(f'a{i}')
    v = sym('v')
    a[4] = v; a[5] = v
    a[6] = sym('a6'); a[7] = sym('a7'); a[10] = sym('a10'); a[11] = sym('a11')
    for i in range(12, 20): a[i] = sym(f'A{i}')
    # e-words: e_r = a_{r-4} + a_r - T2(a_{r-1}, a_{r-2}, a_{r-3})
    def e_of(r): return add(a[r - 4], a[r], neg(T2(a[r - 1], a[r - 2], a[r - 3])))
    # family: e8 = e9 = -1 solved for a8, a9
    a[8] = add(lit(M), neg(a[4]), T2(a[7], a[6], a[5]))
    a[9] = add(lit(M), neg(a[5]), T2(a[8], a[7], a[6]))
    # extra saturations (may depend on a8, a9, a11, chain words)
    for w, fn in extra.items():
        a[w] = fn(a, e_of)
    e = {}
    for i in (-4, -3, -2, -1): e[i] = sym(f'IVe{i}')
    for r in range(0, 20):
        e[r] = e_of(r)
    e[8] = lit(M); e[9] = lit(M)
    # e16..e19 are digest words; the formula gives them in terms of chain words,
    # equivalent; keep them as chain symbols for readability
    for r in range(16, 20): e[r] = sym(f'E{r}')
    W = {}
    for r in range(0, 20):
        W[r] = add(a[r], neg(T2(a[r - 1], a[r - 2], a[r - 3])), neg(e[r - 4]),
                   neg(S1(e[r - 1])), neg(Ch(e[r - 1], e[r - 2], e[r - 3])), neg(sym(f'K{r}')))
    # the per-context constants exactly as attack_context() forms them
    T1_7 = add(a[7], neg(T2(a[6], a[5], a[4])))
    c6 = add(a[6], neg(S0(a[5])))
    W9base = add(a[9], neg(T2(a[8], a[7], a[6])), neg(sym('K9')))
    W10base = add(a[10], neg(T2(a[9], a[8], a[7])), neg(S1(e[9])), neg(sym('K10')))
    W11base = add(a[11], neg(T2(a[10], a[9], a[8])), neg(S1(e[10])), neg(sym('K11')))
    K0p = sub(W[16], s1(W[14]))
    K1p = sub(W[17], s1(W[15]))
    K2p = sub(W[18], s1(W[16]))
    K3p = add(W[19], neg(s1(W[17])), neg(W[12]))
    z = lit(0)
    W9hat = add(W9base, neg(add(a[5], neg(S0(a[4])), neg(Maj(a[4], z, z)))), neg(S1(e[8])),
                neg(Ch(e[8], T1_7, add(c6, neg(Maj(a[5], a[4], z))))))
    D = add(a[6], neg(S0(a[5])), neg(Maj(a[5], a[4], z)), Ch(e[9], e[8], T1_7))
    KC0 = sub(K0p, W9hat)
    KC1 = add(K1p, neg(W10base), D)
    KC2 = add(K2p, neg(W11base), T1_7, Ch(e[10], e[9], e[8]))
    collapse = Maj(v, sym('a3'), sym('a2'))     # == a3 ; depends on v only
    return dict(KC0=KC0, KC1=KC1, KC2=KC2, collapse=collapse, kappa3=K3p,
                W={r: W[r] for r in range(11, 20)}, e=e, a=a)

FREE = {'v', 'a6', 'a7', 'a10'}
FAM = {'a8', 'a9'}          # not symbols in build(); kept for the kind() of extras
CHAIN = {f'A{i}' for i in range(12, 20)} | {f'E{i}' for i in range(16, 20)}

def report(name, e, verbose=True):
    paths = []
    trace(e, 'a11', [], paths)
    print(f"\n=== {name}: {len(paths)} occurrence(s) of a11 ===")
    if not paths:
        print("   a11-FREE")
        return 0, 0
    n_uncut = 0
    for coef, st in paths:
        chain = " -> ".join(f"{o}[arg{i}]" for o, i, _ in st) or "(linear)"
        cuttable, notes = cut_analysis(st, FREE, FAM, CHAIN)
        tag = "cuttable by a saturation" if cuttable else "UNCUTTABLE"
        if not cuttable: n_uncut += 1
        if verbose:
            print(f"  coef {coef:+d}  path: a11 <- {chain}   [{tag}]")
            for n in notes: print(f"        {n}")
    print(f"  => {n_uncut} uncuttable path(s) of {len(paths)}")
    return len(paths), n_uncut

def w_paths(expr, label):
    paths = []
    trace(expr, 'a11', [], paths)
    kinds = []
    for coef, st in paths:
        kinds.append(" -> ".join(f"{o}[{i}]" for o, i, _ in st) or "lin")
    print(f"  {label:6s}: {len(paths):2d} path(s): " + "; ".join(kinds))

if __name__ == "__main__":
    B = build()
    print("R=20 submask family, a4=a5=v, e8=e9=-1 via a8,a9; free: v,a6,a7,a10,a11")
    print("\n--- where a11 sits in the e-words and message words ---")
    for r in range(11, 16):
        print(f"  e{r} = {short(B['e'][r])}")
    for r in range(11, 20):
        w_paths(B['W'][r], f"W{r}")
    tot = {}
    for k in ('KC0', 'KC1', 'KC2', 'collapse', 'kappa3'):
        tot[k] = report(k, B[k])

    # ---- extra saturations: every legal way to pin a10 or a7 that touches an a11 path
    print("\n\n######## RE-TRACE UNDER EXTRA SATURATIONS ########")
    def a10_eq_a12(a, e_of): return a[12]                       # kills Maj(A12,a11,a10) in e13
    def a10_eq_a9(a, e_of): return a[9]                         # kills Maj(a11,a10,a9) in e12/W12
    def a10_eq_a11(a, e_of): return a[11]                       # Maj(a11,a10,a9)->a11 (moves a11 around)
    def e10_ones(a, e_of): return add(lit(M), neg(a[6]), T2(a[9], a[8], a[7]))   # e10=-1 via a10
    def e10_zero(a, e_of): return add(lit(0), neg(a[6]), T2(a[9], a[8], a[7]))   # e10=0 via a10
    def e14_ones(a, e_of): return add(lit(M), neg(a[14]), T2(a[13], a[12], a[11]))  # e14=-1 via a10
    def e14_zero(a, e_of): return add(lit(0), neg(a[14]), T2(a[13], a[12], a[11]))
    # e11 = a7 + a11 - T2(a10,a9,a8) = c via a7: a7 = c - a11 + T2(a10,a9,a8).  a8, a9 depend on a7
    # in the family (circular); we substitute with a8, a9 left as they are, which UNDER-counts
    # the a11 paths (any true solution has a8, a9 depending on a11 too).
    def e11_ones(a, e_of): return add(lit(M), neg(a[11]), T2(a[10], a[9], a[8]))
    def e11_zero(a, e_of): return add(lit(0), neg(a[11]), T2(a[10], a[9], a[8]))

    a10_menu = [("none", None), ("a10=A12", a10_eq_a12), ("a10=a9", a10_eq_a9), ("a10=a11", a10_eq_a11),
                ("e10=-1 via a10", e10_ones), ("e10=0 via a10", e10_zero),
                ("e14=-1 via a10", e14_ones), ("e14=0 via a10", e14_zero)]
    a7_menu = [("none", None), ("e11=-1 via a7", e11_ones), ("e11=0 via a7", e11_zero)]
    print(f"\n{'a10 choice':18s} {'a7 choice':16s} | KC0 paths(uncut)  KC1  KC2  collapse  kappa3 | KC0 has S1(e15)?")
    for (n10, f10), (n7, f7) in itertools.product(a10_menu, a7_menu):
        ex = {}
        if f7 is not None: ex[7] = f7
        if f10 is not None: ex[10] = f10
        # note: substitution order matters when both are present (a7 uses a10, a10 may use a7)
        Bx = build(ex)
        row = []
        for k in ('KC0', 'KC1', 'KC2', 'collapse', 'kappa3'):
            p = []; trace(Bx[k], 'a11', [], p)
            unc = sum(1 for c, st in p if not cut_analysis(st, FREE, FAM, CHAIN)[0])
            row.append(f"{len(p)}({unc})")
        # is the specific atom S1(e15) present in KC0 ?
        e15 = Bx['e'][15]
        has = any(at[0] == 'S1' and at[1] == e15 for at, c in Bx['KC0'])
        print(f"{n10:18s} {n7:16s} | {row[0]:>16s} {row[1]:>4s} {row[2]:>4s} {row[3]:>9s} {row[4]:>7s} | "
              f"{'yes' if has else 'NO'}   e15 = {short(e15)}")
