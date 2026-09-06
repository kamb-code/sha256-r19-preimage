#!/usr/bin/env python3
"""The counting argument for the fourth constraint at R=20.

Everything here is arithmetic on the attack's own measured rates plus an
exhaustive enumeration of absorption orders on the measured dependency graph.
"""
import itertools, math
log2 = math.log2

# ---- measured / exact rates (paper_submask_r20.tex, Sections 4-6) ----
c1 = 0.633673            # single-root table image fraction (exact)
c3 = 0.977443            # three-root coverage (exact)
sol1 = c1 ** 3           # triangular solutions per swept a0, one root
sol3 = c3 ** 3           # ... three roots  (= 0.9338, measured 0.93384)
pcol = 0.75 ** 32        # collapsed consistency condition, exact for uniform pairs
pc3 = 2.0 ** -32         # fourth constraint, measured uniform to 24 bits
N = 2.0 ** 32            # swept a0 per context

print("=== 1. expected preimages per context and per swept a0 ===")
print("  frame 'a4 = v context' : unknowns a0..a3 = 128 bits, constraints C0..C3 = 128 bits")
print("     -> expected 20-round preimages per (context, digest) = 2^0 = 1  (uniform model)")
print("     -> per swept a0: 2^-32")
print("  frame B (a4 unknown)   : unknowns a0..a4 = 160 bits -> 2^32 preimages per context")
print(f"  attack, three roots     : per a0 {sol3:.4f} * (3/4)^32 * 2^-32 = {sol3*pcol*pc3:.3e} = 2^{log2(sol3*pcol*pc3):.2f}")
print(f"     -> candidates per context {N*sol3*pcol:,.0f} (GPU counters: 402,880-402,910)")
print(f"     -> preimages per context {N*sol3*pcol*pc3:.3e} = 2^{log2(N*sol3*pcol*pc3):.2f};"
      f"  cost 2^{-log2(sol3*pcol*pc3):.2f} swept a0 per preimage")
print(f"  attack, one root        : cost 2^{-log2(sol1*pcol*pc3):.2f}")
print()
print("  Reading: of the ~1 solution of {C0,C1,C2,C3} that exists per context, the sweep finds")
print(f"  it with probability {sol3*pcol:.3e} = 2^{log2(sol3*pcol):.1f}.  That factor is NOT the C3 filter;")
print("  it is the fraction of solutions of {C0,C1,C2} that the triangular solve reaches, i.e. those")
print("  whose (a2,a3) satisfy Maj(v,a3,a2) = a3 (the provisional W9 is exact only there).")
print("  Per swept a0 the true system {C0,C1,C2} has ~1 solution (96 unknown bits, 96 constraint")
print(f"  bits); the lookups return the solution of {{C0hat,C1,C2}} instead, which coincides with a true")
print(f"  solution for a (3/4)^32 fraction of a0.  Ideal 3-lookup scheme (all solutions of C0..C2 per a0")
print(f"  in O(1)): 2^32 swept a0 per preimage.  Ideal 4-lookup scheme: {1/c3**4:.2f} per preimage.")

print("\n=== 2. cost per preimage for k lookups and 4-k filters ===")
print("  With one swept word, n further unknowns, k absorbed by lookups, n-k additionally swept and")
print("  4-k constraints left as 2^-32 filters:")
print("     work/context = 2^{32(1+n-k)},   yield/context = 2^{32(1+n-k)} * cov * pen * 2^{-32(4-k)}")
print("     cost/preimage = 2^{32(4-k)} / (cov * pen)          (independent of n)")
print("  where pen = product of freeze penalties of the dependency edges the order has to cut.")
for k in range(0, 5):
    print(f"     k={k}: floor 2^{32*(4-k)} / cov  (pen = 1 requires an acyclic order)")

print("\n=== 3. exhaustive enumeration of absorption orders on the measured graph (frame B) ===")
# Edge classes for C_j (row) vs unknown a_k (col), family-on-context (a5 = v, e8 = e9 = -1 at a4 = v),
# from symdep_r20.py (routes) and numeric_deps.py (weights):
#   'self'   : the absorbed word (C_j -> a_{j+1}), table lookup, coverage cov
#   'heavy'  : passes through S0/S1/s1 of the unknown (one-input): freeze cost 2^-32
#   'bit'    : bitwise residual Maj(v,a3,a2) - a3: JOINT freeze cost (3/4)^32 for (a2,a3) in C0
#   'exact'  : residual identically zero when a4 is frozen at v (family conditions)
#   'none'   : absent
#   'bit'    : bitwise residual (borrow-free Maj difference), freeze cost (3/4)^32 per constraint
#              -- C0: Maj(v,a3,a2) - a3 for the PAIR (a2,a3);  C1: Maj(v,a4,a3) - a4 for the pair
#              (a3,a4), which is identically 0 when a4 is frozen at v (then C1 is free of a3 too);
#              C2: Maj(a6,v,a4) - v, identically 0 when a4 is frozen at v, or when a6 = v and
#              e10 = -1 (shifted family; a10 chosen) -- we grant the latter as a free context option.
def edges(frozen):
    """classes of C_j vs a_k given the set of unknowns frozen at a guess (a4's guess is v)."""
    a4f = 4 in frozen
    return {
        0: {1: 'self', 2: 'bit', 3: 'bit', 4: 'heavy'},
        1: {1: 'heavy', 2: 'self', 3: 'none' if a4f else 'bit', 4: 'exact' if a4f else 'bit'},
        2: {1: 'heavy', 2: 'heavy', 3: 'self', 4: 'exact'},
        3: {1: 'heavy', 2: 'heavy', 3: 'heavy', 4: 'self'},
    }
cost_of = {'heavy': 2.0 ** -32, 'exact': 1.0, 'none': 1.0}


def evaluate(order, absorbed, edges=edges, cov=c3):
    """order: sequence in which unknowns get resolved; absorbed: those resolved by a lookup
    (the rest are frozen at a provisional value = 'context').  Constraint C_{k-1} absorbs a_k.
    Every unresolved dependency of C_{k-1} at the moment of the lookup is a frozen edge."""
    pen = 1.0; ncov = 0; used = set(); frozen_edges = []
    # an unknown that is never absorbed is fixed to a guess ('context'); its unused constraint
    # becomes a 2^-32 filter and that single factor already prices every edge into it, so its
    # edges cost nothing extra here (guess right <=> filter passes).
    frozen = set(k for k in (1, 2, 3, 4) if k not in absorbed)
    resolved = set(frozen)
    E = edges(frozen)
    for k in order:
        if k not in absorbed:
            continue
        j = k - 1
        used.add(j); ncov += 1
        deps = E[j]
        bit_pending = [m for m, cls in deps.items() if cls == 'bit' and m not in resolved]
        for m, cls in deps.items():
            if m == k or m in resolved: continue
            if cls == 'bit': continue
            pen *= cost_of[cls]
            if cls == 'heavy': frozen_edges.append(f"a{m}->C{j}")
        if bit_pending:
            pen *= 0.75 ** 32          # one joint bitwise freeze per constraint
            frozen_edges.append(f"({','.join('a%d'%m for m in bit_pending)})->C{j} bitwise")
        resolved.add(k)
    nfilters = 4 - len(used)
    # unabsorbed unknowns are swept (2^32 each) or, equivalently, frozen and filtered: same count
    cost = (2.0 ** (32 * nfilters)) / (pen * cov ** len(used))
    return cost, frozen_edges, nfilters


def enumerate_all(edges_fn):
    out = []
    for r in range(1, 5):
        for absorbed in itertools.combinations((1, 2, 3, 4), r):
            for order in itertools.permutations((1, 2, 3, 4)):
                cost, fe, nf = evaluate(order, set(absorbed), edges_fn)
                out.append((cost, order, absorbed, fe, nf))
    out.sort(key=lambda x: x[0])
    return out

best = enumerate_all(edges)
seen = set()
print("  cheapest distinct schemes (cost per preimage in swept a0, three-root coverage):")
for cost, order, absorbed, fe, nf in best:
    key = (round(log2(cost), 2), tuple(fe), nf)
    if key in seen: continue
    seen.add(key)
    print(f"    2^{log2(cost):5.2f}  absorb {absorbed} in order {order}  filters={nf}  cut edges={fe}")
    if len(seen) >= 7: break

print("\n  what deleting single edges would buy (same enumeration with the edge deleted):")
def deleted(j, k, both=None):
    def fn(frozen):
        E2 = edges(frozen)
        E2[j][k] = 'none'
        if both: E2[j][both] = 'none'
        return E2
    return fn
for args, label in [((0, 4), "a4->C0 (S0(a4) in e5 in W9)"), ((1, 1), "a1->C1 (S0(a1),S1(e1) in W2)"),
                    ((2, 2), "a2->C2 (S0(a2),S1(e2) in W3)"), ((3, 3), "a3->C3 (S0(a3),S1(e3) in W4)"),
                    ((0, 2, 3), "(a2,a3)->C0 bitwise"), ((1, 3, 4), "(a3,a4)->C1 bitwise")]:
    m = enumerate_all(deleted(*args))[0]
    print(f"    delete {label:<32}: best 2^{log2(m[0]):5.2f}  absorb {m[2]} order {m[1]} cuts {m[3]}")

print("\n=== 4. multi-context amortisation ===")
cand = N * sol3 * pcol
print(f"  one context: 2^32 lookups -> {cand:,.0f} candidates, each with a fixed S = s0(W4)+W3;")
print("  a context c' with the SAME (KC0,KC1,KC2,v) and a different kappa3 would re-use them.")
print("  (KC0,KC1,KC2) is a 96-bit function of the four free words (a6,a7,a10,a11) at fixed v;")
print("  numeric_deps.py (d): every free word moves every KC by ~16 bits, and symdep_r20.py names")
print("  a one-input route for each (a10: s1(W14) with W14 = ... - e10, e10 = a6 + a10 - T2(a9,a8,a7);")
print("  a11: S1(e15) in W16 with e15 = a11 + A15 - T2(A14,A13,A12); a6,a7: S0(a6), S0(a7) in W9hat).")
print("  So fibre-mates exist (2^32 per fibre generically) but must be found by collision search:")
print("  m-way collision on 96 bits costs ~2^(96(m-1)/m) constant evaluations; payoff m*cand*2^-32 preimages.")
for m in (2, 3, 4, 8, 16):
    work = 2.0 ** (96 * (m - 1) / m) + N
    pay = m * cand * pc3
    print(f"    m={m:>2}: work 2^{log2(work):5.1f}, preimages {pay:.2e}, cost/preimage 2^{log2(work/pay):5.1f}"
          f"   (attack: 2^{-log2(sol3*pcol*pc3):.1f})")
print("  Even a FREE fibre enumeration (cost 0 per mate, 2^32 mates) would give at best")
print(f"  2^32 lookups per {cand:,.0f} matched candidates = 2^{log2(N/cand):.1f} per preimage: exactly what")
print("  deleting the a4->C0 edge buys in section 3 (it is the same 32 bits, seen from the kappa side),")
print("  and it needs a (KC0,KC1,KC2)-invariant context motion, which no free word offers.")
