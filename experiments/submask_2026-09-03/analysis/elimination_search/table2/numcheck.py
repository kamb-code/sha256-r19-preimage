#!/usr/bin/env python3
"""Full-width numeric confirmation of the symbolic dependency classes.

For a configuration (promoted word w, equalities among context words,
saturations of e8..e15) build random 20-round states that satisfy it (random
IV-side unknowns a0..a3, random w, random digest words a12..a19), flip one bit
of w, and record the Hamming weight of the change of every constraint value

    C_j = W_{16+j} - s1(W_{14+j}) - W_{9+j} - s0(W_{1+j}) - W_j .

'none' must give exactly 0 in every trial; 'mild' about 1-2 bits; 'heavy'
(a one-input map of w that survives every saturation) far more.

Also checks the SEPARABILITY of C3 in w:  C3(w) - C3(w') must not depend on
(a0..a3) if C3 = F_ctx(w) + G(a0..a3, ctx), which is what a per-context table
for w needs; and on the context if the table is to be global.
"""
import sys, json
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, recover_W

R = 20

def build_context(rng, w, eqmap, sat, tries=20):
    """context a4..a11 satisfying eqmap (i -> rep) and sat (r -> value); a12..a19
    random (digest side).  Each saturation e_r = c is solved through a word x in
    {a_{r-4}, a_r} that enters e_r linearly, is a free context word (not w, not
    consumed by an equality), and does not enter any e_{r'} solved EARLIER
    (a word x enters e_{x}..e_{x+4}); such an order is found by search."""
    import itertools
    consumed = set(eqmap.keys())
    free = [i for i in range(4, 12) if i != w and i not in consumed]
    sats = sorted(sat)
    plan = None
    for order in itertools.permutations(sats):
        for words in itertools.product(*[[x for x in (r, r - 4) if x in free] for r in order]):
            if len(set(words)) != len(words): continue
            ok = True
            for k, (r, x) in enumerate(zip(order, words)):
                # x must not enter any earlier-solved e_{r'}: r' in [x, x+4]
                if any(x <= rp <= x + 4 for rp in order[:k]): ok = False; break
            if ok: plan = list(zip(order, words)); break
        if plan is not None: break
    if plan is None and sats: return None
    for _ in range(tries):
        a = {i: int(rng.integers(0, 1 << 32, dtype=np.uint64)) for i in range(4, R)}
        for i, rep in eqmap.items(): a[i] = a[rep]
        for r, x in (plan or []):
            other = a[r] if x == r - 4 else a[r - 4]
            a[x] = (sat[r] - other + T2(a[r - 1], a[r - 2], a[r - 3])) & M
            for i, rep in eqmap.items():
                if rep == x: a[i] = a[x]
        good = all((a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M == sat[r] for r in sat) \
            and all(a[i] == a[rep] for i, rep in eqmap.items())
        if good: return a
    return None

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

def edge_weights(w, eqmap, sat, trials=200, seed=1):
    rng = np.random.default_rng(seed)
    hw = np.zeros((trials, 4)); nz = np.zeros(4, dtype=int); built = 0
    for t in range(trials):
        ctx = build_context(rng, w, eqmap, sat)
        if ctx is None: return None
        built += 1
        unk = {i: int(rng.integers(0, 1 << 32, dtype=np.uint64)) for i in range(4)}
        a, e = full_state(ctx, unk); c0 = constraints(a, e)
        bit = int(rng.integers(0, 32))
        ctx2 = dict(ctx); ctx2[w] ^= (1 << bit)
        for i, rep in eqmap.items():
            if rep == w: ctx2[i] = ctx2[w]
        a2, e2 = full_state(ctx2, unk); c1 = constraints(a2, e2)
        for j in range(4):
            d = c0[j] ^ c1[j]
            hw[t, j] = bin(d).count('1'); nz[j] += (d != 0)
    return hw.mean(0), nz, built

def separability(w, eqmap, sat, trials=100, seed=2):
    """C3(w) - C3(w') independent of a0..a3?  (per-context one-input form)
       and additionally independent of the context? (global form)"""
    rng = np.random.default_rng(seed)
    dep_unk = 0; dep_ctx = 0; n = 0
    for t in range(trials):
        ctx = build_context(rng, w, eqmap, sat)
        if ctx is None: return None
        w1 = ctx[w]; w2 = int(rng.integers(0, 1 << 32, dtype=np.uint64))
        def c3(ctxd, wv, unk):
            c = dict(ctxd); c[w] = wv
            for i, rep in eqmap.items():
                if rep == w: c[i] = wv
            a, e = full_state(c, unk); return constraints(a, e)[3]
        u1 = {i: int(rng.integers(0, 1 << 32, dtype=np.uint64)) for i in range(4)}
        u2 = {i: int(rng.integers(0, 1 << 32, dtype=np.uint64)) for i in range(4)}
        d1 = (c3(ctx, w1, u1) - c3(ctx, w2, u1)) & M
        d2 = (c3(ctx, w1, u2) - c3(ctx, w2, u2)) & M
        dep_unk += (d1 != d2)
        ctxb = build_context(rng, w, eqmap, sat)
        d3 = (c3(ctxb, w1, u1) - c3(ctxb, w2, u1)) & M
        dep_ctx += (d1 != d3)
        n += 1
    return dep_unk, dep_ctx, n

if __name__ == "__main__":
    res = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "symdep_w_results.json"))
    for w, d in sorted(res.items(), key=lambda x: int(x[0])):
        w = int(w)
        print(f"\n=== w = a{w}: symbolic min non-deferrable entries into C0..C2 = {d['min_block']}; "
              f"C3 absorbs in {d['n_c3']} cfgs; complete schedules {d['n_complete']}")
        for b in d['best'][:2]:
            eqmap = {int(i): int(r) for i, r in b['eq']}
            sat = {int(r): int(v) for r, v in b['sat']}
            print(f"  cfg eq={sorted(eqmap.items())} sat={[(r, hex(v)) for r, v in sorted(sat.items())]} strictDOF={b['strict']}")
            for t in range(4): print(f"     symbolic C{t} <- a{w}: {b['entries'][str(t)]}")
            ew = edge_weights(w, eqmap, sat)
            if ew is None:
                print("     numeric: context not constructible by the linear-word solver (skipped)"); continue
            hw, nz, built = ew
            print("     numeric mean Hamming weight of dC_j per flipped bit of a%d: " % w +
                  "  ".join(f"C{j}={hw[j]:5.2f} (moved {nz[j]}/{built})" for j in range(4)))
            sp = separability(w, eqmap, sat)
            if sp: print(f"     C3 separability: dC3 depends on a0..a3 in {sp[0]}/{sp[2]} trials, on context in {sp[1]}/{sp[2]}")
