"""E2/E3: the atomic operation of the attack as a lattice problem.

   Given c, find u with sigma0(u) - u = c  (mod 2^w).

This is exactly what the 16 GB table does in one lookup.  It is the cleanest
lattice target in the whole attack: no Maj, no Ch, so the bit-level system is
PURELY Z-linear in 0/1 variables -- the most favourable case a lattice could
ever be handed from SHA-256.

Variables: u_0..u_{w-1}, y_0..y_{w-1} (= sigma0(u)), t_0..t_{w-1} (XOR carries),
           m (one 0/1 borrow), N = 3w+1.
Equations: w XOR relations  sum(src u) - y_i - 2 t_i = 0
           1 modular relation  sum 2^i y_i - sum 2^i u_i + 2^w m = c
"""
from __future__ import annotations
import sys, math, time, json
import numpy as np
from wsha import mk, s0_bit_sources, brute_s0_minus_u
import embed


def build_system(w, c, known_high=0, u_true=None):
    """known_high: substitute the top `known_high` bits of u as constants."""
    src = s0_bit_sources(w)
    fixed = {}
    if known_high:
        assert u_true is not None
        for j in range(w - known_high, w):
            fixed[j] = (u_true >> j) & 1
    # variable indexing
    idx = {}
    n = 0
    for j in range(w):
        if j in fixed:
            continue
        idx[('u', j)] = n; n += 1
    for j in range(w):
        idx[('y', j)] = n; n += 1
    for j in range(w):
        idx[('t', j)] = n; n += 1
    idx[('m', 0)] = n; n += 1
    N = n
    A, b = [], []
    for i in range(w):
        row = [0] * N
        const = 0
        for j in src[i]:
            if j in fixed:
                const += fixed[j]
            else:
                row[idx[('u', j)]] += 1
        row[idx[('y', i)]] -= 1
        row[idx[('t', i)]] -= 2
        A.append(row); b.append(-const)
    row = [0] * N
    const = 0
    for i in range(w):
        row[idx[('y', i)]] += (1 << i)
        if i in fixed:
            const -= (1 << i) * fixed[i]
        else:
            row[idx[('u', i)]] -= (1 << i)
    row[idx[('m', 0)]] += (1 << w)
    A.append(row); b.append(c - const)
    return A, b, idx, N, fixed


def build_system_B(w, c, known_high=0, u_true=None):
    """Encoding B: borrow chain instead of one big 2^i equation.
    All coefficients are in {-2,-1,1,2}; numerically benign for float GSO.
    Vars: u_j (free ones), y_j, t_j, b_1..b_w.  Eqs: w XOR + w borrow."""
    src = s0_bit_sources(w)
    fixed = {}
    if known_high:
        for j in range(w - known_high, w):
            fixed[j] = (u_true >> j) & 1
    idx = {}
    n = 0
    for j in range(w):
        if j not in fixed:
            idx[('u', j)] = n; n += 1
    for j in range(w):
        idx[('y', j)] = n; n += 1
    for j in range(w):
        idx[('t', j)] = n; n += 1
    for j in range(1, w + 1):
        idx[('b', j)] = n; n += 1
    N = n
    A, b = [], []
    for i in range(w):
        row = [0] * N; const = 0
        for j in src[i]:
            if j in fixed:
                const += fixed[j]
            else:
                row[idx[('u', j)]] += 1
        row[idx[('y', i)]] -= 1
        row[idx[('t', i)]] -= 2
        A.append(row); b.append(-const)
    for i in range(w):
        row = [0] * N; const = 0
        row[idx[('y', i)]] += 1
        if i in fixed:
            const -= fixed[i]
        else:
            row[idx[('u', i)]] -= 1
        if i >= 1:
            row[idx[('b', i)]] -= 1
        row[idx[('b', i + 1)]] += 2
        A.append(row); b.append(((c >> i) & 1) - const)
    return A, b, idx, N, fixed


def true_assignment_B(w, u, c, idx, fixed):
    f = mk(w); M = f['M']
    y = f['s0'](u)
    z = [0] * len(idx)
    for j in range(w):
        if j not in fixed:
            z[idx[('u', j)]] = (u >> j) & 1
        z[idx[('y', j)]] = (y >> j) & 1
    src = s0_bit_sources(w)
    for i in range(w):
        s = sum((u >> j) & 1 for j in src[i])
        z[idx[('t', i)]] = s >> 1
    bor = 0
    for i in range(w):
        v = ((y >> i) & 1) - ((u >> i) & 1) - bor - ((c >> i) & 1)
        bor = 1 if v < 0 else 0
        z[idx[('b', i + 1)]] = bor
    return z


def true_assignment(w, u, c, idx, fixed):
    f = mk(w); M = f['M']
    y = f['s0'](u)
    z = [0] * len(idx)
    for j in range(w):
        if j not in fixed:
            z[idx[('u', j)]] = (u >> j) & 1
        z[idx[('y', j)]] = (y >> j) & 1
    src = s0_bit_sources(w)
    for i in range(w):
        s = sum((u >> j) & 1 for j in src[i])
        z[idx[('t', i)]] = s >> 1
    z[idx[('m', 0)]] = 1 if (y - u - c) < 0 else 0
    return z


def run(w, trials=50, known_high=0, use_bkz=False, seed=7, verbose=False, enc='B'):
    rng = np.random.default_rng(seed)
    f = mk(w); M = f['M']
    ok = 0
    diag_acc = []
    t0 = time.time()
    done = 0
    build = build_system_B if enc == 'B' else build_system
    truth = true_assignment_B if enc == 'B' else true_assignment
    while done < trials:
        u_true = int(rng.integers(0, 1 << w))
        c = (f['s0'](u_true) - u_true) & M
        A, b, idx, N, fixed = build(w, c, known_high, u_true)
        zt = truth(w, u_true, c, idx, fixed)
        # sanity: the true assignment satisfies the system
        for r, bb in zip(A, b):
            assert sum(x * y for x, y in zip(r, zt)) == bb, "system build is wrong"
        done += 1
        sols, red, Nn, Mm = embed.solve(A, b, use_bkz=use_bkz)
        d = embed.diagnostics(red, Nn, Mm)
        diag_acc.append(d)
        found = False
        for z in sols:
            uu = 0
            for j in range(w):
                if j in fixed:
                    uu |= fixed[j] << j
                else:
                    uu |= z[idx[('u', j)]] << j
            if ((f['s0'](uu) - uu) & M) == c:
                found = True
        ok += found
        if verbose and done <= 2:
            print(f"   [w={w} kh={known_high}] dimL={d['dim_L']} dimL0={d['dim_L0']} "
                  f"|t|={d['target_norm']:.2f} gh={d.get('gh_L0', float('nan')):.2f} "
                  f"ratio={d.get('ratio', float('nan')):.2f} "
                  f"log2#pts={d.get('log2_count', float('nan')):.1f} "
                  f"L0shortest={d.get('L0_shortest')} found={found}")
    el = time.time() - t0
    agg = dict(w=w, enc=enc, known_high=known_high, trials=trials, success=ok,
               rate=ok / trials, secs=el,
               dim_L=diag_acc[0]['dim_L'], dim_L0=diag_acc[0]['dim_L0'],
               target_norm=diag_acc[0]['target_norm'],
               gh=float(np.mean([d.get('gh_L0', np.nan) for d in diag_acc])),
               ratio=float(np.mean([d.get('ratio', np.nan) for d in diag_acc])),
               log2_count=float(np.mean([d.get('log2_count', np.nan) for d in diag_acc])),
               L0_shortest=float(np.mean([d.get('L0_shortest', np.nan) for d in diag_acc])))
    return agg


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    res = []
    if mode == "main":
        for w in (8, 12, 16, 32):
            tr = 50 if w <= 16 else 20
            a = run(w, trials=tr, verbose=True)
            print(json.dumps(a)); res.append(a)
    elif mode == "density":
        w = int(sys.argv[2])
        for kh in range(0, w):
            a = run(w, trials=30, known_high=kh, verbose=(kh in (0, w // 2)))
            print(json.dumps(a)); res.append(a)
            if a['rate'] == 1.0:
                break
    elif mode == "bkz":
        for w in (8, 12, 16):
            a = run(w, trials=20, use_bkz=True, verbose=True)
            print(json.dumps(a)); res.append(a)
    with open(f"e2_{mode}.json", "w") as fh:
        json.dump(res, fh, indent=1)
