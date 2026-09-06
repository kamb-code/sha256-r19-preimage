#!/usr/bin/env python3
"""Edge-weight scan of a5 -> C1 under rotational / linear ties among context words.

Extends the `defs` grammar of elimination_search/a10/numcheck.py (realise) with
new condition types and vectorises the 200-state x 32-flip measurement.

Frame: a5 is the unknown to be absorbed.  a4, a6, a7, a8..a11 are context, a12..a19
digest (random; equivalent to a random digest through the backward chain), a0..a3
random unknowns.  No condition may name a5 explicitly; a saturation e9 = t solved
through a9 makes a9 depend on a5 implicitly (as in the earlier enumerations), and
such configurations are flagged strict=False.

defs: dict i in 4..11 -> tuple
  ('free',)                      free random context word
  ('eq', j) / ('neq', j)         a_i = a_j / ~a_j
  ('rotr', j, n)                 a_i = ROTR^n(a_j)
  ('xor', j, c)                  a_i = a_j ^ c
  ('add', j, c)                  a_i = a_j + c
  ('S0add', j, c)                a_i = Sigma0(a_j) + c
  ('S1add', j, c)                a_i = Sigma1(a_j) + c
  ('maj', j, c1, c2)             a_i = Maj(a_j, c1, c2)
  ('lin', j, (n1, n2, ...))      a_i = XOR_k ROTR^{n_k}(a_j)   (GF(2)-linear tie)
  ('sat_r', t)                   e_i = t solved through a_i
  ('sat_h', t)                   e_{i+4} = t solved through a_i

Edge weight of word w into target T_j = W_{16+j} - s1(W_{14+j}) - W_{9+j} - W_j
(the a_{j+1}-free lookup index; a5, a4 do not enter W_1..W_4 so it coincides with
the full constraint value up to the s0 term): mean Hamming weight of the change of
T_j when one bit of w flips, over N states x 32 bit positions, the context being
re-realised after the flip.
"""
import sys
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, rotr

R = 20
U = np.uint32
MM = U(M)


def u(x):
    return U(x & M)


def deps_of(i, d):
    k = d[0]
    if k == 'free':
        return []
    if k in ('eq', 'neq', 'rotr', 'xor', 'add', 'S0add', 'S1add', 'maj', 'lin'):
        return [d[1]]
    if k == 'sat_r':
        return [i - 4, i - 1, i - 2, i - 3]
    if k == 'sat_h':
        r = i + 4
        return [r, r - 1, r - 2, r - 3]
    raise ValueError(d)


def order_of(defs):
    """Topological order of the context words 4..11 (a5 is given, not defined)."""
    pend = {i: defs.get(i, ('free',)) for i in range(4, 12) if i != 5}
    done = set()
    order = []
    while len(done) < len(pend):
        prog = False
        for i, d in pend.items():
            if i in done:
                continue
            if all((j in done) or (j == 5) or j < 4 or j >= 12 for j in deps_of(i, d)):
                order.append(i)
                done.add(i)
                prog = True
        if not prog:
            return None
    return order


def realise(defs, order, a, free):
    """Fill a[4..11] (uint32 arrays) from definitions; a already holds a5, a0..a3,
    a12..a19 and the IV words; `free` holds the random values of the free words."""
    for i in order:
        d = defs.get(i, ('free',))
        k = d[0]
        if k == 'free':
            a[i] = free[i]
        elif k == 'eq':
            a[i] = a[d[1]]
        elif k == 'neq':
            a[i] = ~a[d[1]]
        elif k == 'rotr':
            a[i] = rotr(a[d[1]], d[2]) if d[2] else a[d[1]]
        elif k == 'xor':
            a[i] = a[d[1]] ^ u(d[2])
        elif k == 'add':
            a[i] = a[d[1]] + u(d[2])
        elif k == 'S0add':
            a[i] = S0(a[d[1]]) + u(d[2])
        elif k == 'S1add':
            a[i] = S1(a[d[1]]) + u(d[2])
        elif k == 'maj':
            a[i] = Maj(a[d[1]], u(d[2]), u(d[3]))
        elif k == 'lin':
            x = a[d[1]]
            y = np.zeros_like(x)
            for n in d[2]:
                y ^= rotr(x, n) if n else x
            a[i] = y
        elif k == 'sat_r':
            r = i
            a[i] = u(d[1]) - a[r - 4] + T2(a[r - 1], a[r - 2], a[r - 3])
        elif k == 'sat_h':
            r = i + 4
            a[i] = u(d[1]) - a[r] + T2(a[r - 1], a[r - 2], a[r - 3])
        else:
            raise ValueError(d)
    return a


def targets(a):
    n = a[0].size
    e = {-1: np.full(n, IV[4], U), -2: np.full(n, IV[5], U), -3: np.full(n, IV[6], U), -4: np.full(n, IV[7], U)}
    for r in range(R):
        e[r] = a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])
    W = {}
    for r in range(R):
        W[r] = (a[r] - T2(a[r - 1], a[r - 2], a[r - 3]) - e[r - 4] - S1(e[r - 1])
                - Ch(e[r - 1], e[r - 2], e[r - 3]) - u(K[r]))
    T = [W[16 + j] - s1(W[14 + j]) - W[9 + j] - W[j] for j in range(4)]
    return e, W, T


def popcount(x):
    x = x.astype(np.uint32)
    x = x - ((x >> U(1)) & U(0x55555555))
    x = (x & U(0x33333333)) + ((x >> U(2)) & U(0x33333333))
    x = (x + (x >> U(4))) & U(0x0F0F0F0F)
    return ((x * U(0x01010101)) >> U(24)).astype(np.int64)


def base_state(rng, N):
    def rnd():
        return rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(U)
    a = {-1: np.full(N, IV[0], U), -2: np.full(N, IV[1], U), -3: np.full(N, IV[2], U), -4: np.full(N, IV[3], U)}
    for i in range(4):
        a[i] = rnd()
    a[5] = rnd()
    for i in range(12, R):
        a[i] = rnd()
    free = {i: rnd() for i in range(4, 12) if i != 5}
    return a, free


def expand(a, free, reps):
    """Tile every array `reps` times (for the 32 flips)."""
    a2 = {k: (np.tile(v, reps) if isinstance(v, np.ndarray) else v) for k, v in a.items()}
    f2 = {k: np.tile(v, reps) for k, v in free.items()}
    return a2, f2


def measure(defs, N=200, seed=1, words=(5, 4)):
    """Return dict: for each flipped word w in `words`, mean Hamming weight of the
    change of T_0..T_3 per flipped bit (all 32 bits x N states), the fraction of
    flips that move each target, and whether the context stayed independent of w."""
    order = order_of(defs)
    if order is None:
        return None
    rng = np.random.default_rng(seed)
    a, free = base_state(rng, N)
    a = realise(defs, order, dict(a), free)
    _, _, T0 = targets(a)
    out = {}
    for w in words:
        aa, ff = expand(a, free, 32)
        bits = np.repeat(np.arange(32, dtype=np.uint32), N)
        mask = (U(1) << bits).astype(U)
        if w == 5:
            aa[5] = aa[5] ^ mask
        else:
            # w is a context word: flip it as a free word (must be free in defs)
            if defs.get(w, ('free',))[0] != 'free':
                out[w] = None
                continue
            ff[w] = ff[w] ^ mask
        aa = realise(defs, order, aa, ff)
        ctx_moved = any(bool(np.any(aa[i] != np.tile(a[i], 32))) for i in range(4, 12) if i != w)
        _, _, T1 = targets(aa)
        hw = [popcount(T1[j] ^ np.tile(T0[j], 32)) for j in range(4)]
        out[w] = dict(mean=[float(h.mean()) for h in hw],
                      moved=[float((h > 0).mean()) for h in hw],
                      min=[int(h.min()) for h in hw],
                      max=[int(h.max()) for h in hw],
                      ctx_dep=ctx_moved)
    return out


def fmt(res, w):
    r = res[w]
    if r is None:
        return f"a{w}: (not free)"
    return (f"a{w}: " + " ".join(f"C{j}={r['mean'][j]:5.2f}" for j in range(4))
            + f"  moved=" + "/".join(f"{m:.2f}" for m in r['moved'])
            + ("  ctx-depends-on-a%d" % w if r['ctx_dep'] else ""))


if __name__ == "__main__":
    import ast
    defs = ast.literal_eval(sys.argv[1]) if len(sys.argv) > 1 else {7: ('eq', 6), 8: ('sat_r', M)}
    res = measure(defs)
    print("defs =", defs)
    print(fmt(res, 5))
    print(fmt(res, 4))
