#!/usr/bin/env python3
"""Vectorised single-bit edge weights for the a5-unknown frame at R = 20.

Extends elim/a10/numcheck.py: same defs format plus the kinds
    ('sat_off', t)   e_i = a_{i-4} + t  solved for a_i  (a_i = t + T2(a_{i-1},a_{i-2},a_{i-3}))
    ('rot', j, k)    a_i = rotr(a_j, k)
and the frame rule that a5 is an UNKNOWN: any definition whose dependency chain
reaches a5 is illegal (raises), because a5 is absorbed last (by C3, which needs
a1,a2,a3 through W3 and W4) and so no context word can be a function of it.

Edge weight = mean over N random states and all 32 single-bit flips of a5 of
popcount(T_j(a5) xor T_j(a5 ^ 2^i)),  T_j = W_{16+j} - s1(W_{14+j}) - W_{9+j} - W_j,
which is the quantity the C_j lookup indexes (up to terms free of a5).
"""
import numpy as np
import sys
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import K, IV, S0, S1, s0, s1, Ch, Maj, T2, rotr

U = np.uint32
M = 0xFFFFFFFF
R = 20
DEPS = {
    'free': lambda i, d: [],
    'eq': lambda i, d: [d[1]],
    'neq': lambda i, d: [d[1]],
    'rot': lambda i, d: [d[1]],
    'sat_r': lambda i, d: [i - 4, i - 1, i - 2, i - 3],
    'sat_off': lambda i, d: [i - 1, i - 2, i - 3],
    'sat_h': lambda i, d: [i + 4, i + 3, i + 2, i + 1],
}


class Illegal(Exception):
    pass


_legal_cache = {}


def legal_check(defs, unknown_words=(0, 1, 2, 3, 5)):
    """Raise Illegal unless every context word resolves (symbolically, with the
    Maj/Ch collapses) to an expression free of the unknown words."""
    key = (tuple(sorted(defs.items())), tuple(unknown_words))
    if key in _legal_cache:
        if _legal_cache[key]:
            raise Illegal(_legal_cache[key])
        return
    import symext
    try:
        symext.build_ext(defs, unknown_words=unknown_words, strict=True)
    except symext.Illegal as ex:
        _legal_cache[key] = str(ex)
        raise Illegal(str(ex))
    _legal_cache[key] = None


def realise(defs, free, dig, unk, unknown_words=(0, 1, 2, 3, 5)):
    """Vectorised context.  free/dig/unk: dicts of uint32 arrays (same length)."""
    a = {i: dig[i] for i in range(12, R)}
    a.update({-1: U(IV[0]), -2: U(IV[1]), -3: U(IV[2]), -4: U(IV[3])})
    for i in unknown_words:
        a[i] = unk[i]
    ctx = [i for i in range(4, 12) if i not in unknown_words]
    pend = {i: defs.get(i, ('free',)) for i in ctx}
    # legality: the SYMBOLIC engine decides (it applies the Maj/Ch collapses, so
    # e8 = -1 through a8 is legal when a7 = a6 even though Maj(a7,a6,a5) names a5)
    legal_check(defs, unknown_words)
    done = set()
    for i, d in pend.items():
        if d[0] == 'free':
            a[i] = free[i]; done.add(i)
    while len(done) < len(ctx):
        prog = False
        for i, d in pend.items():
            if i in done: continue
            deps = DEPS[d[0]](i, d)
            if not all((j in a) for j in deps): continue
            k = d[0]
            if k == 'eq': a[i] = a[d[1]]
            elif k == 'neq': a[i] = ~a[d[1]]
            elif k == 'rot': a[i] = rotr(a[d[1]], d[2])
            elif k == 'sat_r':
                r = i; a[i] = (U(d[1]) - a[r - 4] + T2(a[r - 1], a[r - 2], a[r - 3]))
            elif k == 'sat_off':
                r = i; a[i] = (U(d[1]) + T2(a[r - 1], a[r - 2], a[r - 3]))
            elif k == 'sat_h':
                r = i + 4; a[i] = (U(d[1]) - a[r] + T2(a[r - 1], a[r - 2], a[r - 3]))
            done.add(i); prog = True
        if not prog:
            raise Illegal("cyclic")
    return a


def targets(a):
    e = {-1: U(IV[4]), -2: U(IV[5]), -3: U(IV[6]), -4: U(IV[7])}
    for r in range(R):
        e[r] = a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])
    W = {}
    for r in range(R):
        W[r] = (a[r] - T2(a[r - 1], a[r - 2], a[r - 3]) - e[r - 4] - S1(e[r - 1])
                - Ch(e[r - 1], e[r - 2], e[r - 3]) - U(K[r]))
    T = [W[16 + j] - s1(W[14 + j]) - W[9 + j] - W[j] for j in range(4)]
    return e, W, T


def popcount(x):
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0f0f0f0f0f0f0f0f)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def rand_words(rng, n):
    return rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(U)


def edge_weights(defs, n=200, seed=1, flip_word=5, freeze_e=None, unknown_words=(0, 1, 2, 3, 5)):
    """Returns dict j -> (mean, min-over-bits, max-over-bits) of bits changed in T_j
    when one bit of a_{flip_word} flips, context held fixed (the unknown frame).
    freeze_e: optional dict r -> value: DIAGNOSTIC ONLY, overwrite e_r after the
    state is formed (an illegal freeze, used to attribute weight between atoms)."""
    rng = np.random.default_rng(seed)
    free = {i: rand_words(rng, n) for i in range(4, 12)}
    dig = {i: rand_words(rng, n) for i in range(12, R)}
    unk = {i: rand_words(rng, n) for i in unknown_words}
    a0 = realise(defs, free, dig, unk, unknown_words)
    def T_of(a):
        if freeze_e is None:
            return targets(a)[2]
        return targets_frozen(a, freeze_e)
    T0 = T_of(a0)
    per_bit = np.zeros((4, 32))
    for i in range(32):
        a1 = dict(a0)
        a1[flip_word] = a0[flip_word] ^ U(1 << i)
        T1 = T_of(a1)
        for j in range(4):
            per_bit[j, i] = popcount(T0[j] ^ T1[j]).mean()
    return {j: (per_bit[j].mean(), per_bit[j].min(), per_bit[j].max()) for j in range(4)}, per_bit


def targets_frozen(a, freeze_e):
    e = {-1: U(IV[4]), -2: U(IV[5]), -3: U(IV[6]), -4: U(IV[7])}
    for r in range(R):
        e[r] = a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])
    for r, v in freeze_e.items():
        e[r] = np.full_like(e[r], U(v))
    W = {}
    for r in range(R):
        W[r] = (a[r] - T2(a[r - 1], a[r - 2], a[r - 3]) - e[r - 4] - S1(e[r - 1])
                - Ch(e[r - 1], e[r - 2], e[r - 3]) - U(K[r]))
    return [W[16 + j] - s1(W[14 + j]) - W[9 + j] - W[j] for j in range(4)]


def fmt(w):
    return "  ".join(f"C{j}:{w[j][0]:5.2f} [{w[j][1]:4.1f},{w[j][2]:4.1f}]" for j in range(4))


if __name__ == "__main__":
    import ast
    defs = ast.literal_eval(sys.argv[1]) if len(sys.argv) > 1 else {7: ('eq', 6), 8: ('sat_r', M)}
    np.seterr(over='ignore')
    w, _ = edge_weights(defs)
    print("a5 ->", fmt(w))
    w4, _ = edge_weights(defs, flip_word=4)
    print("a4 ->", fmt(w4), "(control, context held fixed)")
