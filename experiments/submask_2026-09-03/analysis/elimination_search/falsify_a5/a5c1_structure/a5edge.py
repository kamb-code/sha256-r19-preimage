#!/usr/bin/env python3
"""a5 -> C1 edge under an EXTENDED context-condition menu, R = 20.

Extends elimination_search/a10/numcheck.py (same `defs` format, same
targets T_j = W[16+j] - s1(W[14+j]) - W[9+j] - W[j]) in three ways:
  * vectorised over states, so 200 states x 32 single-bit flips is one pass;
  * new condition kinds
        ('rot', j, n)     a_i = rotr(a_j, n)
        ('rotneq', j, n)  a_i = rotr(~a_j, n)
        ('xor', j, c)     a_i = a_j ^ c
        ('add', j, c)     a_i = a_j + c
        ('sat_r', t)      e_i = t, any t (numcheck already allowed any t)
        ('sat_h', t)      e_{i+4} = t, solved for a_i
        ('hoff', t)       e_i - a_{i-4} = t, solved for a_i  (e.g. e9 = a5 + t)
  * a numerical legality test: the realised context (every word except the
    unknown) must be invariant when the unknown changes.
The unknown is a5 (never referenced by a condition).  a4, a6..a11 are context;
a12..a19 come from the backward chain of a random digest; a0..a3 random.
"""
import sys, struct
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, backward_chain

R = 20
U = np.uint32
MM = U(M)


def rotr_a(x, n):
    n = int(n) % 32
    if n == 0:
        return x
    return ((x >> U(n)) | (x << U(32 - n))) & MM


def rand_words(rng, n):
    return rng.integers(0, 1 << 32, size=n, dtype=np.uint64).astype(U)


def cst(v, n):
    return np.full(n, v & M, dtype=U)


def realise(defs, free, n, unknown=5):
    """Vectorised numcheck.realise.  free: dict i -> array(n) for free words
    (and the unknown).  Returns dict i -> array(n) for i in 4..11, or None if
    the definitions are cyclic."""
    a = {}
    pend = {i: defs.get(i, ('free',)) for i in range(4, 12)}
    if unknown in defs:
        raise ValueError("the unknown may not carry a condition")
    done = set()
    for i, d in pend.items():
        if d[0] == 'free':
            a[i] = free[i]
            done.add(i)
    # digest words are needed by sat kinds with r >= 12
    for i in range(12, R):
        a[i] = free[i]
    while len(done) < 8:
        prog = False
        for i, d in pend.items():
            if i in done:
                continue
            k = d[0]
            if k in ('eq', 'neq', 'rot', 'rotneq', 'xor', 'add'):
                deps = [d[1]]
            elif k == 'sat_r':
                deps = [i - 4, i - 1, i - 2, i - 3]
            elif k == 'hoff':
                deps = [i - 1, i - 2, i - 3]
            elif k == 'sat_h':
                r = i + 4
                deps = [r, r - 1, r - 2, r - 3]
            else:
                raise ValueError(k)
            if not all((j in a) for j in deps):
                continue
            if k == 'eq':
                a[i] = a[d[1]].copy()
            elif k == 'neq':
                a[i] = ~a[d[1]]
            elif k == 'rot':
                a[i] = rotr_a(a[d[1]], d[2])
            elif k == 'rotneq':
                a[i] = rotr_a(~a[d[1]], d[2])
            elif k == 'xor':
                a[i] = a[d[1]] ^ U(d[2] & M)
            elif k == 'add':
                a[i] = (a[d[1]] + U(d[2] & M)) & MM
            elif k == 'sat_r':
                r = i
                a[i] = (U(d[1] & M) - a[r - 4] + T2(a[r - 1], a[r - 2], a[r - 3])) & MM
            elif k == 'hoff':
                r = i
                a[i] = (U(d[1] & M) + T2(a[r - 1], a[r - 2], a[r - 3])) & MM
            elif k == 'sat_h':
                r = i + 4
                a[i] = (U(d[1] & M) - a[r] + T2(a[r - 1], a[r - 2], a[r - 3])) & MM
            done.add(i)
            prog = True
        if not prog:
            return None
    return {i: a[i] for i in range(4, 12)}


def targets(a):
    """a: dict -4..19 -> uint32 arrays of one common shape.  Returns e, W, T
    with T[j] the four constraint targets and T[4] the exact C1 table index
    (T1 - W2) and T[5] the exact C0 table index (T0 - W1)."""
    # IV e-words: e_{-1}=IV[4], e_{-2}=IV[5], e_{-3}=IV[6], e_{-4}=IV[7]
    e = {}
    n = a[0].shape[0]
    e[-1], e[-2], e[-3], e[-4] = cst(IV[4], n), cst(IV[5], n), cst(IV[6], n), cst(IV[7], n)
    for r in range(R):
        e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & MM
    W = {}
    for r in range(R):
        W[r] = (a[r] - T2(a[r - 1], a[r - 2], a[r - 3]) - e[r - 4] - S1(e[r - 1])
                - Ch(e[r - 1], e[r - 2], e[r - 3]) - U(K[r])) & MM
    T = [(W[16 + j] - s1(W[14 + j]) - W[9 + j] - W[j]) & MM for j in range(4)]
    T.append((T[1] - W[2]) & MM)   # exact C1 lookup index
    T.append((T[0] - W[1]) & MM)   # exact C0 lookup index
    return e, W, T


def make_states(rng, n):
    """Random digests (backward chain) + random a0..a3; returns dict of arrays."""
    st = {}
    dig = np.zeros((8, n), dtype=np.uint64)
    A = {i: np.zeros(n, dtype=U) for i in range(12, R)}
    for t in range(n):
        h = bytes(rng.integers(0, 256, 32, dtype=np.uint8).tolist())
        ab, _ = backward_chain(h, R)
        for i in range(12, R):
            A[i][t] = ab[i]
    st.update(A)
    for i in range(4):
        st[i] = rand_words(rng, n)
    for i in (-1, -2, -3, -4):
        st[i] = cst(IV[-i - 1], n)
    return st


POPCNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def popcount(x):
    x = x.astype(np.uint32)
    return (POPCNT[x & 0xFF] + POPCNT[(x >> 8) & 0xFF] + POPCNT[(x >> 16) & 0xFF]
            + POPCNT[(x >> 24) & 0xFF]).astype(np.int64)


def legal(defs, rng, n=16, unknown=5):
    """Context words must be invariant under changes of the unknown and the
    definitions must be acyclic.  Returns True/False."""
    st = make_states(rng, n)
    free = {i: rand_words(rng, n) for i in range(4, 12)}
    free.update({i: st[i] for i in range(12, R)})
    c0 = realise(defs, free, n, unknown)
    if c0 is None:
        return False
    free2 = dict(free)
    free2[unknown] = rand_words(rng, n)
    c1 = realise(defs, free2, n, unknown)
    for i in range(4, 12):
        if i == unknown:
            continue
        if not np.array_equal(c0[i], c1[i]):
            return False
    return True


def edge_weights(defs, rng, n=200, unknown=5, flip=None, return_T=False):
    """Mean Hamming weight of dT_j per flipped bit of `flip` (default: the
    unknown), over n states and all 32 bits, with the context re-realised after
    the flip (as in numcheck).  Returns array of 6 weights [C0,C1,C2,C3,idxC1,idxC0]
    and the fraction of (state,bit) pairs in which each target moved."""
    if flip is None:
        flip = unknown
    st = make_states(rng, n)
    free = {i: rand_words(rng, n) for i in range(4, 12)}
    free.update({i: st[i] for i in range(12, R)})
    # 33 copies: row 0 unflipped, rows 1..32 flip bit b-1 of `flip`
    def stack(x):
        return np.tile(x, 33)
    big = {}
    for i in range(4, 12):
        big[i] = stack(free[i])
    for i in range(12, R):
        big[i] = stack(st[i])
    bits = np.repeat(np.arange(33, dtype=np.int64), n)
    mask = np.where(bits == 0, 0, (1 << np.maximum(bits - 1, 0))).astype(np.uint64).astype(U)
    big[flip] = big[flip] ^ mask
    ctx = realise(defs, big, 33 * n, unknown)
    if ctx is None:
        return None
    a = {}
    a.update({i: stack(st[i]) for i in range(4)})
    a.update({i: stack(st[i]) for i in (-1, -2, -3, -4)})
    a.update(ctx)
    a.update({i: stack(st[i]) for i in range(12, R)})
    _, _, T = targets(a)
    out = np.zeros(6)
    moved = np.zeros(6)
    for j in range(6):
        Tm = T[j].reshape(33, n)
        d = Tm[1:] ^ Tm[0][None, :]
        pc = popcount(d)
        out[j] = pc.mean()
        moved[j] = (pc > 0).mean()
    if return_T:
        return out, moved, T
    return out, moved


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    # reproduce numcheck's family numbers for a5 with a7=a6, e8=-1
    for defs in ({7: ('eq', 6), 8: ('sat_r', M)},
                 {6: ('neq', 4), 7: ('eq', 6), 8: ('sat_r', M), 10: ('sat_r', M), 11: ('sat_r', 0)},
                 {}):
        print(defs, "legal:", legal(defs, rng))
        w, mv = edge_weights(defs, rng)
        print("  a5 -> [C0 C1 C2 C3 idxC1 idxC0]:", np.round(w, 2), "moved", np.round(mv, 2))
    # a4 control in a random context (frame B of the barrier paper)
    w, mv = edge_weights({}, rng, flip=4, unknown=4)
    print("control a4 -> [C0 C1 C2 C3 idxC1 idxC0] random context:", np.round(w, 2))
