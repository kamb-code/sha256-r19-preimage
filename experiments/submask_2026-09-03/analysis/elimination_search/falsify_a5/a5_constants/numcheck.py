#!/usr/bin/env python3
"""Numeric perturbation test of a context configuration (frame_search2 style).

For a configuration `defs` (same format as symeng.build) and an absorbed word
w = a_wi, on random digests (planted 20-round messages, backward chain gives
a12..a19) and random free context words:
  (1) per message word W_r, r in 0..19, classify numerically how W_r moves with
      w: none / linear with coefficient c (W_r(w+d) - W_r(w) == c*d for random d)
      / heavy;  compared with the symbolic engine's classification.
  (2) per constraint target T_j = W_{16+j} - s1(W_{14+j}) - W_{9+j} - W_j
      (the right-hand side the lookup needs), the same classification and the
      mean Hamming weight of the change of T_j when one random bit of w flips
      (avalanche, as in the barrier note's edge_weights.py).
"""
import sys, struct, ast
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, digest, forward,
                            recover_W, backward_chain)
import symeng as E

R = 20


def realise(defs, free, bc, rng):
    """Numeric context a4..a11 from definitions; `free` gives values of free words."""
    a = {i: bc[i] for i in range(12, R)}
    a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
    pend = {i: defs.get(i, ('free',)) for i in range(4, 12)}
    done = set()
    for i, d in pend.items():
        if d[0] == 'free':
            a[i] = free[i]; done.add(i)
    while len(done) < 8:
        prog = False
        for i, d in pend.items():
            if i in done: continue
            k = d[0]
            if k in ('eq', 'neq'): deps = [d[1]]
            elif k == 'const': deps = []
            elif k == 'sat_off': deps = [i - 1, i - 2, i - 3]
            elif k == 'sat_r': deps = [i - 4, i - 1, i - 2, i - 3]
            elif k == 'sat_h': r = i + 4; deps = [r, r - 1, r - 2, r - 3]
            elif k == 'sat_s0': r = i + 1; deps = [r - 4, r, i - 1, i - 2]
            if not all(j in a for j in deps): continue
            if k == 'const': a[i] = d[1] & M
            elif k == 'sat_off': a[i] = (d[1] + T2(a[i - 1], a[i - 2], a[i - 3])) & M
            elif k == 'eq': a[i] = a[d[1]]
            elif k == 'neq': a[i] = (M - a[d[1]]) & M
            elif k == 'sat_r':
                r = i; a[i] = (d[1] - a[r - 4] + T2(a[r - 1], a[r - 2], a[r - 3])) & M
            elif k == 'sat_h':
                r = i + 4; a[i] = (d[1] - a[r] + T2(a[r - 1], a[r - 2], a[r - 3])) & M
            elif k == 'sat_s0':
                r = i + 1
                y = (a[r - 4] + a[r] - d[1]) & M
                a[i] = invert_s0maj(y, a[i - 1], a[i - 2])
                if a[i] is None: return None
            done.add(i); prog = True
        if not prog: raise RuntimeError("cyclic")
    return a


def invert_s0maj(y, b, c):
    """x with Sigma0(x) + Maj(x,b,c) == y, by chunked brute force (numpy). None if no root."""
    U = np.uint32
    for lo in range(0, 1 << 32, 1 << 26):
        x = np.arange(lo, lo + (1 << 26), dtype=np.uint64).astype(U)
        v = (S0(x) + Maj(x, U(b), U(c))) & U(M)
        hit = np.nonzero(v == U(y))[0]
        if hit.size: return int(x[hit[0]])
    return None


def state_words(a):
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(R):
        e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M
    W = {r: recover_W(a, e, r) for r in range(R)}
    T = [(W[16 + j] - s1(W[14 + j]) - W[9 + j] - W[j]) & M for j in range(4)]
    return e, W, T


def run(defs, wi, trials=40, seed=1, verbose=True, allbits=False):
    rng = np.random.default_rng(seed)
    w = f'a{wi}'
    S = E.build(defs)
    symW = {r: E.classify(S['W'][r], w) for r in range(R)}
    symC = [E.classify(S['C'][j], w) for j in range(4)]
    numW = {r: set() for r in range(R)}
    numT = [set() for _ in range(4)]
    ham = np.zeros(4); nham = 0
    used = 0
    for t in range(trials):
        msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
        pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
        Wt = [struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)]
        bc, _ = backward_chain(digest(Wt, R), R)
        free = {i: int(rng.integers(0, 1 << 32, dtype=np.uint64)) for i in range(4, 12)}
        unk = {i: int(rng.integers(0, 1 << 32, dtype=np.uint64)) for i in range(4)}
        a0 = realise(defs, free, bc, rng)
        if a0 is None: continue
        a0.update(unk)
        used += 1
        e0, W0, T0 = state_words(a0)
        # random additive perturbation of w, then one-bit flips for avalanche
        d = int(rng.integers(1, 1 << 32, dtype=np.uint64))
        free2 = dict(free); free2[wi] = (free[wi] + d) & M
        a1 = realise(defs, free2, bc, rng)
        if a1 is None: continue
        a1.update(unk)
        e1, W1, T1 = state_words(a1)
        for r in range(R):
            diff = (W1[r] - W0[r]) & M
            if diff == 0: numW[r].add('none')
            else:
                # linear with small integer coefficient?
                cf = None
                for c in (1, 2, 3, M, M - 1, M - 2):
                    if (c * d) & M == diff: cf = c if c < 8 else c - (1 << 32); break
                numW[r].add(f'lin{cf:+d}' if cf is not None else 'heavy')
        for j in range(4):
            diff = (T1[j] - T0[j]) & M
            if diff == 0: numT[j].add('none')
            else:
                cf = None
                for c in (1, 2, 3, M, M - 1, M - 2):
                    if (c * d) & M == diff: cf = c if c < 8 else c - (1 << 32); break
                numT[j].add(f'lin{cf:+d}' if cf is not None else 'heavy')
        for bit in (range(32) if allbits else [int(rng.integers(0, 32)) for _ in range(5)]):
            free3 = dict(free); free3[wi] = free[wi] ^ (1 << bit)
            a3 = realise(defs, free3, bc, rng)
            if a3 is None: continue
            # legality: no condition may reference the unknown w, i.e. the other context
            # words must not move when w is flipped
            if any(a3[i] != a0[i] for i in range(4, 12) if i != wi):
                raise RuntimeError(f"illegal configuration: context word moves with a{wi}: {defs}")
            a3.update(unk)
            _, _, T3 = state_words(a3)
            for j in range(4):
                ham[j] += bin((T3[j] ^ T0[j]) & M).count('1')
            nham += 1
    if verbose:
        print(f"defs = {defs}   absorbed w = {w}   ({used} instances)")
        print("  word  symbolic   numeric")
        for r in range(R):
            if symW[r] == 'none' and numW[r] == {'none'}: continue
            print(f"  W{r:<3}  {symW[r]:<9}  {sorted(numW[r])}")
        print("  target  symbolic   numeric        avalanche bits/flip")
        for j in range(4):
            print(f"  C{j}      {symC[j]:<9}  {str(sorted(numT[j])):<14} {ham[j]/max(nham,1):5.2f}")
    return symC, numT, ham / max(nham, 1)


if __name__ == "__main__":
    wi = int(sys.argv[1].lstrip('a'))
    defs = ast.literal_eval(sys.argv[2]) if len(sys.argv) > 2 else {5: ('eq', 4), 8: ('sat_r', M), 9: ('sat_r', M)}
    run(defs, wi)
