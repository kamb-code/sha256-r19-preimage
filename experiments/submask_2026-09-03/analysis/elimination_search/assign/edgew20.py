#!/usr/bin/env python3
"""Numeric companion to symtrace20: realise a context-condition set on random
20-round instances and measure
  (a) which constraint residuals move with which unknown (perturbation test),
  (b) the mean Hamming weight of dR_j per single-bit flip of a_k (edge weights,
      same method as the barrier note's edge_weights.py),
  (c) for a chosen 'mild' edge set, P(residual part == provisional value).
"""
import sys, struct
import numpy as np
sys.path.insert(0, '/home/administrator/sha/publish/code')
from submask_family import (M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, digest,
                            recover_W, backward_chain, forward)
R = 20


def rand_digest(rng):
    msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
    pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
    return digest([struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)], R)


def realise(a, eqmap, sat, unknown):
    """Apply equalities and saturations to the a-dict in place.  Saturation of e_r
    is solved for a_r or a_{r-4} (linear entries) if free, else reported."""
    for i, j in eqmap.items():
        a[i] = a[j]
    consumed = set(eqmap.keys())
    reps = set(eqmap.values())
    for r in sorted(sat):
        tgt = sat[r]
        cands = [k for k in (r, r - 4, r - 1, r - 2, r - 3)
                 if 0 <= k <= 11 and k not in unknown and k not in consumed and k not in reps]
        if not cands:
            return False
        k = cands[0]
        t2 = T2(a[r - 1], a[r - 2], a[r - 3]) if k not in (r - 1, r - 2, r - 3) else None
        if k == r:
            a[r] = (tgt - a[r - 4] + t2) & M
        elif k == r - 4:
            a[r - 4] = (tgt - a[r] + t2) & M
        else:
            return False       # nonlinear solve not implemented
        consumed.add(k)
    return True


def full_state(a):
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(R):
        e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M
    return e


def residuals(a):
    e = full_state(a)
    W = {r: recover_W(a, e, r) for r in range(R)}
    return [(W[16 + j] - s1(W[14 + j]) - W[9 + j] - s0(W[1 + j]) - W[j]) & M for j in range(4)]


def instance(rng, eqmap, sat, unknown):
    h = rand_digest(rng)
    ab, _ = backward_chain(h, R)
    a = dict(ab)
    a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
    for i in range(12):
        a[i] = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    ok = realise(a, eqmap, sat, unknown)
    if not ok: return None
    # verify saturations
    e = full_state(a)
    for r, t in sat.items():
        assert e[r] == t, (r, hex(e[r]), hex(t))
    for i, j in eqmap.items():
        assert a[i] == a[j]
    return a


def edge_weights(eqmap, sat, unknown, trials=200, seed=1):
    rng = np.random.default_rng(seed)
    unk = sorted(unknown)
    Wt = np.zeros((4, len(unk)))
    Dep = np.zeros((4, len(unk)), dtype=bool)
    n = 0
    while n < trials:
        a = instance(rng, eqmap, sat, unknown)
        if a is None: continue
        n += 1
        r0 = residuals(a)
        for ki, k in enumerate(unk):
            b = dict(a)
            b[k] ^= 1 << int(rng.integers(0, 32))
            r1 = residuals(b)
            for j in range(4):
                d = r0[j] ^ r1[j]
                Wt[j, ki] += bin(d).count('1')
                if d: Dep[j, ki] = True
    return Wt / trials, Dep, unk


def show(name, eqmap, sat, unknown, trials=200):
    Wt, Dep, unk = edge_weights(eqmap, sat, unknown, trials)
    print(f'\n{name}: unknowns {unk}, eq={eqmap}, sat={ {k: hex(v) for k, v in sat.items()} }')
    print('        ' + ''.join(f'{"a"+str(k):>7}' for k in unk))
    for j in range(4):
        print(f'  C{j}:  ' + ''.join(f'{Wt[j, ki]:7.1f}' if Dep[j, ki] else f'{"  .":>7}' for ki in range(len(unk))))
    return Wt, Dep, unk


if __name__ == '__main__':
    show('current family (a4=a5, e8=e9=-1)', {5: 4}, {8: M, 9: M}, {0, 1, 2, 3})
    show('random context', {}, {}, {0, 1, 2, 3})
    show('frame B, random context', {}, {}, {0, 1, 2, 3, 4})
    show('frame B, shifted family a5=a6 e9=e10=-1', {6: 5}, {9: M, 10: M}, {0, 1, 2, 3, 4})
    show('a5 unknown, a6=a7=a4, e8=-1 e10=-1 e11=0', {6: 4, 7: 4}, {8: M, 10: M, 11: 0}, {0, 1, 2, 3, 5})
