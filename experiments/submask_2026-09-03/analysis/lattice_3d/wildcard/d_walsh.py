#!/usr/bin/env python3
"""(d) EXACT best linear approximation of c3 on a 20-variable subcube.

Random-approximation scans can only bound the bias from below.  Here the full
Walsh-Hadamard spectrum is computed exactly over a 2^20 subcube of the input
(20 chosen unknown bits varying, the rest fixed), for every output bit of c3
and for random output masks.  The exact maximum over all 2^20 input masks is
reported and compared with the expected maximum of a random Boolean function.
"""
import numpy as np
from alg import Alg, family_instance

W = 32
A = Alg(W)
NB = 20


def fwht(a):
    """in-place fast Walsh-Hadamard on axis 0, length 2^n."""
    h = 1
    n = a.shape[0]
    while h < n:
        a = a.reshape(-1, 2 * h, *a.shape[1:])
        x = a[:, :h].copy()
        y = a[:, h:].copy()
        a[:, :h] = x + y
        a[:, h:] = x - y
        a = a.reshape(n, *a.shape[2:])
        h *= 2
    return a


def run(inst, bitsel, seed=0):
    """bitsel: list of (word k, bit b) - the NB varying input bits."""
    rng = np.random.default_rng(seed)
    n = 1 << NB
    idx = np.arange(n, dtype=np.uint64)
    base = [np.uint64(int(rng.integers(0, 1 << W))) for _ in range(4)]
    vals = [np.full(n, base[k], dtype=np.uint64) for k in range(4)]
    for t, (k, b) in enumerate(bitsel):
        m = np.uint64(1) << np.uint64(b)
        vals[k] = (vals[k] & ~m) | (((idx >> np.uint64(t)) & np.uint64(1)) << np.uint64(b))
    c3 = inst.residuals(*vals)[3]
    res = []
    # per output bit
    for ob in range(W):
        f = ((c3 >> np.uint64(ob)) & np.uint64(1)).astype(np.int32)
        s = 1 - 2 * f
        s = fwht(s.astype(np.float64))
        s[0] = 0.0                      # drop the trivial (all-zero mask) term
        res.append(np.abs(s).max() / n)
    # random output masks
    rmask = []
    for _ in range(64):
        om = np.uint64(int(rng.integers(1, 1 << W)))
        f = np.array([bin(int(x) & int(om)).count('1') & 1 for x in c3], dtype=np.int32)
        s = fwht((1 - 2 * f).astype(np.float64))
        s[0] = 0.0
        rmask.append(np.abs(s).max() / n)
    return np.array(res), np.array(rmask)


if __name__ == "__main__":
    rng = np.random.default_rng(99)
    n = 1 << NB
    # expected max |correlation| of a random balanced function over 2^20-1 masks
    exp_max = np.sqrt(2 * np.log(n) / n)
    print(f"subcube 2^{NB} = {n}; expected max |corr| for a random function "
          f"~ sqrt(2 ln 2^{NB} / 2^{NB}) = {exp_max:.5f}")
    for tag, sel in (
        ("20 low bits of a3", [(3, b) for b in range(20)]),
        ("5 low bits of each of a0..a3", [(k, b) for k in range(4) for b in range(5)]),
        ("20 bits spread over a0..a3", [(k, b) for k in range(4)
                                        for b in (0, 7, 13, 19, 27)]),
    ):
        allmax = []
        allr = []
        for t in range(3):
            inst, v = family_instance(A, rng)
            r, rm = run(inst, sel, seed=t)
            allmax.append(r)
            allr.append(rm)
        r = np.concatenate(allmax)
        rm = np.concatenate(allr)
        print(f"  {tag:32s}: per-output-bit exact max |corr| "
              f"min {r.min():.5f} mean {r.mean():.5f} max {r.max():.5f}; "
              f"random output masks max {rm.max():.5f}")
