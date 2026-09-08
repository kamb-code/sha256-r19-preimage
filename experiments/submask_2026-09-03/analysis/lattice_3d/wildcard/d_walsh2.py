#!/usr/bin/env python3
"""Localise the linear structure found by d_walsh.py: which output bits of c3
carry it, how strong is the best mask, and is it the low-bit carry linearity of
W3 = a3 + F23?  A random-function control is run on the same subcube."""
import numpy as np
from alg import Alg, family_instance

W = 32
A = Alg(W)
NB = 20


def fwht(a):
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


rng = np.random.default_rng(4242)
n = 1 << NB
idx = np.arange(n, dtype=np.uint64)
exp_max = np.sqrt(2 * np.log(n) / n)
prof = np.zeros(W)
bestmask = [0] * W
ninst = 4
for t in range(ninst):
    inst, v = family_instance(A, rng)
    base = [np.uint64(int(rng.integers(0, 1 << W))) for _ in range(4)]
    A3 = (base[3] & np.uint64(0xFFFFF00000)) | idx        # low 20 bits vary
    A3 = (base[3] & ~np.uint64(0xFFFFF)) | idx
    vals = [np.full(n, base[k], dtype=np.uint64) for k in range(3)] + [A3]
    c3 = inst.residuals(*vals)[3]
    for ob in range(W):
        f = ((c3 >> np.uint64(ob)) & np.uint64(1)).astype(np.int32)
        s = fwht((1 - 2 * f).astype(np.float64))
        s[0] = 0.0
        j = int(np.argmax(np.abs(s)))
        prof[ob] += abs(s[j]) / n
        if t == 0:
            bestmask[ob] = j
prof /= ninst
print(f"exact max |corr| over all 2^{NB}-1 input masks, per output bit of c3, "
      f"mean of {ninst} instances")
print(f"noise floor (expected max for a random function) = {exp_max:.5f}")
for ob in range(W):
    bar = '#' * int(prof[ob] / 0.004)
    print(f"  c3 bit {ob:2d}: {prof[ob]:.5f}  ({prof[ob]/exp_max:5.1f}x floor) "
          f"best mask 0x{bestmask[ob]:05x}  {bar}")

# control: a genuinely random function on the same subcube
ctrl = []
for t in range(ninst):
    r = rng.integers(0, 1 << W, n, dtype=np.uint64)
    for ob in range(0, W, 8):
        f = ((r >> np.uint64(ob)) & np.uint64(1)).astype(np.int32)
        s = fwht((1 - 2 * f).astype(np.float64))
        s[0] = 0.0
        ctrl.append(np.abs(s).max() / n)
print(f"\ncontrol (uniform random words, same subcube): max |corr| "
      f"mean {np.mean(ctrl):.5f} max {np.max(ctrl):.5f}")

# what a bias eps buys: the filter c3 == 0 needs all 32 bits; predicting one
# linear combination with correlation eps saves at most 1 - H2((1+eps)/2) bits
def bits_saved(eps):
    p = (1 + eps) / 2
    import math
    h = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    return 1 - h
print("\ninformation in the best approximations:")
for ob in (0, 1, 2, 3):
    print(f"  c3 bit {ob}: corr {prof[ob]:.4f} -> {bits_saved(prof[ob]):.5f} bits")
print(f"  sum over all 32 output bits (an upper bound that ignores that the "
      f"masks are not independent): {sum(bits_saved(p) for p in prof):.4f} bits")
