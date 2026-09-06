#!/usr/bin/env python3
"""Best-possible non-strict branch: e9 saturated through a9 (a9 = c' - a5), with every
other route of a9 into C1 collapsed:  a10 = a8 kills Maj(a10,a9,a8) in e11 (under s1(W15)),
e14 = 0 via a10 kills Ch(e14,e13,e12) -> e12 in W15, e8 solved through a4 (a7 = a6 so
a4 is a5-free).  Remaining: kernel S0(c'-a5) - S0(a5), +a5 from W17 (e13), Maj terms."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tie_scan as T
from probe import residual
M = 0xFFFFFFFF
rng = np.random.default_rng(3)
cfgs = [
 ({7: ('eq', 6), 8: ('sat_r', M), 9: ('sat_r', M)}, "a7=a6, e8=e9=-1 (baseline non-strict)"),
 ({7: ('eq', 6), 8: ('sat_r', M), 9: ('sat_r', M), 10: ('eq', 8)}, "+ a10=a8"),
 ({7: ('eq', 6), 10: ('sat_h', 0), 8: ('eq', 10), 4: ('sat_h', M), 9: ('sat_r', M)}, "a7=a6, e14=0 via a10, a8=a10, e8=-1 via a4, e9=-1 via a9"),
 ({7: ('eq', 6), 10: ('sat_h', 0), 8: ('eq', 10), 4: ('sat_h', M), 9: ('sat_r', 0)}, "same with e9=0"),
 ({7: ('eq', 6), 10: ('sat_h', 0), 8: ('eq', 10), 4: ('sat_h', M), 9: ('neq', 5)}, "same with a9=~a5 (c'=-1: kernel -1-2*S0(a5))"),
 ({7: ('eq', 6), 10: ('sat_h', 0), 8: ('eq', 10), 4: ('sat_h', 0), 9: ('neq', 5)}, "a9=~a5 variant, e8=0 via a4"),
]
for d, lab in cfgs:
    residual(d, lab)
    res = T.measure(d, N=200, seed=1, words=(5,))
    print("      full a5 row:", T.fmt(res, 5))
# refuter: different seeds for the best one
d = cfgs[2][0]
for s in (11, 12, 13):
    res = T.measure(d, N=200, seed=s, words=(5,))
    print(f"   seed {s}:", T.fmt(res, 5))
