"""Reuse cost at integer m, for the published frame (j = 3 lookups)."""
from math import log2, lgamma
c3 = 0.9337; pc = 0.75 ** 32
P = c3 * pc


def Xj(n, m):
    if m <= 1:
        return 0.0
    return 2.0 ** (n * (1 - 1 / m) + lgamma(m + 1) / (m * 0.6931471805599453) - log2(m))


for R, S, base in ((20, 2.0 ** 32, 45.38), (21, 2.0 ** 32, 77.38)):
    PP = P if R == 20 else P * 2.0 ** -32       # extra 32-bit filter at R=21
    print(f"R={R}: baseline 2^{base:.2f}; P(context solvable) = 2^{log2(PP):.2f}; "
          f"one sweep = 2^{log2(S):.0f}; signature = 96 bits (KC0,KC1,KC2)")
    print(f"   {'m':>10} {'find fibre':>12} {'per member':>11} {'cost/preimage':>14} {'delta':>8}")
    for lm in (0, 1, 2, 3, 10, 20, 32):
        m = 2.0 ** lm
        X = max(Xj(96, m), 1.0) if m > 1 else 0.0
        N = X * m
        cost = (S / m + X) / PP
        print(f"   2^{lm:<8d} 2^{log2(max(N,1)):10.2f} 2^{log2(max(X,1)):9.2f} "
              f"2^{log2(cost):12.2f} {log2(cost)-base:+7.2f}")
    mm = 2.0 ** (32 if R == 20 else 64)
    ideal = (S / mm + 1.0) / PP
    print(f"   IDEAL (X = 1 op per fibre member, m = full fibre 2^{32 if R==20 else 64}): "
          f"2^{log2(ideal):.2f}  ({base - log2(ideal):.1f} bits of gain -- the prize)")
    print(f"   break-even X: 2^{log2(2.0**base * PP):.2f}   "
          f"cheapest achievable X (m=2 birthday): 2^{log2(Xj(96,2)):.2f}   "
          f"deficit {log2(Xj(96,2)) - log2(2.0**base * PP):.1f} bits")
    print()
