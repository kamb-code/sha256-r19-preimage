"""Optimise the whole reuse family: absorb j of the four constraints by table
lookup (j = 3 is the published attack), sweep the remaining unknowns, and reuse
one candidate list across m contexts that share the j-word signature.

  sweep per context        S_j   = 2^(32(4-j))
  P(context yields a hit)  P_j   = c^j * (3/4)^32          [independent of S_j]
  candidate list size      N_j   = S_j * c^j * (3/4)^32    [= storage]
  cost of m contexts in one fibre: an m-way multicollision on n = 32j bits.
      With N samples in 2^n bins the number of m-loaded bins is
      2^n (N/2^n)^m / m!, so N(m) = 2^(n(1-1/m)) (m!)^(1/m), and the amortised
      cost per member is X_j(m) = N(m)/m  (>= 2^(n/2)/sqrt(2), min at m = 2).
  cost per preimage        = (S_j/m + X_j(m)) / P_j
"""
import numpy as np
from math import log2, lgamma

c = 0.633673
pc = 0.75 ** 32
c3 = 0.9337                     # measured three-root value for j = 3 (paper)


def Xj(n, m):
    """Amortised cost of one member of an m-element fibre on n bits."""
    if m <= 1:
        return 0.0
    return 2.0 ** (n * (1 - 1 / m) + lgamma(m + 1) / (m * 0.6931471805599453) - log2(m))

print(f"{'j':>2} {'sweep S_j':>10} {'P_j':>9} {'list N_j':>10} {'m*':>10} "
      f"{'X_j(m*)':>9} {'cost/preimage':>14} {'vs 2^45.38':>11}")
for j in (1, 2, 3):
    cj = c3 if j == 3 else c ** j
    S = 2.0 ** (32 * (4 - j))
    P = cj * pc
    N = S * P
    fibre = 2.0 ** (128 - 32 * j)          # members available (R=20, v fixed)
    best = None
    for lm in np.arange(0, min(128 - 32 * j, 64) + 0.5, 0.5):
        m = 2.0 ** lm
        if m < 1:
            continue
        X = Xj(32 * j, m)
        cost = (S / m + X) / P
        if best is None or cost < best[0]:
            best = (cost, m, X)
    cost, m, X = best
    print(f"{j:2d} 2^{log2(S):8.2f} 2^{log2(P):7.2f} 2^{log2(N):8.2f} 2^{log2(m):8.2f} "
          f"2^{log2(max(X,1)):7.2f} 2^{log2(cost):12.2f} {log2(cost)-45.38:+11.2f} bits")

print("\nsame, at R = 21 (five constraints, five unknowns a0..a4; j lookups):")
print(f"{'j':>2} {'sweep S_j':>10} {'P_j':>9} {'list N_j':>10} {'m*':>10} "
      f"{'X_j(m*)':>9} {'cost/preimage':>14} {'vs 2^77.38':>11}")
for j in (2, 3, 4):
    cj = c3 * c if j == 4 else (c3 if j == 3 else c ** j)
    S = 2.0 ** (32 * (5 - j))
    P = cj * pc
    N = S * P
    best = None
    for lm in np.arange(0, min(160 - 32 * j, 96) + 0.5, 0.5):
        m = 2.0 ** lm
        X = Xj(32 * j, m)
        cost = (S / m + X) / P
        if best is None or cost < best[0]:
            best = (cost, m, X)
    cost, m, X = best
    print(f"{j:2d} 2^{log2(S):8.2f} 2^{log2(P):7.2f} 2^{log2(N):8.2f} 2^{log2(m):8.2f} "
          f"2^{log2(max(X,1)):7.2f} 2^{log2(cost):12.2f} {log2(cost)-77.38:+11.2f} bits")

print("\nThe rule: reuse pays iff a fibre member costs less than one sweep, i.e.")
print("  2^(16j) < 2^(32(4-j))/m ... in practice iff the birthday exponent 16j <= 32,")
print("  i.e. the candidate list may depend on at most TWO 32-bit context constants.")
print("  It depends on three (KC0, KC1, KC2), one per absorbed unknown, and dropping")
print("  to two costs a full 2^32 in sweep, which is the same 2^32 the reuse saves.")
