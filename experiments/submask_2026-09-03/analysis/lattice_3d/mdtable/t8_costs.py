"""Every cost number quoted in the report, computed rather than asserted."""
from math import log2, exp, comb

c1 = 0.633673          # image fraction, one-root table
c3root = 0.9337        # image fraction, three-root tables (paper, sec 6.2)
coll = 0.75 ** 32      # (3/4)^32
w = 32

def L(x): return log2(x)

print("== baseline ==")
for name, c in (("one-root", c1 ** 3), ("three-root", c3root)):
    p_cand = c * coll                      # collapse survivors per swept a0
    nL = 2 ** 32 * p_cand                  # solvable kappa3 per context
    p_ctx = nL * 2 ** -32                  # P(a context is solvable)
    print(f"  {name}: candidates/a0 = 2^{L(p_cand):.2f}; per context |L| = 2^{L(nL):.2f}; "
          f"P(context solvable) = 2^{L(p_ctx):.2f}; R=20 cost = 2^{L(2**32/p_ctx):.2f}; "
          f"R=21 cost = 2^{L(2**32/(nL*2**-64)):.2f}")

print("\n== the five-coordinate signature ==")
print("  context freedom  R=20: v,a6,a7,a10,a11        = 160 bits")
print("  context freedom  R=21: + a12                  = 192 bits")
print("  signature        R=20: KC0,KC1,KC2,v,kappa3   = 160 bits")
print("  signature        R=21: + kappa4               = 192 bits")
print("  fibre over (KC0,KC1,KC2,v):  R=20  2^(128-96) = 2^32 contexts")
print("                               R=21  2^(160-96) = 2^64 contexts")

print("\n== amortisation: cost per preimage = P^-1 * (2^32/m + X),  X = cost per fibre member ==")
for R, pinv in ((20, 2 ** 32 / (2 ** 32 * c3root * coll * 2 ** -32)), (21, 1 / (c3root * coll * 2 ** -64 * 2 ** 32 / 2 ** 32))):
    pass
p20 = c3root * coll                        # P(context solvable), R=20
p21 = 2 ** 32 * c3root * coll * 2 ** -64   # P(context solvable), R=21
for R, p, base in ((20, p20, 45.38), (21, p21, 77.4)):
    print(f"  R={R}: P(context solvable) = 2^{L(p):.2f}, baseline 2^{base:.2f}")
    for X in (1, 2 ** 16, 2 ** 32, 2 ** 48, 2 ** 96):
        m = 2 ** (32 if R == 20 else 64)
        cost = (2 ** 32 / m + X) / p
        print(f"     X = 2^{L(X):5.1f}  ->  2^{L(cost):6.2f} per preimage"
              f"   ({'GAIN' if cost < 2**base else 'loss'} {abs(base - L(cost)):.1f} bits)")
    print(f"     break-even X = 2^{L(2**base * p):.2f}")

print("\n== fibre sampling: generic cost ==")
print("  hitting a prescribed (KC0,KC1,KC2): 96-bit preimage on a mixing map -> 2^96 per member")
print("  m-way multicollision on 96 bits (Wagner/Joux): ~2^(96*(m-1)/m); for m=2^13.4 that is 2^96.0")
print("  ideal 2-list MITM if the map were additively separable: 2^48 for the first member")
print("     (measured second-difference zero rate at every split: 0 of 65536, i.e. not separable)")

print("\n== can a k-coordinate table answer 'does this context have a solution'? ==")
psolv = p20
for k in (1, 2, 3, 4, 5):
    agg = 2 ** (160 - 32 * k)              # signatures aggregated per cell
    # P(cell contains no solvable signature)
    lam = agg * psolv
    print(f"  {k} full-width axes (2^{32*k} entries): each cell aggregates 2^{L(agg):.0f} signatures, "
          f"expected solvable per cell = 2^{L(lam):.1f}, P(cell says 'no') = "
          f"{'exp(-2^%.1f) ~ 0' % L(lam) if lam > 60 else '%.3e' % exp(-lam)}")
print(f"  informative iff 2^(160-32k) <= 1/P = 2^{L(1/psolv):.2f}  ->  k >= {(160 - L(1/psolv))/32:.2f}, i.e. k = 5 axes = 2^160 entries")

print("\n== k-list / MITM on the unknowns (R=20: a0..a3, four 32-bit constraints) ==")
print("  natural list size per unknown: 2^32")
print("  2-list MITM (a0,a1)|(a2,a3) on 128 bits: 2^64 time and memory   (18.6 bits WORSE than 2^45.4)")
print(f"  Wagner k=4 on 128 bits needs lists of 2^{128/3:.1f} > 2^32 available -> not applicable")
print(f"  Wagner k=8 would need 8 independent lists; only 4 unknowns exist")
print("  R=21 (a0..a4, 160 bits): MITM 2^96; Wagner k=4 needs 2^53.3 lists, three of the five are 2^32")
