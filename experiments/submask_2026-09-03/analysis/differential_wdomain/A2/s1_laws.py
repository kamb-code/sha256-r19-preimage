"""S1: exact per-primitive XOR-difference laws for SHA-256.

(i)   Sigma0/Sigma1/sigma0/sigma1 : GF(2)-linear, prob 1.
(ii)  modular addition            : P[(a^d)+b == (a+b)^d] = 2^-hw(d & 0x7fffffff)
(iii) Ch, Maj                     : per-bit transition tables (exact, 2^3 enum)
"""
import numpy as np
import itertools
from core import S0, S1, s0, s1, ch, maj, popcnt32

rng = np.random.default_rng(20260908)
N = 1 << 20

print("=" * 70)
print("(i) LINEAR LAYER: exact for XOR differences")
for name, fn in [("Sigma0", S0), ("Sigma1", S1), ("sigma0", s0), ("sigma1", s1)]:
    x = rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32)
    d = rng.integers(0, 1 << 32, N, dtype=np.uint32)
    bad = int(np.count_nonzero(fn(x ^ d) != (fn(x) ^ fn(d))))
    print(f"   {name}(x^d) == {name}(x)^{name}(d) : violations {bad} / {N}")

print()
print("=" * 70)
print("(ii) MODULAR ADDITION: P[(a^d)+b == (a+b)^d] vs 2^-hw(d & 0x7fffffff)")
print("     w = hw(d) ; w' = hw(d & 0x7fffffff)  (MSB is free)")
print(f"     {'d':>10} {'hw':>3} {'hw<31':>5} {'measured p':>12} {'-log2':>7} {'pred -log2':>10}")
tests = []
for w in range(0, 9):
    # d with w random bits among 0..30
    bits = rng.choice(31, size=w, replace=False)
    tests.append(int(sum(1 << int(b) for b in bits)))
tests += [0x80000000, 0x80000001, 0xC0000000, 0xFFFFFFFF, 0x00000001, 0x00000003]
for d in tests:
    a = rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32)
    b = rng.integers(0, 1 << 32, N, dtype=np.uint64).astype(np.uint32)
    dd = np.uint32(d)
    ok = int(np.count_nonzero(((a ^ dd) + b).astype(np.uint32)
                              == ((a + b).astype(np.uint32) ^ dd)))
    p = ok / N
    wl = bin(d).count("1")
    wp = bin(d & 0x7FFFFFFF).count("1")
    lm = -np.log2(p) if p > 0 else float("inf")
    print(f"     {d:#010x} {wl:3d} {wp:5d} {p:12.6f} {lm:7.3f} {wp:10d}")

print()
print("=" * 70)
print("(iii) Ch / Maj : exact per-bit XOR difference distribution table")
print("      rows = (da,db,dc) input difference pattern at one bit position")
print("      value = P[output difference = 1] over the 8 input values")
for name, fn in [("Ch", lambda x, y, z: (x & y) ^ ((1 - x) & z)),
                 ("Maj", lambda x, y, z: (x & y) ^ (x & z) ^ (y & z))]:
    print(f"   -- {name} --")
    for da, db, dc in itertools.product([0, 1], repeat=3):
        cnt = 0
        for x, y, z in itertools.product([0, 1], repeat=3):
            if fn(x, y, z) != fn(x ^ da, y ^ db, z ^ dc):
                cnt += 1
        cost = 0.0 if cnt in (0, 8) else 1.0
        print(f"      ({da}{db}{dc}) -> P[dout=1] = {cnt}/8 = {cnt/8:.3f}"
              f"   cost to fix output = {cost:.0f} bit")
