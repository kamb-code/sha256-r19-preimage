"""Exact decomposition eps = Q - P, P and Q disjoint, giving a rigorous (3/4)^32."""
import random
M = 0xFFFFFFFF
Maj = lambda x,y,z: ((x&y)^(x&z)^(y&z)) & M
r = random.Random(17)
bad = 0
for _ in range(400000):
    v,a2,a3 = r.getrandbits(32), r.getrandbits(32), r.getrandbits(32)
    P = a3 & ~a2 & ~v & M          # Maj bit is 0 where a3 bit is 1
    Q = a2 & ~a3 &  v & M          # Maj bit is 1 where a3 bit is 0
    assert P & Q == 0
    assert P & a3 == P and Q & a3 == 0
    if Maj(v,a3,a2) != ((a3 - P + Q) & M): bad += 1          # borrow-free
    if ((Maj(v,a3,a2) - a3) & M) != ((Q - P) & M): bad += 1
print("eps = Q - P with P,Q disjoint, P<=a3, Q&a3=0 :  mismatches", bad, "/400000")
# per-bit independence: bit i of P is 1 iff (v_i,a3_i,a2_i)=(0,1,0)  -> prob 1/4 when v_i=0
# bit i of Q is 1 iff (v_i,a3_i,a2_i)=(1,0,1)                        -> prob 1/4 when v_i=1
# exactly one of the two is possible at each bit, so P|Q has iid Bernoulli(1/4) bits
# and eps=0 <=> P=Q=0 <=> P|Q=0, probability (3/4)^32 for every v.
import collections
c = collections.Counter()
for _ in range(200000):
    v,a2,a3 = r.getrandbits(32), r.getrandbits(32), r.getrandbits(32)
    P = a3 & ~a2 & ~v & M; Q = a2 & ~a3 & v & M
    c[bin(P|Q).count('1')] += 1
tot = sum(c.values())
from math import comb
print("popcount(P|Q) empirical vs Binomial(32,1/4):")
for k in range(0, 20):
    if c[k]:
        print(f"  {k:2d}: {c[k]/tot:.5f} vs {comb(32,k)*0.25**k*0.75**(32-k):.5f}")
