import random, math
M = 0xFFFFFFFF
Maj = lambda x,y,z: ((x&y)^(x&z)^(y&z)) & M
r = random.Random(9)
bad = 0
for _ in range(500000):
    v,a2,a3 = r.getrandbits(32), r.getrandbits(32), r.getrandbits(32)
    lhs = (((Maj(v,a3,a2) - a3) & M) == 0)
    rhs = (((a2 ^ a3) & (a3 ^ v)) == 0)
    if lhs != rhs: bad += 1
print("eps==0  <=>  (a2^a3)&(a3^v)==0 :  mismatches", bad, "/500000")
# exact count of solutions, all v, small width
for n in (1,2,3,8,10):
    m=(1<<n)-1
    cnts=set()
    for v in range(1<<n):
        cnts.add(sum(1 for a2 in range(1<<n) for a3 in range(1<<n) if ((a2^a3)&(a3^v))&m==0))
    print(f" n={n}: distinct counts over all v = {cnts}, 3^n={3**n}")
print("p = (3/4)^32 =", repr((3/4)**32), "= 2^%.6f"%math.log2((3/4)**32))
print("3^32 =", 3**32, " / 2^64 =", 3**32/2**64)
print("claimed 1.0037e-4 relative error:", abs(1.0037e-4-(3/4)**32)/((3/4)**32))
