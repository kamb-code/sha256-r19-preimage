"""Does a SPECIAL family word v (rather than a random one) collapse c3?
The known collapse holds for every v; a c3 collapse might need a special one.
Exhaustive over (a2,a3) at w=8 for structured v."""
import numpy as np
from alg import Alg, Instance, R
W=8; A=Alg(W); M=A.M; n=1<<W
a=np.arange(n,dtype=np.uint64); X=np.repeat(a,n); Y=np.tile(a,n)
rng=np.random.default_rng(1234)
print(f"w={W}: |{{c3=0}}| over all {n*n} pairs (a2,a3); uniform prediction {n};")
print(f"     a genuine per-bit collapse would give >= (3/2)^{W} x {n} = {1.5**W*n:.0f}")
vs=[0,M,0x55&M,0xAA&M,1,M-1,0x0F&M,0xF0&M]+[int(rng.integers(0,n)) for _ in range(4)]
for v in vs:
    tot=[]
    for t in range(25):
        rv=lambda: int(rng.integers(0,n))
        Wt=[rv() for _ in range(16)]; af,ef,_=A.forward(Wt,R)
        chain={i:af[i] for i in range(12,R)}
        inst=Instance(A,chain,A.family(v,rv(),rv(),rv(),rv()))
        f=[np.full(X.shape,rv(),dtype=np.uint64),np.full(X.shape,rv(),dtype=np.uint64)]
        c=inst.residuals(f[0],f[1],X,Y)
        tot.append(int((c[3]==0).sum()))
    print(f"  v=0x{v:02x}: mean {np.mean(tot):7.1f}  max {max(tot):5d}  "
          f"ratio {np.mean(tot)/n:.3f}")
