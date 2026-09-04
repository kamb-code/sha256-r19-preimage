import numpy as np, warnings; warnings.simplefilter("ignore")
M=0xFFFFFFFF; U32=np.uint32; MISS=U32(M)
def rotr(x,n): return ((x>>U32(n))|(x<<U32(32-n)))&MISS
def s0(x): return rotr(x,7)^rotr(x,18)^(x>>U32(3))
T=np.load('/nvme0n1-disk/Kamvid/sigma0_u_table.npy',mmap_mode='r').view(np.uint32)
print("table entries:", T.size, "expected", 1<<32)
rng=np.random.default_rng(7)
u=rng.integers(0,1<<32,size=1<<22,dtype=np.uint64).astype(U32)
v=np.asarray(T[u]); hit=v!=MISS
print(f"coverage on 2^22 random u: {hit.mean():.5f}  (1-1/e = 0.63212)")
vv=v[hit]; uu=u[hit]
lhs=(s0(vv)-vv)&MISS
print("all hits satisfy s0(w)-w == u :", bool(np.all(lhs==uu)), " checked", vv.size)
# is 0xFFFFFFFF genuinely unreachable as a *value* of s0(w)-w, or is it a sentinel collision?
w=rng.integers(0,1<<32,size=1<<24,dtype=np.uint64).astype(U32)
t=(s0(w)-w)&MISS
print("fraction of random w with s0(w)-w == 0xFFFFFFFF:", float((t==MISS).mean()))
print("fraction of random w with w == 0xFFFFFFFF (lost true preimages):", float((w==MISS).mean()))
