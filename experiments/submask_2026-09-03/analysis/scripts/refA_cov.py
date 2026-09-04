import numpy as np,time
M=np.uint32(0xFFFFFFFF)
def rotr(x,n): return (x>>np.uint32(n))|(x<<np.uint32(32-n))
def s0(x): return rotr(x,7)^rotr(x,18)^(x>>np.uint32(3))
seen=np.zeros(1<<32,dtype=np.uint8)   # 4 GiB
maxrep=None
t0=time.time(); CH=1<<25
for s in range(0,1<<32,CH):
    u=np.arange(s,s+CH,dtype=np.uint64).astype(np.uint32)
    v=(s0(u)-u)&M
    seen[v]=1
n=int(seen.sum(dtype=np.int64))
print("image count:",n,"  paper: 2721603628   diff:",n-2721603628, " time",time.time()-t0)
print("seen[0x20000000] =",int(seen[0x20000000]))
