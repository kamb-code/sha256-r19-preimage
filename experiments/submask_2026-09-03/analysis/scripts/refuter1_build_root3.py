import sys, time, numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import M, U32, MISS, s0
D="/nvme0n1-disk/Kamvid/"
first=np.load(D+"sigma0_u_table.npy",mmap_mode="r").view(np.uint32)
second=np.load(D+"sigma0_u_table_root2.npy",mmap_mode="r").view(np.uint32)
third=np.full(1<<32, M, dtype=np.uint32); CH=1<<24; t0=time.time()
for s in range(0,1<<32,CH):
    u=np.arange(s,s+CH,dtype=np.uint64).astype(U32); v=(s0(u)-u)&MISS
    m=(np.asarray(first[v])!=u)&(np.asarray(second[v])!=u)&(third[v]==MISS)
    if m.any(): third[v[m]]=u[m]
    if (s//CH)%64==63: print(f"  {(s+CH)/2**32*100:5.1f}% {time.time()-t0:6.0f}s",flush=True)
n3=int((third[::64]!=MISS).sum())*64
print(f"third-root coverage ~{n3/2**32*100:.2f}% (Poisson(1) P(N>=3): 8.03%)")
np.save("/nvme0n1-disk/Kamvid/sigma0_u_table_root3_refuter1.npy", third); print("saved", time.time()-t0)
