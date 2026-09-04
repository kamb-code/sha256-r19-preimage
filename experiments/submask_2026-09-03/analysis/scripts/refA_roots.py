import numpy as np
M=np.uint32(0xFFFFFFFF)
def rotr(x,n): return (x>>np.uint32(n))|(x<<np.uint32(32-n))
def s0(x): return rotr(x,7)^rotr(x,18)^(x>>np.uint32(3))
targets=[0x20000000,0x91002000]
found={t:[] for t in targets}
CH=1<<25
for s in range(0,1<<32,CH):
    u=np.arange(s,s+CH,dtype=np.uint64).astype(np.uint32)
    v=(s0(u)-u)&M
    for t in targets:
        idx=np.nonzero(v==np.uint32(t))[0]
        if idx.size: found[t].extend(int(x) for x in u[idx])
for t in targets:
    print(hex(t), "roots:", [hex(r) for r in sorted(found[t])], "count", len(found[t]))
