import numpy as np
M32=np.uint32(0xFFFFFFFF); Mi=0xFFFFFFFF
IV=[0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]
K0=0x428a2f98
def rr(x,n): n%=32; return ((x>>n)|(x<<(32-n)))&Mi
def S1i(x): return rr(x,6)^rr(x,11)^rr(x,25)
def S0i(x): return rr(x,2)^rr(x,13)^rr(x,22)
def Chi(e,f,g): return ((e&f)^(~e&g))&Mi
def Maji(a,b,c): return (a&b)^(a&c)^(b&c)
C=(IV[7]+S1i(IV[4])+Chi(IV[4],IV[5],IV[6])+K0)&Mi
D=(S0i(IV[0])+Maji(IV[0],IV[1],IV[2]))&Mi
def rotl_a(x,n):
    n=int(n)%32
    return x if n==0 else ((x<<np.uint32(n))|(x>>np.uint32(32-n))).astype(np.uint32)
GS=[1,2,3,8,16,24,31]
totT1={g:0 for g in GS}; totAll={g:0 for g in GS}
CH=1<<26
print(f"C={C:08x} D={D:08x} IV3={IV[3]:08x}",flush=True)
for base in range(0,1<<32,CH):
    x=(np.arange(base,base+CH,dtype=np.uint64)&0xFFFFFFFF).astype(np.uint32)
    T1a=(x+np.uint32(C))&M32
    a0a=(T1a+np.uint32(D))&M32
    e0a=(np.uint32(IV[3])+T1a)&M32
    for g in GS:
        T1b=(rotl_a(x,g)+np.uint32(C))&M32
        ok=T1b==rotl_a(T1a,g)
        totT1[g]+=int(ok.sum())
        if ok.any():
            a0b=(T1b+np.uint32(D))&M32; e0b=(np.uint32(IV[3])+T1b)&M32
            totAll[g]+=int((ok&(a0b==rotl_a(a0a,g))&(e0b==rotl_a(e0a,g))).sum())
    if base % (1<<30)==0: print("  ..",base,flush=True)
for g in GS:
    print(f"g={g:2d}: W0 giving T1 rotational: {totT1[g]:,} of 2^32; also a0,e0 rotational: {totAll[g]:,}",flush=True)
