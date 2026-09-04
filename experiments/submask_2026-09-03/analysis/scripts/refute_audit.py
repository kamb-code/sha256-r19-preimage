"""Adversarial counting audit of the submask family.
Reuses ONLY the algebra of super_degenerate.run(); every count and every
verification is redone here.  Verification uses ref_sha.digest_words (classic
a..h formulation, constants derived from primes)."""
import os, sys, struct, time, hashlib, json
import numpy as np
sys.path.insert(0,'/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad')
from ref_sha import digest_words as REF_DIGEST

M=0xFFFFFFFF
K = [0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
     0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
     0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
     0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967]
IV=[0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]
U32=np.uint32; MISS=U32(M); Z=U32(0)
def rotr(x,n):
    if isinstance(x,np.ndarray): return ((x>>U32(n))|(x<<U32(32-n)))&MISS
    return ((x>>n)|(x<<(32-n)))&M
def S0(x): return rotr(x,2)^rotr(x,13)^rotr(x,22)
def S1(x): return rotr(x,6)^rotr(x,11)^rotr(x,25)
def s0(x): return rotr(x,7)^rotr(x,18)^(x>>(U32(3) if isinstance(x,np.ndarray) else 3))
def s1(x): return rotr(x,17)^rotr(x,19)^(x>>(U32(10) if isinstance(x,np.ndarray) else 10))
def Ch(e,f,g): return ((e&f)^(~e&g)) if isinstance(e,np.ndarray) else (((e&f)^((~e)&g))&M)
def Maj(a,b,c_): return (a&b)^(a&c_)^(b&c_)
def T2(a,b,c_): return (S0(a)+Maj(a,b,c_))&(MISS if isinstance(a,np.ndarray) else M)
def c(x): return U32(x&M)
def recW(a,e,r):
    return (a[r]-T2(a[r-1],a[r-2],a[r-3])-e[r-4]-S1(e[r-1])-Ch(e[r-1],e[r-2],e[r-3])-K[r])&M
def bc(h,R):
    s=[(struct.unpack(">I",h[4*i:4*i+4])[0]-IV[i])&M for i in range(8)]
    a={R-1:s[0],R-2:s[1],R-3:s[2],R-4:s[3]}; e={R-1:s[4],R-2:s[5],R-3:s[6],R-4:s[7]}
    for r in (R-1,R-2,R-3,R-4):
        a[r-4]=(e[r]-((a[r]-T2(a[r-1],a[r-2],a[r-3]))&M))&M
    return a,e

Tinv=np.load('/nvme0n1-disk/Kamvid/sigma0_u_table.npy',mmap_mode='r').view(np.uint32)

def make_ctx(rng,R,v=0):
    ctx={4:v,5:v}
    ctx[6]=int(rng.integers(0,1<<32,dtype=np.uint64)); ctx[7]=int(rng.integers(0,1<<32,dtype=np.uint64))
    a4,a5,a6,a7=ctx[4],ctx[5],ctx[6],ctx[7]
    ctx[8]=(M-a4+S0(a7)+Maj(a7,a6,a5))&M; a8=ctx[8]
    ctx[9]=(M-a5+S0(a8)+Maj(a8,a7,a6))&M
    ctx[10]=int(rng.integers(0,1<<32,dtype=np.uint64))
    if R>=20: ctx[11]=int(rng.integers(0,1<<32,dtype=np.uint64))
    return ctx

def run(h,ctx,N,seed,R=19,v=0,collect=None):
    ab,eb=bc(h,R)
    a=dict(ab); a.update(ctx); a.update({-1:IV[0],-2:IV[1],-3:IV[2],-4:IV[3]})
    e={-1:IV[4],-2:IV[5],-3:IV[6],-4:IV[7]}
    for r in range(8,R): e[r]=(a[r-4]+a[r]-T2(a[r-1],a[r-2],a[r-3]))&M
    a4,a5,a6,a7,a8,a9,a10=(a[i] for i in range(4,11))
    am1,am2,am3,am4=a[-1],a[-2],a[-3],a[-4]; em1,em2,em3,em4=e[-1],e[-2],e[-3],e[-4]
    e8=e[8]; e9v=e[9]; e10=e[10]
    assert e8==M and e9v==M and a4==v and a5==v
    c6=(a6-S0(a5))&M
    T1_7=(a7-T2(a6,a5,a4))&M
    W9base=((a9-T2(a8,a7,a6))-K[9])&M
    W10base=((a10-T2(a9,a8,a7))-S1(e9v)-K[10])&M
    W11base=((a[11]-T2(a10,a9,a8))-S1(e10)-K[11])&M
    Wr={r:recW(a,e,r) for r in range(12,R)}
    K0p=(Wr[16]-s1(Wr[14]))&M; K1p=(Wr[17]-s1(Wr[15]))&M; K2p=(Wr[18]-s1(Wr[16]))&M
    T2iv=T2(am1,am2,am3)
    C0c=(-T2iv-em4-S1(em1)-Ch(em1,em2,em3)-K[0])&M; Ce0=(am4-T2iv)&M
    T15h=(a5-S0(a4)-Maj(a4,0,0))&M
    e7h=(0+T1_7)&M; e6h=(0+c6-Maj(a5,a4,0))&M
    W9hat=(W9base-T15h-S1(e8)-Ch(e8,e7h,e6h))&M
    D=((a6-S0(a5)-Maj(a5,a4,0))+Ch(e9v,e8,e7h))&M
    r=np.random.default_rng(seed)
    swept=0; surv=0; fp=0; sub=0; ver=0; badver=0
    B=1<<20
    nb=(N+B-1)//B
    for b in range(nb):
        bs=min(B,N-swept)
        if bs<=0: break
        A0=r.integers(0,1<<32,size=bs,dtype=np.uint64).astype(U32); swept+=bs
        E0=(A0+c(Ce0))&MISS; W0=(A0+c(C0c))&MISS
        G=(-(S0(A0)+Maj(A0,c(am1),c(am2)))-c(em3)-S1(E0)-Ch(E0,c(em1),c(em2))-c(K[1]))&MISS
        F0=(c(K0p)-W0-c(W9hat)-G)&MISS
        W1=np.asarray(Tinv[F0]); ok=W1!=MISS
        A0,E0,W0,G,W1=(x[ok] for x in (A0,E0,W0,G,W1)); surv+=A0.size
        A1=(W1-G)&MISS
        E1=(c(am3)+A1-(S0(A0)+Maj(A0,c(am1),c(am2))))&MISS
        F12=(-(S0(A1)+Maj(A1,A0,c(am1)))-c(em2)-S1(E1)-Ch(E1,E0,c(em1))-c(K[2]))&MISS
        R1=(c(K1p)-W1-F12-c(W10base)+c(D))&MISS
        W2=np.asarray(Tinv[R1]); ok=W2!=MISS
        A0,A1,E0,E1,W1,F12,W2=(x[ok] for x in (A0,A1,E0,E1,W1,F12,W2))
        A2=(W2-F12)&MISS
        e2=(c(am2)+A2-(S0(A1)+Maj(A1,A0,c(am1))))&MISS
        F23=(-(S0(A2)+Maj(A2,A1,A0))-c(em1)-S1(e2)-Ch(e2,E1,E0)-c(K[3]))&MISS
        R2=(c(K2p)-c(W11base)+c(T1_7)+c(Ch(e10,e9v,e8))-W2-F23)&MISS
        W3=np.asarray(Tinv[R2]); ok=W3!=MISS
        A0,A1,A2,E0,E1,e2,F23,W3=(x[ok] for x in (A0,A1,A2,E0,E1,e2,F23,W3))
        A3=(W3-F23)&MISS
        fp+=A0.size
        if A0.size==0: continue
        subm=(Maj(c(v),A3,A2)-A3)&MISS==Z
        sub+=int(subm.sum())
        idx=np.nonzero(subm)[0]          # NO [:50] cap
        for j in idx:
            aa=dict(a); aa.update({0:int(A0[j]),1:int(A1[j]),2:int(A2[j]),3:int(A3[j])})
            ee=dict(e)
            for rr in range(0,R): ee[rr]=(aa[rr-4]+aa[rr]-T2(aa[rr-1],aa[rr-2],aa[rr-3]))&M
            Wm=[recW(aa,ee,rr) for rr in range(16)]
            if REF_DIGEST(Wm,R)==h:
                ver+=1
                if collect is not None: collect.append((h.hex(),tuple(Wm)))
            else: badver+=1
    return dict(swept=swept,surv=surv,fp=fp,sub=sub,ver=ver,badver=badver)
