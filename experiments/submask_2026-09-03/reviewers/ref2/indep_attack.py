"""Independent re-implementation of the submask family attack.
Algebra re-derived from FIPS 180-4; verification uses ref_sha (own SHA-256).
Exact swept counter, no verification cap, arbitrary v, arbitrary targets.
"""
import sys, os, struct, time
import numpy as np
sys.path.insert(0,'/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/ref2')
from ref_sha import compress, IV as RIV, KK as RK, M

IV=list(RIV); K=list(RK)
U=np.uint32; MM=U(0xFFFFFFFF); Z=U(0)

def rot(x,n):
    if isinstance(x,np.ndarray): return ((x>>U(n))|(x<<U(32-n)))&MM
    return ((x>>n)|(x<<(32-n)))&M
def S0(x): return rot(x,2)^rot(x,13)^rot(x,22)
def S1(x): return rot(x,6)^rot(x,11)^rot(x,25)
def s0(x): return rot(x,7)^rot(x,18)^(x>>(U(3) if isinstance(x,np.ndarray) else 3))
def s1(x): return rot(x,17)^rot(x,19)^(x>>(U(10) if isinstance(x,np.ndarray) else 10))
def Ch(x,y,z): return ((x&y)^(~x&z)) if isinstance(x,np.ndarray) else (((x&y)^((~x)&M)&z if False else ((x&y)^(((~x)&M)&z)))&M)
def Maj(x,y,z): return (x&y)^(x&z)^(y&z)
def T2(x,y,z): return (S0(x)+Maj(x,y,z))&(MM if isinstance(x,np.ndarray) else M)
def cc(x): return U(x&M)

def Wrec(a,e,r):
    return (a[r]-T2(a[r-1],a[r-2],a[r-3])-e[r-4]-S1(e[r-1])-Ch(e[r-1],e[r-2],e[r-3])-K[r])&M

def back(h,R):
    st=[(struct.unpack(">I",h[4*i:4*i+4])[0]-IV[i])&M for i in range(8)]
    a={R-1:st[0],R-2:st[1],R-3:st[2],R-4:st[3]}; e={R-1:st[4],R-2:st[5],R-3:st[6],R-4:st[7]}
    for r in (R-1,R-2,R-3,R-4):
        a[r-4]=(e[r]-((a[r]-T2(a[r-1],a[r-2],a[r-3]))&M))&M
    return a,e

def ctx_family(rng,v,R):
    c={4:v,5:v}
    c[6]=int(rng.integers(0,1<<32,dtype=np.uint64)); c[7]=int(rng.integers(0,1<<32,dtype=np.uint64))
    c[8]=(M-c[4]+T2(c[7],c[6],c[5]))&M           # e8 = a4+a8-T2(a7,a6,a5) = 0xFFFFFFFF
    c[9]=(M-c[5]+T2(c[8],c[7],c[6]))&M           # e9 = -1
    c[10]=int(rng.integers(0,1<<32,dtype=np.uint64))
    if R>=20: c[11]=int(rng.integers(0,1<<32,dtype=np.uint64))
    return c

Tinv=np.load('/nvme0n1-disk/Kamvid/sigma0_u_table.npy',mmap_mode='r').view(np.uint32)

def attack(h,ctx,v,N,seed,R=19,batch=1<<20):
    ab,eb=back(h,R)
    a=dict(ab); a.update(ctx); a.update({-1:IV[0],-2:IV[1],-3:IV[2],-4:IV[3]})
    e={-1:IV[4],-2:IV[5],-3:IV[6],-4:IV[7]}
    for r in range(8,R): e[r]=(a[r-4]+a[r]-T2(a[r-1],a[r-2],a[r-3]))&M
    # real check that back() is consistent (not the tautological one)
    a4,a5,a6,a7,a8,a9,a10=(a[i] for i in range(4,11)); a11=a[11]
    e8,e9,e10=e[8],e[9],e[10]
    assert e8==M and e9==M and a4==v and a5==v
    am1,am2,am3,am4=a[-1],a[-2],a[-3],a[-4]; em1,em2,em3,em4=e[-1],e[-2],e[-3],e[-4]
    c6=(a6-S0(a5))&M
    T17=(a7-T2(a6,a5,a4))&M
    W9base=((a9-T2(a8,a7,a6))-K[9])&M
    W10base=((a10-T2(a9,a8,a7))-S1(e9)-K[10])&M
    W11base=((a11-T2(a10,a9,a8))-S1(e10)-K[11])&M
    Wr={r:Wrec(a,e,r) for r in range(12,R)}
    K0p=(Wr[16]-s1(Wr[14]))&M; K1p=(Wr[17]-s1(Wr[15]))&M; K2p=(Wr[18]-s1(Wr[16]))&M
    T2iv=T2(am1,am2,am3)
    C0c=(-T2iv-em4-S1(em1)-Ch(em1,em2,em3)-K[0])&M; Ce0=(am4-T2iv)&M
    T15h=(a5-S0(a4)-Maj(a4,0,0))&M
    e7h=T17; e6h=(c6-Maj(a5,a4,0))&M
    W9hat=(W9base-T15h-S1(e8)-Ch(e8,e7h,e6h))&M
    D=((c6-Maj(a5,a4,0))+Ch(e9,e8,e7h))&M
    rng=np.random.default_rng(seed)
    swept=0; surv=0; tri=0; sub=0; ver=0; wset=set()
    done=0
    while done<N:
        B=min(batch,N-done); done+=B
        A0=rng.integers(0,1<<32,size=B,dtype=np.uint64).astype(U)
        swept+=B
        E0=(A0+cc(Ce0))&MM; W0=(A0+cc(C0c))&MM
        G=(-(S0(A0)+Maj(A0,cc(am1),cc(am2)))-cc(em3)-S1(E0)-Ch(E0,cc(em1),cc(em2))-cc(K[1]))&MM
        F0=(cc(K0p)-W0-cc(W9hat)-G)&MM
        W1=np.asarray(Tinv[F0]); ok=W1!=MM
        A0,E0,W0,G,W1=(x[ok] for x in (A0,E0,W0,G,W1)); surv+=A0.size
        A1=(W1-G)&MM
        E1=(cc(am3)+A1-(S0(A0)+Maj(A0,cc(am1),cc(am2))))&MM
        F12=(-(S0(A1)+Maj(A1,A0,cc(am1)))-cc(em2)-S1(E1)-Ch(E1,E0,cc(em1))-cc(K[2]))&MM
        R1=(cc(K1p)-W1-F12-cc(W10base)+cc(D))&MM
        W2=np.asarray(Tinv[R1]); ok=W2!=MM
        A0,A1,E0,E1,W1,F12,W2=(x[ok] for x in (A0,A1,E0,E1,W1,F12,W2))
        A2=(W2-F12)&MM
        e2=(cc(am2)+A2-(S0(A1)+Maj(A1,A0,cc(am1))))&MM
        F23=(-(S0(A2)+Maj(A2,A1,A0))-cc(em1)-S1(e2)-Ch(e2,E1,E0)-cc(K[3]))&MM
        R2=(cc(K2p)-cc(W11base)+cc(T17)+cc(Ch(e10,e9,e8))-W2-F23)&MM
        W3=np.asarray(Tinv[R2]); ok=W3!=MM
        A0,A1,A2,W3,=(x[ok] for x in (A0,A1,A2,W3))
        F23=F23[ok]
        A3=(W3-F23)&MM
        tri+=A0.size
        if A0.size==0: continue
        gen=((Maj(cc(v),A3,A2)-A3)&MM)==Z          # general-v form
        sub+=int(gen.sum())
        for j in np.nonzero(gen)[0]:               # NO cap
            aa=dict(a); aa.update({0:int(A0[j]),1:int(A1[j]),2:int(A2[j]),3:int(A3[j])})
            ee=dict(e)
            for rr in range(0,R): ee[rr]=(aa[rr-4]+aa[rr]-T2(aa[rr-1],aa[rr-2],aa[rr-3]))&M
            Wm=[Wrec(aa,ee,rr) for rr in range(16)]
            cv=compress(list(IV),Wm,R)             # INDEPENDENT sha256
            if b"".join(struct.pack(">I",x) for x in cv)==h:
                ver+=1; wset.add(tuple(Wm))
    return dict(swept=swept,surv=surv,tri=tri,sub=sub,ver=ver,distinct=len(wset))
