#!/usr/bin/env python3
"""Exhaustive (no replacement) a0 enumeration + across-target variance.

Kills two possible counting artefacts:
  - sampling a0 with replacement (birthday duplicates)
  - the 'preimages' being duplicates of each other
and measures how the yield varies over many independent random targets.
"""
import os, sys, struct, time, json
import numpy as np
import refute_indep as RI

M = RI.M; U32 = np.uint32; MISS = U32(M)
c = RI.c; nS0=RI.nS0; nS1=RI.nS1; nCh=RI.nCh; nMaj=RI.nMaj
S0=RI.S0; S1=RI.S1; s0=RI.s0; s1=RI.s1; Ch=RI.Ch; Maj=RI.Maj
IV=RI.IV; K=RI.K; T2f=RI.T2f
Tinv = RI.Tinv

def setup(h32, ctx, R=19):
    ab,_ = RI.back_chain(h32,R)
    a=dict(ab); a.update(ctx); a.update({-1:IV[0],-2:IV[1],-3:IV[2],-4:IV[3]})
    e={-1:IV[4],-2:IV[5],-3:IV[6],-4:IV[7]}
    for r in range(8,R): e[r]=(a[r-4]+a[r]-T2f(a[r-1],a[r-2],a[r-3]))&M
    return a,e

def enum_run(h32, ctx, lo, hi, R=19, batch=1<<22):
    a,e = setup(h32,ctx,R)
    v=a[4]; assert a[5]==v and e[8]==M and e[9]==M
    am1,am2,am3,am4=a[-1],a[-2],a[-3],a[-4]; em1,em2,em3,em4=e[-1],e[-2],e[-3],e[-4]
    a6,a7,a8,a9,a10=a[6],a[7],a[8],a[9],a[10]; e8,e9,e10=e[8],e[9],e[10]
    def recW(A,E,r): return (A[r]-T2f(A[r-1],A[r-2],A[r-3])-E[r-4]-S1(E[r-1])-Ch(E[r-1],E[r-2],E[r-3])-K[r])&M
    Wr={r:recW(a,e,r) for r in range(12,R)}
    K0p=(Wr[16]-s1(Wr[14]))&M; K1p=(Wr[17]-s1(Wr[15]))&M; K2p=(Wr[18]-s1(Wr[16]))&M
    T1_7=(a7-T2f(a6,a[5],a[4]))&M; c6=(a6-S0(a[5]))&M
    W9base=((a9-T2f(a8,a7,a6))-K[9])&M
    W10base=((a10-T2f(a9,a8,a7))-S1(e9)-K[10])&M
    W11base=((a[11]-T2f(a10,a9,a8))-S1(e10)-K[11])&M
    T15h=(v-S0(v)-Maj(v,0,0))&M; e7h=T1_7; e6h=(c6-Maj(v,v,0))&M
    W9hat=(W9base-T15h-S1(e8)-Ch(e8,e7h,e6h))&M
    D=(c6-Maj(v,v,0)+Ch(e9,e8,e7h))&M
    T2iv=T2f(am1,am2,am3)
    C0c=(-T2iv-em4-S1(em1)-Ch(em1,em2,em3)-K[0])&M; Ce0=(am4-T2iv)&M

    swept=0; tri=0; sub=0; ver=0; bad=0; blocks=set(); a0s=set()
    x=lo
    while x < hi:
        top=min(x+batch,hi)
        A0=np.arange(x,top,dtype=np.uint64).astype(U32); swept += (top-x); x=top
        E0=(A0+c(Ce0)); W0=(A0+c(C0c))
        G=(-(nS0(A0)+nMaj(A0,c(am1),c(am2)))-c(em3)-nS1(E0)-nCh(E0,c(em1),c(em2))-c(K[1]))
        F0=(c(K0p)-W0-c(W9hat)-G)
        W1=np.asarray(Tinv[F0]); ok=W1!=MISS
        A0,E0,W0,G,W1=(y[ok] for y in (A0,E0,W0,G,W1))
        A1=(W1-G); E1=(c(am3)+A1-(nS0(A0)+nMaj(A0,c(am1),c(am2))))
        F12=(-(nS0(A1)+nMaj(A1,A0,c(am1)))-c(em2)-nS1(E1)-nCh(E1,E0,c(em1))-c(K[2]))
        R1=(c(K1p)-W1-F12-c(W10base)+c(D))
        W2=np.asarray(Tinv[R1]); ok=W2!=MISS
        A0,A1,E0,E1,W1,F12,W2=(y[ok] for y in (A0,A1,E0,E1,W1,F12,W2))
        A2=(W2-F12); e2=(c(am2)+A2-(nS0(A1)+nMaj(A1,A0,c(am1))))
        F23=(-(nS0(A2)+nMaj(A2,A1,A0))-c(em1)-nS1(e2)-nCh(e2,E1,E0)-c(K[3]))
        R2=(c(K2p)-c(W11base)+c(T1_7)+c(Ch(e10,e9,e8))-W2-F23)
        W3=np.asarray(Tinv[R2]); ok=W3!=MISS
        A0,A1,A2,F23,W3=(y[ok] for y in (A0,A1,A2,F23,W3))
        A3=(W3-F23); tri+=A0.size
        if A0.size==0: continue
        good=nMaj(c(v),A3,A2)==A3
        sub+=int(good.sum())
        for j in np.nonzero(good)[0]:
            aa=dict(a); aa.update({0:int(A0[j]),1:int(A1[j]),2:int(A2[j]),3:int(A3[j])})
            ee=dict(e)
            for rr in range(0,R): ee[rr]=(aa[rr-4]+aa[rr]-T2f(aa[rr-1],aa[rr-2],aa[rr-3]))&M
            Wm=[recW(aa,ee,rr) for rr in range(16)]
            dg=b"".join(struct.pack(">I",z) for z in RI.compress_R(Wm,R))
            if dg==h32:
                ver+=1; blocks.add(tuple(Wm)); a0s.add(int(A0[j]))
            else: bad+=1
    return dict(swept=swept,tri=tri,sub=sub,ver=ver,bad=bad,
                distinct_blocks=len(blocks),distinct_a0=len(a0s))

if __name__=="__main__":
    mode=sys.argv[1]
    if mode=="exh":
        NBITS=int(sys.argv[2])
        h32=os.urandom(32)
        rng=np.random.default_rng(int.from_bytes(os.urandom(8),"little"))
        v=int(rng.integers(0,1<<32,dtype=np.uint64))
        ctx=RI.make_ctx(rng,v)
        t0=time.time()
        r=enum_run(h32,ctx,0,1<<NBITS)
        print("EXHAUSTIVE a0 in [0,2^%d), one context, v=0x%08x"%(NBITS,v))
        print(" target",h32.hex())
        print(" ",r, " %.1fs"%(time.time()-t0))
        print("  a0/preimage = %.0f  (2^%.3f per a0)"%(r['swept']/max(r['ver'],1),
              np.log2(max(r['ver'],1)/r['swept'])))
    else:
        NT=int(sys.argv[2]); N=int(sys.argv[3])
        vals=[]
        t0=time.time()
        for i in range(NT):
            h32=os.urandom(32)
            rng=np.random.default_rng(int.from_bytes(os.urandom(8),"little"))
            v=int(rng.integers(0,1<<32,dtype=np.uint64))
            ctx=RI.make_ctx(rng,v)
            lo=int.from_bytes(os.urandom(4),"little")
            lo=min(lo,(1<<32)-N)
            r=enum_run(h32,ctx,lo,lo+N)
            vals.append((r['ver'],r['tri'],r['sub'],r['bad'],r['distinct_blocks']))
            print(f"  t{i:02d} v=0x{v:08x} tri={r['tri']:,} sub={r['sub']} ver={r['ver']} "
                  f"distinct={r['distinct_blocks']} bad={r['bad']}",flush=True)
        vs=np.array([x[0] for x in vals]); tr=np.array([x[1] for x in vals])
        print("")
        print(f"{NT} independent random targets x {N:,} exhaustive a0 each")
        print(f"  verified per target: min {vs.min()} max {vs.max()} mean {vs.mean():.2f} "
              f"sd {vs.std(ddof=1):.2f}  (Poisson sd would be {np.sqrt(vs.mean()):.2f})")
        print(f"  zero-yield targets: {(vs==0).sum()}")
        print(f"  total ver {vs.sum()} / {NT*N:,} a0 = 2^{np.log2(vs.sum()/(NT*N)):.3f}"
              f"  -> {NT*N/vs.sum():,.0f} a0/preimage")
        print(f"  tri/a0 min {tr.min()/N:.5f} max {tr.max()/N:.5f}")
        print(f"  false positives total {sum(x[3] for x in vals)}")
        print(f"  [{time.time()-t0:.0f}s]")
