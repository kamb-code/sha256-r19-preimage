"""Instrumented re-run: exact denominators, duplicate detection, eps!=0 control."""
import warnings; warnings.simplefilter("ignore")
import importlib.util, sys, struct, numpy as np, time

spec = importlib.util.spec_from_file_location("sd", "super_degenerate.py")
sd = importlib.util.module_from_spec(spec)
sys.argv = ["sd", "19", "0", "0"]
try:
    spec.loader.exec_module(sd)
except ZeroDivisionError:
    pass

M, U32, MISS, Z, K, IV = sd.M, sd.U32, sd.MISS, sd.Z, sd.K, sd.IV
S0,S1,s0,s1,Ch,Maj,T2,c = sd.S0,sd.S1,sd.s0,sd.s1,sd.Ch,sd.Maj,sd.T2,sd.c
Tinv = sd.Tinv; R = 19

def analyse(tseed, N, B=1<<20):
    rng = np.random.default_rng(tseed)
    msg = bytes(rng.integers(0,256,55,dtype=np.uint8).tolist())
    h = sd.digest([struct.unpack(">I",(msg+b"\x80"+b"\x00"*0+struct.pack(">Q",440))[4*i:4*i+4])[0] for i in range(16)],R)
    ctx = sd.make_ctx(rng, R)
    ab, eb = sd.bc(h, R)
    a = dict(ab); a.update(ctx); a.update({-1:IV[0],-2:IV[1],-3:IV[2],-4:IV[3]})
    e = {-1:IV[4],-2:IV[5],-3:IV[6],-4:IV[7]}
    for r in range(8,R): e[r] = (a[r-4]+a[r]-T2(a[r-1],a[r-2],a[r-3]))&M
    a4,a5,a6,a7,a8,a9,a10 = (a[i] for i in range(4,11)); a11=a[11]
    am1,am2,am3,am4 = a[-1],a[-2],a[-3],a[-4]; em1,em2,em3,em4 = e[-1],e[-2],e[-3],e[-4]
    e8,e9v,e10 = e[8],e[9],e[10]
    c6 = (a6-S0(a5))&M; T1_7 = (a7-T2(a6,a5,a4))&M
    W9base = ((a9-T2(a8,a7,a6))-K[9])&M
    W10base = ((a10-T2(a9,a8,a7))-S1(e9v)-K[10])&M
    W11base = ((a11-T2(a10,a9,a8))-S1(e10)-K[11])&M
    Wr = {r: sd.recW(a,e,r) for r in range(12,R)}
    K0p=(Wr[16]-s1(Wr[14]))&M; K1p=(Wr[17]-s1(Wr[15]))&M; K2p=(Wr[18]-s1(Wr[16]))&M
    T2iv=T2(am1,am2,am3)
    C0c=(-T2iv-em4-S1(em1)-Ch(em1,em2,em3)-K[0])&M; Ce0=(am4-T2iv)&M
    T15h=(a5-S0(a4)-Maj(a4,0,0))&M; e7h=(0+T1_7)&M; e6h=(0+c6-Maj(a5,a4,0))&M
    W9hat=(W9base-T15h-S1(e8)-Ch(e8,e7h,e6h))&M
    D=((a6-S0(a5)-Maj(a5,a4,0))+Ch(e9v,e8,e7h))&M

    r_ = np.random.default_rng(tseed*17)
    swept=0; surv=0; fp=0; sub=0; ver=0
    a0_hits=[]; W_hits=set(); a0_all=[]
    ctrl_tested=0; ctrl_ver=0
    nb = N//B
    for b in range(nb):
        A0 = r_.integers(0,1<<32,size=B,dtype=np.uint64).astype(U32); swept += B
        a0_all.append(A0.copy())
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
        A3=(W3-F23)&MISS; fp+=A0.size
        if A0.size==0: continue
        smask=(A3&~A2)==Z; sub+=int(smask.sum())
        idx=np.nonzero(smask)[0]                       # NO [:50] cap
        for j in idx:
            aa=dict(a); aa.update({0:int(A0[j]),1:int(A1[j]),2:int(A2[j]),3:int(A3[j])})
            ee=dict(e)
            for rr in range(0,R): ee[rr]=(aa[rr-4]+aa[rr]-T2(aa[rr-1],aa[rr-2],aa[rr-3]))&M
            Wm=[sd.recW(aa,ee,rr) for rr in range(16)]
            if sd.digest(Wm,R)==h:
                ver+=1; a0_hits.append(int(A0[j])); W_hits.add(tuple(Wm))
        # CONTROL: triangular solutions with eps != 0 -- do any verify?
        nidx=np.nonzero(~smask)[0]
        for j in nidx[:3000]:
            aa=dict(a); aa.update({0:int(A0[j]),1:int(A1[j]),2:int(A2[j]),3:int(A3[j])})
            ee=dict(e)
            for rr in range(0,R): ee[rr]=(aa[rr-4]+aa[rr]-T2(aa[rr-1],aa[rr-2],aa[rr-3]))&M
            Wm=[sd.recW(aa,ee,rr) for rr in range(16)]
            ctrl_tested+=1
            if sd.digest(Wm,R)==h: ctrl_ver+=1
    A0all=np.concatenate(a0_all)
    return dict(swept=swept,surv=surv,fp=fp,sub=sub,ver=ver,
                n_a0_hits=len(a0_hits),n_distinct_a0_hits=len(set(a0_hits)),
                n_distinct_W=len(W_hits),
                dup_a0_draws=int(A0all.size-np.unique(A0all).size),
                ctrl_tested=ctrl_tested,ctrl_ver=ctrl_ver)

t0=time.time(); G=dict()
for ts in range(400,404):
    d=analyse(ts, 1<<22)
    print(ts, d, flush=True)
    for k,v in d.items(): G[k]=G.get(k,0)+v
print("TOTAL", G)
print(f"exact swept={G['swept']:,}  ver={G['ver']}  per-a0={G['ver']/G['swept']:.4e}  a0/preimage={G['swept']/G['ver']:,.0f}")
print(f"sub/fp = {G['sub']/G['fp']:.4e}   predicted 1.0037e-04")
print(f"[{time.time()-t0:.0f}s]")
