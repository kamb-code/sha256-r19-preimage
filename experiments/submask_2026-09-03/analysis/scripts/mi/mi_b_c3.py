#!/usr/bin/env python3
"""(b) Entropy of the R=20 fourth-constraint residual c3 and its MI with every predictor.

c3 = sigma0(W4) + W3 - K3p  (mod 2^32),  K3p a per-context constant.
Cost of R=20 = 2^H(c3).  Any conditional-entropy reduction = that many bits of speedup.
"""
import numpy as np, math, sys
BASE='/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/'
d=np.load(BASE+'corpus_r20.npz')
F={k:d[k].astype(np.uint32) for k in d.files if k!='messages'}
c3=F['c3']; ctx=F['ctx'].astype(int); n=len(c3); nctx=ctx.max()+1
print('n = %d, contexts = %d'%(n,nctx))
rng=np.random.default_rng(1)

def H(c):
    c=np.asarray(c,float); c=c[c>0]; p=c/c.sum(); return -(p*np.log2(p)).sum()
def Hmm(c):
    c=np.asarray(c,float); N=c.sum(); k=(c>0).sum()
    return H(c)+(k-1)/(2*N*math.log(2))
def MI2(x,y,kx,ky):
    """MI in bits, Miller-Madow corrected, for integer labels."""
    t=np.bincount(x.astype(np.int64)*ky+y.astype(np.int64),minlength=kx*ky).reshape(kx,ky)
    return Hmm(t.sum(1))+Hmm(t.sum(0))-Hmm(t.ravel())

POP=np.array([bin(i).count('1') for i in range(1<<16)],np.uint8)
def par(x):  # parity of popcount of uint32
    return ((POP[x&0xFFFF]+POP[x>>16])&1).astype(np.uint8)

print('\n=== 1. per-bit marginals of c3 (pooled over all 40 contexts) ===')
print(' bit   p(bit=1)    dev/sd      |  within-ctx-centred chi2 vs 40 df')
sd=0.5/math.sqrt(n)
worstz=0
for k in range(32):
    b=((c3>>k)&1)
    p=b.mean(); z=(p-0.5)/sd
    # per-context deviations
    zs=[]
    for c in range(nctx):
        m=ctx==c; pc=b[m].mean(); zs.append((pc-0.5)/(0.5/math.sqrt(m.sum())))
    zs=np.array(zs); chi2=(zs**2).sum()
    worstz=max(worstz,abs(z))
    print('%4d   %.6f  %+8.2f     | chi2=%7.1f (E=40, sd=8.9)'%(k,p,z,chi2))
print('largest |z| over 32 bits: %.2f  (Bonferroni 32 tests: need |z|>3.4 for p<0.05)'%worstz)

print('\n=== 2. entropy of c3 ===')
hb=np.array([Hmm(np.bincount(((c3>>k)&1),minlength=2)) for k in range(32)])
print('sum of per-bit entropies (upper bound on H(c3)) = %.6f bits  (deficit %.2e)'%(hb.sum(),32-hb.sum()))
# low-order blocks: directly estimable
print('\n H(c3 mod 2^b) plug-in, pooled and worst/best context:')
print('   b   H_pooled   b-H     H_perctx(mean)  worst-ctx H  (n/ctx=%d)'%(n//nctx))
for b in range(1,15):
    x=(c3&((1<<b)-1)).astype(np.int64)
    hp=Hmm(np.bincount(x,minlength=1<<b))
    hs=[Hmm(np.bincount(x[ctx==c],minlength=1<<b)) for c in range(nctx)]
    hs=np.array(hs)
    print('  %2d   %8.5f  %+.2e   %8.5f      %8.5f'%(b,hp,b-hp,hs.mean(),hs.min()))
# null for the same
print('  null (uniform random 32-bit, same n):')
u=rng.integers(0,1<<32,n,dtype=np.uint64).astype(np.uint32)
for b in [8,12,14]:
    x=(u&((1<<b)-1)).astype(np.int64)
    print('   b=%2d  H=%8.5f  (b-H = %+.2e)'%(b,Hmm(np.bincount(x,minlength=1<<b)),b-Hmm(np.bincount(x,minlength=1<<b))))

print('\n=== 3. Walsh / linear test on c3 alone: corr of <beta,c3> ===')
print('    (any bias in a linear combination of c3 bits is directly exploitable)')
def walsh_scan(vals, masks, label, cond=None):
    """corr = 1-2*mean(parity(vals&mask)); optionally XOR with cond parities."""
    out=np.empty(len(masks))
    for i,m in enumerate(masks):
        p=par(vals&np.uint32(m))
        if cond is not None: p=p^cond
        out[i]=1-2*p.mean()
    return out
masks1=[1<<k for k in range(32)]
masks2=[(1<<i)|(1<<j) for i in range(32) for j in range(i+1,32)]
rmasks=list(rng.integers(1,1<<32,4000,dtype=np.uint64).astype(np.uint32))
allm=masks1+masks2+rmasks
w=walsh_scan(c3,allm,'c3')
sdc=1/math.sqrt(n)
print(' n masks tested = %d ; corr sd under null = %.5f ; Bonferroni 4.5sd = %.5f'%(len(allm),sdc,4.5*sdc))
o=np.argsort(-np.abs(w))[:8]
print(' top |corr|:', [(hex(int(allm[i])),round(float(w[i]),5),round(float(w[i]/sdc),2)) for i in o])
print(' max |z| = %.2f'%(np.abs(w).max()/sdc))
# same test within-context (removes any per-context shift washing out a bias)
print(' within-context (mean over ctx of |corr|, best single ctx/mask):')
best=(0,None,None)
for c in range(nctx):
    m=ctx==c; nc=m.sum(); wc=walsh_scan(c3[m],masks1+masks2,'')
    z=np.abs(wc).max()*math.sqrt(nc)
    if z>best[0]: best=(z,c,int(np.argmax(np.abs(wc))))
print('   best |z| over 40 ctx x 528 masks = %.2f  (need >4.6 for p<0.05 after 21120 tests)'%best[0])

print('\n=== 4. MI( c3 bit ; predictor ) ===')
# noise floor: 2x2 MI under null ~ chi2_1/(2 n ln2)
nf2=1.0/(2*n*math.log(2))
print('null E[MI] for 2x2 with n=%d is %.2e bits; 99.99%%ile ~ %.2e'%(n,nf2,15.1*nf2))
pre =['a0','v','tgt','W0','e0']          # available BEFORE the three table lookups
post=['a1','a2','a3','W1','W2','W3','W4','e1','e2','e3']
res={}
for grp,names in (('PRE-LOOKUP (exploitable)',pre),('POST-LOOKUP (worthless for speed)',post)):
    print('\n --- %s ---'%grp)
    for nm in names:
        X=F[nm]
        mx=0.0; arg=None
        for k in range(32):
            y=((c3>>k)&1).astype(np.int64)
            for j in range(32):
                x=((X>>j)&1).astype(np.int64)
                m=MI2(x,y,2,2)
                if m>mx: mx,arg=m,(j,k)
        # also byte-level (256x256 too big; use 4bit x 4bit nibbles)
        mxn=0.0; argn=None
        for k in range(0,32,4):
            y=((c3>>k)&0xF).astype(np.int64)
            for j in range(0,32,4):
                x=((X>>j)&0xF).astype(np.int64)
                m=MI2(x,y,16,16)
                if m>mxn: mxn,argn=m,(j,k)
        res[nm]=(mx,mxn)
        print('  %-4s  max bit-bit MI = %.3e bits (a0bit%d,c3bit%d)   max nibble-nibble MI = %.4f bits (n%d,n%d)'
              %(nm,mx,arg[0],arg[1],mxn,argn[0]//4,argn[1]//4))

print('\n --- categorical / aggregate pre-lookup predictors ---')
pc=(POP[F['a0']&0xFFFF]+POP[F['a0']>>16]).astype(np.int64)  # popcount(a0)
preds={'ctx (=v, 40 vals)':(ctx,nctx),
       'popcount(a0) (33)':(pc,33),
       'a0 high 8 bits':((F['a0']>>24).astype(np.int64),256),
       'a0 low 8 bits':((F['a0']&0xFF).astype(np.int64),256),
       'a0 top 12 bits':((F['a0']>>20).astype(np.int64),4096)}
for nm,(X,kx) in preds.items():
    mx=0.0; arg=None
    for k in range(32):
        y=((c3>>k)&1).astype(np.int64)
        m=MI2(X,y,kx,2)
        if m>mx: mx,arg=m,k
    # vs c3 low byte
    m8=MI2(X,(c3&0xFF).astype(np.int64),kx,256)
    print('  %-20s  max MI with a c3 bit = %.3e (bit %d);  MI with c3 low byte = %.4f'%(nm,mx,arg,m8))
    # null
    p=rng.permutation(n)
    mnull=max(MI2(X,((c3[p]>>k)&1).astype(np.int64),kx,2) for k in range(32))
    m8n=MI2(X,(c3[p]&0xFF).astype(np.int64),kx,256)
    print('  %-20s  SHUFFLE NULL       = %.3e            ;                       = %.4f'%('',mnull,m8n))

print('\n=== 5. per-context heterogeneity of the c3 distribution ===')
print(' If some contexts give a systematically better residual distribution, the')
print(' per-context low-bit histograms should differ by more than multinomial noise.')
for b in [8,10,12]:
    x=(c3&((1<<b)-1)).astype(np.int64)
    T=np.zeros((nctx,1<<b))
    for c in range(nctx):
        T[c]=np.bincount(x[ctx==c],minlength=1<<b)
    mi=Hmm(T.sum(1))+Hmm(T.sum(0))-Hmm(T.ravel())
    p=rng.permutation(n)
    T2_=np.zeros((nctx,1<<b))
    for c in range(nctx):
        T2_[c]=np.bincount(x[p][ctx==c],minlength=1<<b)
    min_=Hmm(T2_.sum(1))+Hmm(T2_.sum(0))-Hmm(T2_.ravel())
    print('  I(ctx ; c3 mod 2^%d) = %.5f bits   shuffle null = %.5f bits'%(b,mi,min_))
