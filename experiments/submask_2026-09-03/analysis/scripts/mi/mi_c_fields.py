#!/usr/bin/env python3
"""(c) per-bit entropy of every field + the 16 message words; (ii) exact GF(2) relations."""
import numpy as np, math
BASE='/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/'
d19=np.load(BASE+'corpus_r19.npz'); d20=np.load(BASE+'corpus_r20.npz')
rng=np.random.default_rng(5)

def bitp(x): return np.array([((x>>k)&1).mean() for k in range(32)])
def Hb(p):
    p=np.clip(p,1e-12,1-1e-12); return -(p*np.log2(p)+(1-p)*np.log2(1-p))

print('='*80)
print('(c) PER-BIT MARGINALS.  n19=%d n20=%d ; sd of p is %.5f / %.5f ;'
      %(len(d19['a0']),len(d20['a0']),0.5/math.sqrt(len(d19['a0'])),0.5/math.sqrt(len(d20['a0']))))
print('    with 32 bits x ~15 fields = ~500 tests, |z|>3.9 needed for p<0.05 (Bonferroni).')
print('    NOTE: pooling 60 contexts is legitimate -- a0 is swept uniformly & indep. of ctx.')
for tag,d in (('R19',d19),('R20',d20)):
    n=len(d['a0']); sd=0.5/math.sqrt(n)
    print('\n  --- %s ---'%tag)
    print('  field  bits with |z|>3.9                                     sumH(32)  maxdev')
    for nm in ['a0','a1','a2','a3','W0','W1','W2','W3','W4','e0','e1','e2','e3','c3']:
        p=bitp(d[nm]); z=(p-0.5)/sd
        big=[(k,round(float(p[k]),4),round(float(z[k]),1)) for k in range(32) if abs(z[k])>3.9]
        print('  %-5s %-55s %8.4f  %+6.2f'%(nm,str(big)[:55],Hb(p).sum(),z[np.argmax(np.abs(z))]))

print('\n  --- the same, computed WITHIN context then averaged (removes any ctx confound) ---')
ctx=d19['ctx'].astype(int); nc=ctx.max()+1
for nm in ['a1','a2','a3','W1','W2','W3']:
    P=np.zeros((nc,32))
    for c in range(nc): P[c]=bitp(d19[nm][ctx==c])
    m=P.mean(0); sd_within=P.std(0)/math.sqrt(nc)
    z=(m-0.5)/sd_within
    big=[(k,round(float(m[k]),4),round(float(z[k]),1)) for k in range(32) if abs(z[k])>3.9]
    print('  %-4s consistent-across-context biased bits: %s'%(nm,big if big else 'NONE'))

print('\n'+'='*80)
print('(c) MESSAGE WORDS -- corpus_r19["messages"], %d complete 16-word blocks'%len(d19['messages']))
Mm=d19['messages']
print('  W#   p(bit31)  sumH(32)   #distinct values   note')
for j in range(16):
    col=Mm[:,j]; p=bitp(col)
    print('  W%-3d %8.4f  %8.4f   %6d           %s'%(j,p[31],Hb(p).sum(),len(np.unique(col)),
          'CONSTANT within each ctx (forced)' if len(np.unique(col))<=80 else ''))
print('\n  Which words are free per block?  distinct values vs %d blocks:'%len(Mm))
print('  ',[int(len(np.unique(Mm[:,j]))) for j in range(16)])

print('\n'+'='*80)
print('(ii) EXACT AFFINE RELATIONS OVER GF(2)')
print('  Take all bits of the listed words as a GF(2) matrix (+ constant column).')
print('  dim(nullspace) = number of exactly-satisfied affine relations.')
def gf2_nullity(bitmat):
    """bitmat: (n_samples, n_vars) uint8 0/1.  Return nullity of the column space,
       i.e. n_vars - rank, computed by row-reducing the transpose."""
    A=bitmat.astype(np.uint8).copy()
    nrow,ncol=A.shape
    # rank over GF(2) by column elimination
    r=0; piv=[]
    for c in range(ncol):
        rows=np.nonzero(A[r:,c])[0]
        if len(rows)==0: continue
        i=r+rows[0]
        A[[r,i]]=A[[i,r]]
        sel=np.nonzero(A[:,c])[0]; sel=sel[sel!=r]
        A[sel]^=A[r]
        piv.append(c); r+=1
        if r==nrow: break
    return ncol-r, r
def bits_of(words,d,idx=None):
    cols=[]
    for nm in words:
        x=d[nm] if idx is None else d[nm][idx]
        for k in range(32): cols.append(((x>>k)&1).astype(np.uint8))
    return np.stack(cols,1)

for tag,d,words in (('R19 state+msg',d19,['a0','a1','a2','a3','v','W0','W1','W2','W3','W4','e0','e1','e2','e3']),
                    ('R20 state+msg',d20,['a0','a1','a2','a3','v','W0','W1','W2','W3','W4','e0','e1','e2','e3','c3'])):
    nvar=len(words)*32
    sub=rng.choice(len(d['a0']),min(nvar+400,len(d['a0'])),replace=False)
    B=bits_of(words,d,sub)
    B=np.concatenate([B,np.ones((len(sub),1),np.uint8)],1)   # affine constant
    nul,rk=gf2_nullity(B)
    print('  %-14s %d vars(+const), %d samples -> rank %d, nullity %d  %s'
          %(tag,nvar,len(sub),rk,nul,'<-- '+str(nul)+' exact affine relations' if nul else '(no exact relations)'))
    # per-context (v, W12..W15 constant there, so more relations are expected trivially)
    c0=np.nonzero(d['ctx']==0)[0]
    sub=c0[rng.choice(len(c0),min(nvar+300,len(c0)),replace=False)]
    B=bits_of(words,d,sub); B=np.concatenate([B,np.ones((len(sub),1),np.uint8)],1)
    nul,rk=gf2_nullity(B)
    print('  %-14s within ctx 0 (%d samples) -> rank %d, nullity %d (32 of these are v=const)'%('',len(sub),rk,nul))

print('\n'+'='*80)
print('(ii) BEST LINEAR APPROXIMATION: corr of any GF(2) combination of {a0,v} bits with any of c3')
print('     (random-mask search; this is the linear-cryptanalysis test on the R=20 residual)')
POP=np.array([bin(i).count('1') for i in range(1<<16)],np.uint8)
def par(x): return ((POP[x&0xFFFF]+POP[x>>16])&1).astype(np.uint8)
n=len(d20['c3']); best=(0,None)
NM=200000
for chunk in range(20):
    ma=rng.integers(0,1<<32,NM//20,dtype=np.uint64).astype(np.uint32)
    mc=rng.integers(1,1<<32,NM//20,dtype=np.uint64).astype(np.uint32)
    for i in range(len(ma)):
        c=1-2*(par(d20['a0']&ma[i])^par(d20['c3']&mc[i])).mean()
        if abs(c)>best[0]: best=(abs(c),(int(ma[i]),int(mc[i])))
sdc=1/math.sqrt(n)
print('  %d random (a0-mask, c3-mask) pairs tested.  corr sd = %.5f'%(NM,sdc))
print('  best |corr| = %.5f  = %.2f sd.  Expected max of %d gaussians = %.2f sd.'
      %(best[0],best[0]/sdc,NM,math.sqrt(2*math.log(NM))))
print('  masks:',[hex(x) for x in best[1]])

print('\n'+'='*80)
print('(d) HOW MUCH OF THE PREIMAGE SPACE DOES THE ATTACK REACH?')
n19=len(d19['a0']); sw=None
print('  Per swept a0 the attack yields a solution iff all three table lookups hit')
print('  AND the (3/4)^32 bitwise condition holds.  Each lookup hits with prob')
print('  1-e^{-1}=0.63212 (measured below), and only the MAXIMAL root of each class')
print('  is stored, so of the E[#roots|>=1]=1/(1-e^-1)=1.582 roots only 1 is reachable.')
r=1-math.exp(-1)
print('  fraction of (W1,W2,W3) triples reachable = (1/1.582)^3 = %.4f = 2^%.3f'%(r**3,3*math.log2(r)))
print('  -> the attack reaches 2^%.3f = %.1f%% of the solutions available per swept a0,'%(3*math.log2(r),100*r**3))
print('     i.e. it discards %.3f bits.  ("about a quarter" in the brief: %.3f)'%(-3*math.log2(r),r**3))
# verify the 1-e^-1 law against the measured W3 density
two32=4294967296.0
w=d19['W3'].astype(np.float64)/two32
print('\n  CHECK of the max-representative law:  density should be f(x)=e^{x-1}/(1-e^{-1}).')
print('    predicted E[W]/2^32 = e^{-1}/(1-e^{-1}) = %.5f'%(math.exp(-1)/(1-math.exp(-1))))
print('    measured   E[W1,W2,W3]/2^32 = %.5f %.5f %.5f'
      %(d19['W1'].astype(float).mean()/two32,d19['W2'].astype(float).mean()/two32,d19['W3'].astype(float).mean()/two32))
h,_=np.histogram(w,bins=32,range=(0,1)); h=h/(len(w)/32)
x=(np.arange(32)+0.5)/32; pred=np.exp(x-1)/(1-math.exp(-1))
print('    max |measured/predicted - 1| over 32 bins = %.4f'%np.abs(h/pred-1).max())
kl=(pred/32*np.log2(pred)).sum()
print('    entropy deficit of that law = %.5f bits; measured at 5-bit resolution = %.5f bits'
      %( (lambda:  (np.exp(np.linspace(0,1,100001)-1)/(1-math.exp(-1))*np.log2(np.exp(np.linspace(0,1,100001)-1)/(1-math.exp(-1)))).mean())(), (h/32*np.log2(np.clip(h,1e-9,None))).sum()))
