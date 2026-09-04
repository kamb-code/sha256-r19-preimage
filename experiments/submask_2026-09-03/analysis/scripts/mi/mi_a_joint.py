#!/usr/bin/env python3
"""(a) Joint entropy of (a2,a3) in corpus_r19 vs the forced 32*log2(3)=50.719 bits."""
import numpy as np, math
BASE='/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/'
d=np.load(BASE+'corpus_r19.npz')
a2=d['a2'].astype(np.uint32); a3=d['a3'].astype(np.uint32); v=d['v'].astype(np.uint32)
n=len(a2)
print('n =',n)

# sanity: forced constraint
assert (((a2^a3)&(a3^v))==0).all(), 'forced Maj constraint violated!'
print('forced constraint Maj(v,a3,a2)==a3 holds on all rows: OK')

L3=math.log2(3)
def H(counts):
    c=np.asarray(counts,float); c=c[c>0]; p=c/c.sum()
    return -(p*np.log2(p)).sum()
def Hmm(counts):
    """Miller-Madow corrected entropy."""
    c=np.asarray(counts,float); N=c.sum(); k=(c>0).sum()
    return H(c)+(k-1)/(2*N*math.log(2))

# ---- per-bit symbol: which of the 3 allowed (a2_i,a3_i) states, relative to v_i
# forbidden pair is (a2_i,a3_i)=(v_i, ~v_i).  Encode state canonically as
# s = 2*(a2_i^v_i) + (a3_i^v_i)  -> forbidden is s=2 (a2=v,a3=~v) ... check
# a2_i^v_i=0, a3_i^v_i=1 -> s=1. So forbidden symbol is s==1.
print()
print('per-bit joint entropy H(a2_i,a3_i)  [forced value = log2(3) = %.6f]'%L3)
print(' bit  H(a2i,a3i)   H(a2i)   H(a3i)   p(s=1)[forbidden]  state probs (s=0,1,2,3)')
Hbits=[]
S=np.empty((32,n),np.uint8)
for i in range(32):
    b2=((a2>>i)&1); b3=((a3>>i)&1); bv=((v>>i)&1)
    s=(2*(b2^bv)+(b3^bv)).astype(np.uint8)
    S[i]=s
    cnt=np.bincount(s,minlength=4)
    h=Hmm(cnt); Hbits.append(h)
    h2=Hmm(np.bincount(b2,minlength=2)); h3=Hmm(np.bincount(b3,minlength=2))
    print('%4d  %9.5f  %7.5f  %7.5f   %9.3e     %s'%(i,h,h2,h3,cnt[1]/n,(cnt/n).round(4)))
Hbits=np.array(Hbits)
print()
print('sum of per-bit entropies (upper bd on joint H(a2,a3)) = %.4f bits'%Hbits.sum())
print('forced prediction 32*log2(3)                          = %.4f bits'%(32*L3))
print('deficit                                               = %.4f bits'%(32*L3-Hbits.sum()))

# noise floor on a single-bit entropy: bootstrap the null (uniform over 3)
rng=np.random.default_rng(0)
null=[]
for _ in range(2000):
    x=rng.integers(0,3,n)
    null.append(Hmm(np.bincount(x,minlength=3)))
null=np.array(null)
print('null H for uniform-over-3 with n=%d: mean %.6f  sd %.6f  1%%ile %.6f'%(n,null.mean(),null.std(),np.percentile(null,1)))

# ---- total correlation between bit positions: pairwise MI on the 3-symbol alphabet
print()
print('pairwise MI I(S_i;S_j) over the 32 bit-symbols (bias-corrected, shuffle null)')
MI=np.zeros((32,32))
for i in range(32):
    for j in range(i+1,32):
        c=np.bincount(S[i].astype(int)*4+S[j].astype(int),minlength=16).reshape(4,4)
        hi=Hmm(c.sum(1)); hj=Hmm(c.sum(0)); hij=Hmm(c.ravel())
        MI[i,j]=MI[j,i]=hi+hj-hij
# shuffle null
nullmi=[]
for _ in range(300):
    i,j=rng.integers(0,32,2)
    if i==j: continue
    p=rng.permutation(n)
    c=np.bincount(S[i].astype(int)*4+S[j][p].astype(int),minlength=16).reshape(4,4)
    nullmi.append(Hmm(c.sum(1))+Hmm(c.sum(0))-Hmm(c.ravel()))
nullmi=np.array(nullmi)
iu=np.triu_indices(32,1)
print('  observed pairwise MI: max %.5f  mean %.5f  (496 pairs)'%(MI[iu].max(),MI[iu].mean()))
print('  shuffle null        : max %.5f  mean %.5f  99.9%%ile %.5f'%(nullmi.max(),nullmi.mean(),np.percentile(nullmi,99.9)))
k=np.argsort(MI[iu])[::-1][:10]
print('  top pairs:', [(int(iu[0][t]),int(iu[1][t]),round(float(MI[iu][t]),5)) for t in k])
print('  sum of all pairwise MI (crude total-correlation proxy) = %.4f bits'%MI[iu].sum())

# ---- direct plug-in estimate of joint H on a coarsened alphabet (blocks of bits)
print()
print('block estimates: H of S over groups of b consecutive bit-symbols, extrapolated x(32/b)')
for b in [1,2,3,4,5,6]:
    tot=0.0; ngrp=0
    for st in range(0,32,b):
        idx=list(range(st,min(st+b,32)))
        key=np.zeros(n,np.int64)
        for t in idx: key=key*4+S[t]
        tot+=Hmm(np.bincount(key,minlength=4**len(idx)))
        ngrp+=len(idx)
    print('  b=%d: sum over %d bits = %.4f bits (forced %.4f)'%(b,ngrp,tot,ngrp*L3))
