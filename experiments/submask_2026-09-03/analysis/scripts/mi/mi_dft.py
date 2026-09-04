#!/usr/bin/env python3
"""Additive-character (DFT mod 2^32) test -- the matched filter for c3 = s0(W4)+W3-K3p.

For frequency beta,  S_c(beta) = (1/n_c) sum_j exp(2 pi i beta c3_j / 2^32).
Under a uniform residual, n_c|S_c|^2 ~ Exp(1).  Pooling over the 40 contexts,
   T(beta) = sum_c n_c |S_c(beta)|^2  ~  Gamma(40,1):  mean 40, sd 6.32.
This is phase-blind, so it survives the unknown per-context shift K3p.
A large T(beta) is a directly exploitable non-uniformity: the deficit in bits is
   log2(2^32) - H(c3) >= (1/ln2) * sum_beta!=0 |Shat(beta)|^2 / 2   (chi^2 bound)
"""
import numpy as np, math
BASE='/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/'

def dft_scan(vals, groups, betas):
    """T(beta) = sum_c n_c |mean_c exp(2pi i beta x /2^32)|^2 ."""
    T=np.zeros(len(betas))
    two32=np.float64(4294967296.0)
    for c in np.unique(groups):
        x=vals[groups==c].astype(np.float64); nc=len(x)
        for i,b in enumerate(betas):
            ph=2*math.pi*np.float64(b)*x/two32
            T[i]+=(np.cos(ph).sum()**2+np.sin(ph).sum()**2)/nc
    return T

def report(name, vals, groups, tag):
    ng=len(np.unique(groups))
    betas=[]; lab=[]
    for b in range(1,17):                       # low-b-bit structure
        step=1<<(32-b)
        for j in range(1,min(1<<b,9),2):
            betas.append(j*step); lab.append('low%dbit j=%d'%(b,j))
    for k in list(range(1,200))+[256,512,1024,4096,65536]:   # high-bit smoothness
        betas.append(k); lab.append('beta=%d'%k)
    rng=np.random.default_rng(7)
    rb=rng.integers(1,1<<32,600,dtype=np.uint64)
    for b in rb: betas.append(int(b)); lab.append('random')
    T=dft_scan(vals,groups,betas)
    mu,sd=ng,math.sqrt(ng)
    z=(T-mu)/sd
    o=np.argsort(-T)[:10]
    print('\n--- %s : %s  (%d groups, n=%d, %d frequencies) ---'%(tag,name,ng,len(vals),len(betas)))
    print('  T ~ Gamma(%d,1): mean %.1f sd %.2f ; Bonferroni p<0.05 over %d tests needs T > %.1f'
          %(ng,mu,sd,len(betas),mu+sd*4.3))
    print('  observed: mean %.2f  max %.2f (z=%.2f)'%(T.mean(),T.max(),z.max()))
    for i in o[:6]:
        print('    beta=%-12d T=%8.2f z=%+6.2f   [%s]'%(betas[i],T[i],z[i],lab[i]))
    return T,betas,lab

d20=np.load(BASE+'corpus_r20.npz'); d19=np.load(BASE+'corpus_r19.npz')
rng=np.random.default_rng(3)

print('='*78)
print('CONTROL: uniform random 32-bit values, same shape as corpus_r20')
ctx20=d20['ctx'].astype(int)
report('uniform control',rng.integers(0,1<<32,len(ctx20),dtype=np.uint64).astype(np.uint32),ctx20,'NULL')

print('\n'+'='*78)
print('THE TARGET: fourth-constraint residual c3 (R=20)')
report('c3',d20['c3'],ctx20,'R20')

print('\n'+'='*78)
print('ITS TWO ADDITIVE PARTS (structure here is what could be steered)')
M=0xFFFFFFFF
def s0(x): x=x.astype(np.uint32); return ((x>>np.uint32(7))|(x<<np.uint32(25)))^((x>>np.uint32(18))|(x<<np.uint32(14)))^(x>>np.uint32(3))
report('W3  (max table representative)',d20['W3'],ctx20,'R20')
report('sigma0(W4)',s0(d20['W4']),ctx20,'R20')
report('W4',d20['W4'],ctx20,'R20')

print('\n'+'='*78)
print('THE TABLE-REPRESENTATIVE BIAS, measured on R=19 (W1,W2,W3 are all maximal reps)')
ctx19=d19['ctx'].astype(int)
for nm in ['W1','W2','W3','W0','W4','a0']:
    report(nm,d19[nm],ctx19,'R19')
