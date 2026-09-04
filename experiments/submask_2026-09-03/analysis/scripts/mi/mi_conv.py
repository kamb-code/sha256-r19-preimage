#!/usr/bin/env python3
"""Why the fourth constraint is protected: c3 = s0(W4) + W3 - K3p is a CONVOLUTION.

Additive DFT turns the mod-2^32 sum into a product of characteristic functions:
    phi_c3(beta) = phi_{s0(W4)}(beta) * phi_{W3}(beta) * e^{-2pi i beta K3p/2^32}
                                                       (exactly, if the two are independent)
So a huge non-uniformity in W3 (the maximal-representative bias) is multiplied by
the near-zero coefficient of s0(W4) and vanishes.  Changing the table's
representative policy moves phi_W3 only -- it CANNOT lift phi_c3 above
|phi_{s0(W4)}(beta)|, which we bound here.

Also: EXHAUSTIVE frequency scan.  For each b, histogram c3 mod 2^b (and c3 >> 32-b)
and FFT it: this covers ALL 2^b frequencies in that subgroup exactly, per context.
"""
import numpy as np, math
BASE='/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/'
d20=np.load(BASE+'corpus_r20.npz'); d19=np.load(BASE+'corpus_r19.npz')
ctx=d20['ctx'].astype(int); nctx=ctx.max()+1
def s0(x): x=x.astype(np.uint32); return (((x>>np.uint32(7))|(x<<np.uint32(25)))^((x>>np.uint32(18))|(x<<np.uint32(14)))^(x>>np.uint32(3))).astype(np.uint32)
rng=np.random.default_rng(11)

print('='*76)
print('PART 1 -- exhaustive per-context FFT scan  T(beta)=sum_c n_c|phi_c(beta)|^2 ~ Gamma(ngrp,1)')
print('         covers EVERY frequency in the low-b-bit and high-b-bit subgroups.')
def exhaustive(vals,groups,b,high=False,label=''):
    ng=len(np.unique(groups))
    x=(vals>>np.uint32(32-b)) if high else (vals & np.uint32((1<<b)-1))
    T=np.zeros(1<<b)
    for c in np.unique(groups):
        h=np.bincount(x[groups==c].astype(np.int64),minlength=1<<b).astype(float)
        nc=h.sum()
        F=np.fft.fft(h)
        T+=np.abs(F)**2/nc
    T[0]=0
    return T,ng

for nm,vals in [('c3',d20['c3']),('UNIFORM CONTROL',rng.integers(0,1<<32,len(ctx),dtype=np.uint64).astype(np.uint32)),
                ('W3 (max rep)',d20['W3']),('s0(W4)',s0(d20['W4']))]:
    print('\n  %s :'%nm)
    for b in [4,8,12,16]:
        for high in [False,True]:
            T,ng=exhaustive(vals,ctx,b,high)
            nf=(1<<b)-1
            thr=ng+math.sqrt(ng)*(2.5+math.sqrt(2*math.log(max(nf,2))))
            print('    b=%2d %-4s : %6d freqs, max T=%9.2f (z=%+7.2f)  [null mean %d, sd %.1f, exp. max ~%.0f]'
                  %(b,'HIGH' if high else 'low',nf,T.max(),(T.max()-ng)/math.sqrt(ng),ng,math.sqrt(ng),thr))

print('\n'+'='*76)
print('PART 2 -- the convolution bound, at the frequencies where W3 is most biased')
print(' phi(beta) magnitudes, averaged over the 40 contexts as sqrt(T/(ngrp)) / sqrt(n_per_ctx):')
two32=4294967296.0
def phimag(vals,beta):
    """rms |phi_c(beta)| over contexts, and the noise floor 1/sqrt(n_c)."""
    tot=0.0; k=0
    for c in range(nctx):
        x=vals[ctx==c].astype(np.float64); nc=len(x)
        ph=2*math.pi*beta*x/two32
        tot+=(np.cos(ph).sum()**2+np.sin(ph).sum()**2)/nc**2; k+=1
    return math.sqrt(tot/k), 1/math.sqrt(len(vals)/nctx)
W3=d20['W3']; S0W4=s0(d20['W4']); C3=d20['c3']
print('  beta | |phi_W3|  |phi_s0(W4)|  product   |phi_c3| observed   noise floor')
for beta in [1,2,3,4,5,8,16,1<<16,1<<24]:
    a,nf=phimag(W3,beta); b_,_=phimag(S0W4,beta); c_,_=phimag(C3,beta)
    print('  %-9d %.5f    %.5f     %.6f   %.5f            %.5f  %s'
          %(beta,a,b_,a*b_,c_,nf,'<-- product predicts observed' if beta==1 else ''))

print('\n  => entropy deficit contributed by a frequency: dH ~ |phi|^2/ln2 bits (per +-beta pair).')
a,nf=phimag(W3,1); b_,_=phimag(S0W4,1)
print('     W3 alone at beta=1 : |phi|=%.4f  -> %.4f bits of deficit in W3'%(a,a*a/math.log(2)))
print('     s0(W4) at beta=1   : |phi|=%.4f  (noise floor %.4f) -> consistent with 0'%(b_,nf))
print('     c3 at beta=1       : |phi|=%.4f  (noise floor %.4f) -> %.2e bits'%(phimag(C3,1)[0],nf,phimag(C3,1)[0]**2/math.log(2)))
print('     HARD CAP: even a table policy driving |phi_W3(1)| -> 1.0 gives |phi_c3(1)| <= %.4f'%(b_))
print('              i.e. at most %.2e bits of speedup from that frequency.'%((b_)**2/math.log(2)))

print('\n'+'='*76)
print('PART 3 -- shape of the maximal-representative bias in W1,W2,W3 (R=19)')
print(' (this IS the known forced bias; quantified here because it is the only large')
print('  non-uniformity anywhere in the corpus, and it is what a policy change would move)')
for nm in ['W1','W2','W3']:
    w=d19[nm].astype(np.float64)
    print('  %s: mean/2^32 = %.5f (uniform 0.5)  E[w]/2^32 rel.excess %+.2f%%   top-bit p1=%.4f  p(w>2^31)=%.4f'
          %(nm,w.mean()/two32,100*(w.mean()/two32-0.5)/0.5,((d19[nm]>>31)&1).mean(),(d19[nm]>=2**31).mean()))
    q=np.percentile(w,[1,10,25,50,75,90,99])/two32
    print('      quantiles/2^32 (1,10,25,50,75,90,99): '+' '.join('%.4f'%t for t in q)+'   (uniform: .01 .10 .25 .50 .75 .90 .99)')
h,_=np.histogram(d19['W3'].astype(np.float64)/two32,bins=16,range=(0,1))
print('  W3 density in 16 equal bins (x %.1f = uniform): '%(len(d19['W3'])/16)+' '.join('%.2f'%t for t in h/(len(d19['W3'])/16)))
Hdef=-(lambda p:(p*np.log2(p/(1/16))).sum())(h/h.sum())
print('  => entropy deficit visible at 4-bit resolution of W3: %.4f bits (of 32)'%(-Hdef))

print('\n'+'='*76)
print('PART 4 -- are W3 and W4 independent?  (they are not, algebraically: a3=W3-F23 feeds W4)')
def MIb(x,y,kx,ky):
    t=np.bincount(x.astype(np.int64)*ky+y.astype(np.int64),minlength=kx*ky).reshape(kx,ky).astype(float)
    def H(c):
        c=c[c>0]; p=c/c.sum(); return -(p*np.log2(p)).sum()
    N=t.sum()
    def Hm(c): return H(c)+((c>0).sum()-1)/(2*N*math.log(2))
    return Hm(t.sum(1))+Hm(t.sum(0))-Hm(t.ravel())
for lab,X,Y in [('W3 hi8 vs W4 hi8',d20['W3']>>24,d20['W4']>>24),
                ('W3 hi8 vs s0W4 hi8',d20['W3']>>24,S0W4>>24),
                ('W3 lo8 vs W4 lo8',d20['W3']&0xFF,d20['W4']&0xFF),
                ('W3 hi8 vs c3 hi8',d20['W3']>>24,d20['c3']>>24)]:
    p=rng.permutation(len(X))
    print('  I(%-20s) = %.5f bits   shuffle null = %.5f'%(lab,MIb(X,Y,256,256),MIb(X,Y[p],256,256)))
