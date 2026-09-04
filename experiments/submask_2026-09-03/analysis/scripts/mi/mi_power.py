#!/usr/bin/env python3
"""Detection power + closing checks.
1. The a2/a3 per-bit 'bias' is FORCED: the Maj condition fixes p(a2_i=1)=1/3 or 2/3 by v_i.
2. The max-representative law is exactly f(x)=e^{x-1}/(1-e^{-1}); its total deficit.
3. Aggregate (collision / chi-square) power on c3 -- how big a deficit could hide.
"""
import numpy as np, math
BASE='/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/'
d19=np.load(BASE+'corpus_r19.npz'); d20=np.load(BASE+'corpus_r20.npz')

print('='*78)
print('1. THE a2/a3 PER-BIT BIAS IS A COROLLARY OF THE FORCED Maj CONDITION')
print("   Allowed (a2_i,a3_i) states are {(v,v),(~v,v),(~v,~v)}, each 1/3.")
print("   => p(a2_i = 1) = 1/3 if v_i=1 else 2/3 ;  p(a3_i = 1) = 2/3 if v_i=1 else 1/3.")
ctx=d19['ctx'].astype(int); nc=ctx.max()+1
v=d19['v']
err2=[];err3=[]
for c in range(nc):
    m=ctx==c; vc=int(v[m][0])
    for k in range(32):
        bit=(vc>>k)&1
        p2=((d19['a2'][m]>>k)&1).mean(); p3=((d19['a3'][m]>>k)&1).mean()
        err2.append(p2-(1/3 if bit else 2/3)); err3.append(p3-(2/3 if bit else 1/3))
err2=np.array(err2); err3=np.array(err3)
sd=math.sqrt((1/3)*(2/3)/ (len(ctx)/nc))
print('   predicted vs measured over all %d (context,bit) cells:'%len(err2))
print('   a2: mean residual %+.5f  rms %.5f   (binomial sd %.5f) -> ratio %.2f'%(err2.mean(),err2.std(),sd,err2.std()/sd))
print('   a3: mean residual %+.5f  rms %.5f   (binomial sd %.5f) -> ratio %.2f'%(err3.mean(),err3.std(),sd,err3.std()/sd))
print('   spread-inflation of pooled p across contexts, predicted (1/6)/%.5f = %.2f'%(0.5/math.sqrt(len(ctx)/nc),(1/6)/(0.5/math.sqrt(len(ctx)/nc))))
print('   => the pooled "biases" at bits 25/29 are 60-context sampling noise in v, not structure.')

print('\n'+'='*78)
print('2. THE MAXIMAL-REPRESENTATIVE LAW')
print('   For a random function, a class has R~Poisson(1) roots; P(u is the max of its')
print('   class) = sum_j e^-1/j! * x^j = e^{x-1}.  Normalising: f(x)=e^{x-1}/(1-e^{-1}).')
two32=4294967296.0; c=1-math.exp(-1)
print('   predicted p(bit31=1) = (1-e^{-1/2})/(1-e^{-1}) = %.5f'%((1-math.exp(-0.5))/c))
for nm in ['W1','W2','W3']:
    print('     %s measured %.5f   E[W]/2^32 pred %.5f meas %.5f'%(nm,((d19[nm]>>31)&1).mean(),
          math.exp(-1)/c, d19[nm].astype(float).mean()/two32))
xs=(np.arange(1,100000)-0.5)/100000; f=np.exp(xs-1)/c
print('   TOTAL entropy deficit of this law = %.5f bits (out of 32)'%( (f*np.log2(f)).mean() ))
print('   fraction of classes non-empty = 1-e^{-1} = %.5f  (matches the quoted 0.634 hit rate)'%c)
print('   solutions reached per swept a0 = (1-e^{-1})^3 = %.5f = 2^%.3f  -> %.3f bits discarded'%(c**3,3*math.log2(c),-3*math.log2(c)))

print('\n'+'='*78)
print('3. DETECTION POWER ON c3 (this bounds any speedup that could be hiding)')
n=len(d20['c3']); ctx20=d20['ctx'].astype(int); ng=ctx20.max()+1; npc=n/ng
print('   n=%d in %d contexts (%.0f per context).'%(n,ng,npc))
print('   (i) CONCENTRATED deficit (one frequency / one predictor):')
print('       T(beta)=sum_c n_c|phi_c|^2 ~ Gamma(%d,1).  A 4.5-sigma excess is T-%d = %.1f,'%(ng,ng,4.5*math.sqrt(ng)))
phi2=4.5*math.sqrt(ng)/ng/npc
print('       i.e. mean |phi_c(beta)|^2 = %.2e.  Entropy deficit of a +-beta pair = |phi|^2/ln2:'%phi2)
print('       ==> any SINGLE-FREQUENCY deficit above %.2e bits (= 2^%.1f speedup) would have shown.'%(phi2/math.log(2),phi2/math.log(2)))
print('   (ii) BROADLY SPREAD deficit: aggregate collision test on c3 mod 2^b, within context.')
for b in [8,12,16]:
    x=(d20['c3']&np.uint32((1<<b)-1)).astype(np.int64)
    col=0; pairs=0
    for cc in range(ng):
        h=np.bincount(x[ctx20==cc],minlength=1<<b).astype(float)
        col+=(h*(h-1)/2).sum(); pairs+=len(x[ctx20==cc])*(len(x[ctx20==cc])-1)/2
    exp=pairs/(1<<b); sdc=math.sqrt(exp)
    excess=(col-exp)/exp
    # sum_{beta!=0}|phi|^2 = 2^b * sum p^2 - 1 ~= excess
    dH=excess/(2*math.log(2)); dHlim=4*sdc/exp/(2*math.log(2))
    print('       b=%2d: %d collisions vs %.0f expected (%.2f sd).  Total deficit inside this'%(b,int(col),exp,(col-exp)/sdc))
    print('             2^%d subgroup = %+.4f bits; 4-sigma upper limit %.4f bits.'%(b,dH,dHlim))
print('   (iii) MI noise floor: 2x2 tables, n=%d -> E[MI_null]=%.2e bits, 99.99%%ile %.2e bits.'%(n,1/(2*n*math.log(2)),15.1/(2*n*math.log(2))))
print('        per-bit probability resolution: sd(p)=%.5f, so |p-0.5|>%.4f detectable.'%(0.5/math.sqrt(n),4*0.5/math.sqrt(n)))

print('\n'+'='*78)
print('4. WHAT THE CONVOLUTION COSTS AN ATTACKER WHO RESTRICTS THE TABLE')
print('   Suppose the policy stores roots only in a subset of density 2^-k (H(W3)=32-k).')
print('   Then the table hit rate falls by 2^-k, so the lookups cost 2^k MORE swept a0,')
print('   while c3 could at best gain 2^k.  Exactly zero-sum -- before the convolution,')
print('   which additionally destroys the gain because phi_c3 = phi_W3 * phi_{s0(W4)}.')
