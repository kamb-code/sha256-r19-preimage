#!/usr/bin/env python3
"""Chase the one per-bit anomaly seen in part (a): H(a2_25),H(a2_29) ~ 0.990 not 1.000.
Is it a genuine pooled bias, or a few contexts pulling in the same direction?"""
import numpy as np, math, sys
BASE='/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/'
d=np.load(BASE+'corpus_r19.npz')
ctx=d['ctx'].astype(int); nc=ctx.max()+1; n=len(ctx)
sd=0.5/math.sqrt(n)
print('n=%d, %d contexts, pooled sd(p)=%.5f'%(n,nc,sd))
print('\npooled p(bit=1), and the SPREAD of the 60 per-context p values')
print('  (if pooled bias is real & steerable it must be consistent ACROSS contexts;')
print('   if the per-context values scatter far more than binomial, it is a per-context')
print('   effect that averages away and cannot be exploited without choosing contexts.)')
for nm in ['a1','a2','a3','W1','W2','W3','e1','e2','e3']:
    x=d[nm]
    P=np.zeros((nc,32))
    for c in range(nc): P[c]=[((x[ctx==c]>>k)&1).mean() for k in range(32)]
    pooled=np.array([((x>>k)&1).mean() for k in range(32)])
    zp=(pooled-0.5)/sd
    # binomial expectation for per-context spread
    npc=np.bincount(ctx).mean(); exp_sd=0.5/math.sqrt(npc)
    obs_sd=P.std(0)
    infl=obs_sd/exp_sd
    k=int(np.argmax(np.abs(zp)))
    print('\n %s: worst pooled bit %d p=%.4f z=%+.1f'%(nm,k,pooled[k],zp[k]))
    print('   per-ctx spread inflation (obs sd / binomial sd), 32 bits: max %.2f  mean %.2f'%(infl.max(),infl.mean()))
    bad=[(j,round(float(pooled[j]),4),round(float(zp[j]),1),round(float(infl[j]),2)) for j in range(32) if abs(zp[j])>3.9]
    print('   bits with |pooled z|>3.9 (bit,p,z,spread-inflation):',bad if bad else 'NONE')
    if bad:
        j=bad[int(np.argmax([abs(b[2]) for b in bad]))][0]
        # t-test treating each context as one observation -> the honest test
        t=(P[:,j].mean()-0.5)/(P[:,j].std(ddof=1)/math.sqrt(nc))
        print('   >> context-level t-test on bit %d: t=%.2f on %d df  (|t|>4.1 needed after 32x9 tests)'%(j,t,nc-1))
        print('   >> per-context p on bit %d: min %.3f max %.3f  (binomial sd %.4f)'%(j,P[:,j].min(),P[:,j].max(),exp_sd))
sys.stdout.flush()
