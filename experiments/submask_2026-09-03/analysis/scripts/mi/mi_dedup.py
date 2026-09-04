#!/usr/bin/env python3
"""The b=16 collision excess: real structure, or repeated a0 in the sweep?
The sweep draws a0 with replacement, so a repeated (ctx,a0) gives an EXACT c3 repeat."""
import numpy as np, math
BASE='/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/'
d=np.load(BASE+'corpus_r20.npz')
ctx=d['ctx'].astype(np.int64); a0=d['a0'].astype(np.int64); c3=d['c3']
n=len(ctx); ng=ctx.max()+1
key=ctx*(1<<32)+a0
u,inv,cnt=np.unique(key,return_inverse=True,return_counts=True)
dup_pairs=int((cnt*(cnt-1)//2).sum())
print('rows %d ; distinct (ctx,a0) %d ; duplicate PAIRS from repeated a0 = %d'%(n,len(u),dup_pairs))
# are duplicated (ctx,a0) really identical in c3?
rep=np.nonzero(cnt>1)[0][:5]
for r in rep[:3]:
    idx=np.nonzero(inv==r)[0]
    print('  example dup (ctx=%d,a0=%d): c3 values %s -> identical: %s'%(ctx[idx[0]],a0[idx[0]],
          [hex(int(x)) for x in c3[idx]], len(set(c3[idx].tolist()))==1))

print('\nCollision test on c3 mod 2^b, WITHIN context, before and after de-duplicating (ctx,a0):')
first=np.zeros(n,bool); seen=set()
_,ui=np.unique(key,return_index=True)
first[ui]=True
print('  kept %d of %d rows after dedup'%(first.sum(),n))
for b in [8,12,16,20]:
    for lab,mask in (('raw   ',np.ones(n,bool)),('dedup ',first)):
        x=(c3[mask]&np.uint32((1<<b)-1)).astype(np.int64); cx=ctx[mask]
        col=0.0; pairs=0.0
        for cc in range(ng):
            h=np.bincount(x[cx==cc],minlength=1<<b).astype(float)
            m=h.sum(); col+=(h*(h-1)/2).sum(); pairs+=m*(m-1)/2
        exp=pairs/(1<<b); sd=math.sqrt(exp)
        dH=((col-exp)/exp)/(2*math.log(2))
        print('   b=%2d %s: %8d collisions vs %9.1f expected  z=%+6.2f   implied deficit %+8.4f bits'
              %(b,lab,int(col),exp,(col-exp)/sd,dH))
    print('        [dup pairs alone would add %d collisions at every b]'%dup_pairs)

print('\nSame check on the R=19 corpus (a0 uniqueness):')
d19=np.load(BASE+'corpus_r19.npz')
k19=d19['ctx'].astype(np.int64)*(1<<32)+d19['a0'].astype(np.int64)
u19,c19=np.unique(k19,return_counts=True)
print('  rows %d distinct (ctx,a0) %d -> %d duplicate pairs'%(len(k19),len(u19),int((c19*(c19-1)//2).sum())))
