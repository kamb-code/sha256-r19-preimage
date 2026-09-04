import numpy as np, math
BASE='/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/'
d20=np.load(BASE+'corpus_r20.npz')
rng=np.random.default_rng(9)
k=d20['ctx'].astype(np.int64)*(1<<32)+d20['a0'].astype(np.int64)
_,ui=np.unique(k,return_index=True); keep=np.zeros(len(k),bool); keep[ui]=True
a0=d20['a0'][keep]; c3=d20['c3'][keep]; n=len(a0)
print('LINEAR (GF(2)) CRYPTANALYSIS OF c3 -- deduped n=%d'%n)
POP=np.array([bin(i).count('1') for i in range(1<<16)],np.uint8)
def parM(x,masks):
    out=np.empty((len(masks),len(x)),np.int8)
    for i,m in enumerate(masks):
        y=x&m; out[i]=1-2*((POP[y&0xFFFF]+POP[y>>16])&1)
    return out
best=(0,None,None); tested=0; CH=1200
for rep in range(12):
    ma=rng.integers(0,1<<32,CH,dtype=np.uint64).astype(np.uint32)
    mc=rng.integers(1,1<<32,CH,dtype=np.uint64).astype(np.uint32)
    A=parM(a0,ma).astype(np.float32); C=parM(c3,mc).astype(np.float32)
    G=(A@C.T)/n; tested+=CH*CH
    i,j=np.unravel_index(np.argmax(np.abs(G)),G.shape)
    if abs(G[i,j])>best[0]: best=(float(abs(G[i,j])),int(ma[i]),int(mc[j]))
sdc=1/math.sqrt(n)
print(' tested %d (a0-mask, c3-mask) linear approximations (a0 is PRE-LOOKUP)'%tested)
print(' corr sd under null = %.5f ; best |corr| = %.5f = %.2f sd'%(sdc,best[0],best[0]/sdc))
print(' expected max of %d indep gaussians = %.2f sd'%(tested,math.sqrt(2*math.log(tested))))
print(' verdict: %s'%('CONSISTENT WITH NOISE' if best[0]/sdc < math.sqrt(2*math.log(tested))+0.8 else 'EXCESS -- investigate'))
print(' best masks: a0=%s c3=%s'%(hex(best[1]),hex(best[2])))
print(' => any linear (GF(2)) predictor of a c3 bit-combination from a0 has bias < %.5f,'%(4.8*sdc/2))
print('    i.e. removes less than %.2e bits of entropy.'%((4.8*sdc)**2/(2*math.log(2))))
