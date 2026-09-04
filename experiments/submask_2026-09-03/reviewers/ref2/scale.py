import indep_attack as ia, numpy as np, os, time, struct, json
rng=np.random.default_rng(777)
h=b"\xff"*32
v=int(rng.integers(0,1<<32,dtype=np.uint64))
ctx=ia.ctx_family(rng,v,19)
t0=time.time(); t0c=time.process_time()
r=ia.attack(h,ctx,v,1<<26,12345)
print("all-ones, ONE context, v=%08x"%v, r)
print("swept/preimage %.0f  wall %.1fs cpu %.1fs  a0/cpu-sec %.3g"%(r['swept']/r['ver'],time.time()-t0,time.process_time()-t0c,r['swept']/(time.process_time()-t0c)))
