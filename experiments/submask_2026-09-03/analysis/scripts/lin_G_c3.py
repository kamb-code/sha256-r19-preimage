import numpy as np, sys
sys.path.insert(0,'.')
from lin_lib import *
d=np.load('corpus_r20.npz'); N=d['c3'].size
c3=d['c3']; ctx=d['ctx'].astype(np.int64); nC=ctx.max()+1
print(f"N={N} contexts={nC}, per-ctx n min/max {np.bincount(ctx).min()}/{np.bincount(ctx).max()}")

print("\n=== 1. uniformity of c3 mod 2^k, pooled ===")
for k in range(1,13):
    b=1<<k
    cnt=np.bincount(c3 & np.uint32(b-1), minlength=b)
    chi=((cnt-N/b)**2/(N/b)).sum()
    dof=b-1
    z=(chi-dof)/np.sqrt(2*dof)
    print(f"  k={k:2d} bins={b:5d} chi2={chi:9.1f} dof={dof:5d} z={z:+.2f}")
print("=== 1b. uniformity of TOP k bits of c3 ===")
for k in range(1,13):
    b=1<<k
    cnt=np.bincount((c3>>np.uint32(32-k)).astype(np.int64), minlength=b)
    chi=((cnt-N/b)**2/(N/b)).sum(); dof=b-1
    print(f"  k={k:2d} chi2={chi:9.1f} dof={dof:5d} z={(chi-dof)/np.sqrt(2*dof):+.2f}")

print("\n=== 2. Fourier |E[e^{2pi i f c3/2^32}]| at many f; noise ~ 1/sqrt(N) = %.5f ==="%(1/np.sqrt(N)))
rng=np.random.default_rng(1)
fs=np.concatenate([np.arange(1,257,dtype=np.uint64),
                   rng.integers(1,1<<32,20000,dtype=np.uint64)])
c=c3.astype(np.float64)
best=[]
for f in fs:
    ph=2*np.pi*((np.uint64(f)*c3.astype(np.uint64))&np.uint64(0xFFFFFFFF)).astype(np.float64)/2**32
    z=np.abs(np.exp(1j*ph).mean())
    best.append((z,int(f)))
best.sort(reverse=True)
print("  top 8:", [(round(z,5),f) for z,f in best[:8]])
print("  expected max over %d tests ~ %.5f"%(len(fs), np.sqrt(np.log(len(fs))/N)))

print("\n=== 3. does c3 depend on context?  chi2 of (ctx, c3 mod 2^k) ===")
for k in (1,2,3,4,5,6):
    b=1<<k
    tab=np.zeros((nC,b))
    lo=(c3 & np.uint32(b-1)).astype(np.int64)
    np.add.at(tab,(ctx,lo),1)
    exp=tab.sum(1,keepdims=True)*tab.sum(0,keepdims=True)/N
    chi=((tab-exp)**2/exp).sum(); dof=(nC-1)*(b-1)
    print(f"  k={k} chi2={chi:9.1f} dof={dof} z={(chi-dof)/np.sqrt(2*dof):+.2f}")

print("\n=== 4. per-context Fourier of c3 (steerability): |E_ctx[e^{2pi i f c3/2^32}]| ===")
# strongest single frequency per context, f=1 and small f, plus the pooled X = c3+K3p unknown
for f in (1,2,3,17,65537):
    vals=[]
    for cc in range(nC):
        m=ctx==cc
        ph=2*np.pi*((np.uint64(f)*c3[m].astype(np.uint64))&np.uint64(0xFFFFFFFF)).astype(np.float64)/2**32
        vals.append(np.abs(np.exp(1j*ph).mean()))
    vals=np.array(vals); n=np.bincount(ctx).mean()
    print(f"  f={f:6d}: mean|coef|={vals.mean():.4f} max={vals.max():.4f}  noise E|coef|={np.sqrt(np.pi/4/n):.4f}")
