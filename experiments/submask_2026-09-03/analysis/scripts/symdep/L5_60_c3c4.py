"""Are the fourth (C3) and fifth (C4) constraint residuals independent?

If they were correlated, a 21-round preimage would cost less than 2^64 above 19.
"""
import sys, numpy as np
P="/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/lens5/"
d=np.load(P+"big21.npz")
key=d['ctx'].astype(np.int64)*(1<<32)+d['a0'].astype(np.int64)
_,fi=np.unique(key,return_index=True); fi=np.sort(fi)
D={k:d[k][fi] for k in d.files if k!='n_a0_total'}
N=len(D['a0']); swept=int(d['n_a0_total'][0])
print(f"R=21 candidates (C0,C1,C2 + submask satisfied): {len(key)} rows -> {N} distinct a0")
print(f"swept a0 total {swept:,}; density {N/swept:.3e} (predicted 0.634^3*(3/4)^32 = {0.63381**3*0.75**32:.3e})")
c3=D['c3']; c4=D['c4']; ctx=D['ctx']
def bits(x): return ((x[:,None]>>np.arange(32,dtype=np.uint32)[None,:])&1).astype(np.int8)
B3=2*bits(c3).astype(np.float64)-1; B4=2*bits(c4).astype(np.float64)-1
print(f"\n1 sigma on a bit-mean = {0.5/np.sqrt(N):.5f};  on a bit-pair correlation = {1/np.sqrt(N):.5f}")

m3=(B3.mean(0)); m4=(B4.mean(0))
print(f"[m] C3 per-bit bias: max|z| {np.abs(m3*np.sqrt(N)).max():.2f}; "
      f"C4: max|z| {np.abs(m4*np.sqrt(N)).max():.2f}  (64 tests, Bonferroni 5% -> 3.1)")

Cmat=(B3.T@B4)/N
Z=Cmat*np.sqrt(N)
i,j=np.unravel_index(np.argmax(np.abs(Z)),Z.shape)
print(f"[1] bit-pair correlations C3_i x C4_j: max|z| {np.abs(Z).max():.2f} at (C3 bit {i}, C4 bit {j})")
print(f"    1024 tests -> Bonferroni 5% |z|>4.06 ; expected max of 1024 |N(0,1)| ~ 3.4")
print(f"    top singular value {np.linalg.svd(Z,compute_uv=False)[0]:.2f} "
      f"(iid null ~{2*np.sqrt(32):.1f})")

# joint low-k mutual information
print("\n[2] mutual information between low-k bits of C3 and low-k bits of C4")
for k in (4,6,8):
    x=(c3&((1<<k)-1)).astype(np.int64); y=(c4&((1<<k)-1)).astype(np.int64)
    H=np.bincount(x*(1<<k)+y,minlength=1<<(2*k)).reshape(1<<k,1<<k).astype(float)
    p=H/N; px=p.sum(1,keepdims=True); py=p.sum(0,keepdims=True)
    nz=p>0
    mi=(p[nz]*np.log2(p[nz]/(px@py)[nz])).sum()
    bias=( (1<<k)-1 )**2/(2*N*np.log(2))   # Miller-Madow style null expectation
    print(f"   k={k}: MI = {mi:.5f} bits ; null expectation from finite sample "
          f"= {bias:.5f} bits ; excess = {mi-bias:+.5f}")

# chi-square of the joint on low-k
for k in (4,6):
    x=(c3&((1<<k)-1)).astype(np.int64); y=(c4&((1<<k)-1)).astype(np.int64)
    H=np.bincount(x*(1<<k)+y,minlength=1<<(2*k)).reshape(1<<k,1<<k).astype(float)
    E=np.outer(H.sum(1),H.sum(0))/N
    chi=((H-E)**2/E).sum(); dof=((1<<k)-1)**2
    print(f"   k={k}: independence chi2={chi:.1f} dof={dof} z={(chi-dof)/np.sqrt(2*dof):+.2f}")

# GF(2): is any linear combination of C3 bits correlated with any of C4 bits?
print("\n[3] best GF(2) linear approximation  <u,C3> = <w,C4>  found by SVD of Z:"
      f" {np.linalg.svd(Z,compute_uv=False)[0]/np.sqrt(N):.5f} correlation "
      f"(detection floor ~{3.4/np.sqrt(N):.5f})")

# does C4 depend on the context, given C3 uniform?
print("\n[4] per-context: does any context give a skewed C3 or C4?")
for nm,BB in (("C3",bits(c3)),("C4",bits(c4))):
    worst=0; where=None
    for c in np.unique(ctx):
        b=BB[ctx==c]; n=len(b); z=(b.mean(0)-0.5)/(0.5/np.sqrt(n))
        if np.abs(z).max()>worst: worst=np.abs(z).max(); where=(int(c),int(np.argmax(np.abs(z))))
    print(f"   {nm}: max|z| over {len(np.unique(ctx))}x32 tests = {worst:.2f} at ctx {where[0]} bit {where[1]}"
          f"  (Bonferroni 5% -> {4.3:.1f})")

# collision test on C3, C4 and on the PAIR
for nm,x in (("C3",c3),("C4",c4)):
    u,cnt=np.unique(x,return_counts=True); coll=int(((cnt*(cnt-1))//2).sum())
    print(f"\n[5] {nm} exact collisions {coll}, uniform expectation {N*(N-1)/2*2.0**-32:.2f}")
