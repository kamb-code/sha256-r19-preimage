"""H100 extended run — 2000 contexts with W9_hi residual logging.

Identical to h100_attack.py but also records W9_hi residuals for every
lo_pass fixed point. At end, runs chi-squared test to determine whether
the 1/2^16 uniform assumption holds or there's structural bias.

W9_lo-consistent initialization (K_seeds=4 by default):
  Pre-compute K W9_lo-consistent (a2,a3) seed pairs per context.
  Replace the fixed (0,0) start with these seeds — empirically ~3.3×
  more W9_lo events per unit compute (10-context test on H100, R=19).
"""
import numpy as np
import os, sys, time, collections
import torch

from sha256_core import sha256_full_trace
from utils import (H0, K, MASK32, big_sigma0, big_sigma1, ch, maj,
                   small_sigma0, small_sigma1)
from extended_solver import backward_chain, verify_preimage, compute_e_from_a, recover_W

device = 'cuda'
M = 0xFFFFFFFF; LO = 0xFFFF

def tr(x,n): return ((x>>n)|(x<<(32-n)))&M
def gS0(x): return tr(x,2)^tr(x,13)^tr(x,22)
def gS1(x): return tr(x,6)^tr(x,11)^tr(x,25)
def gs0(x): return tr(x,7)^tr(x,18)^(x>>3)
def gs1(x): return tr(x,17)^tr(x,19)^(x>>10)
def gch(e,f,g): return (e&f)^(~e&M&g)
def gmaj(a,b,c): return (a&b)^(a&c)^(b&c)

a_m1=H0[0]; a_m2=H0[1]; a_m3=H0[2]; a_m4=H0[3]
e_m1=H0[4]; e_m2=H0[5]; e_m3=H0[6]; e_m4=H0[7]
T2_0iv=(big_sigma0(a_m1)+maj(a_m1,a_m2,a_m3))&M
C0_CONST=(-T2_0iv-e_m4-big_sigma1(e_m1)-ch(e_m1,e_m2,e_m3)-K[0])&M
CONST_E0=(a_m4-T2_0iv)&M

SEN32 = torch.iinfo(torch.int32).min

def build_table():
    print("Building s0(u)-u table (2^32, int32, 16 GB)...", flush=True)
    t0 = time.time()
    tbl = torch.full((2**32,), SEN32, dtype=torch.int32, device=device)
    for s in range(0, 2**32, 2**25):
        e = min(s+2**25, 2**32)
        u = torch.arange(s, e, dtype=torch.int64, device=device)
        v = (gs0(u)-u) & M
        tbl[v] = u.to(torch.int32)
        del u, v
        torch.cuda.empty_cache()
    n = (tbl != SEN32).sum().item()
    print(f"  {n:,} entries ({n/2**32*100:.1f}%), {time.time()-t0:.1f}s", flush=True)
    return tbl

def tlookup(tbl, t):
    r = torch.full_like(t, -1, dtype=torch.int64)
    raw = tbl[t.long()]; ok = raw != SEN32
    if ok.any(): r[ok] = raw[ok].to(torch.int64) & M
    return r

def find_w9_seeds(w9g_fn, W9_init, n_seeds=500, seed_batch=2**20):
    """Return up to n_seeds (a2,a3) pairs with W9_lo(a2,a3) == W9_init & 0xFFFF."""
    target_lo = W9_init & LO
    found_a2, found_a3 = [], []
    checked = 0
    while len(found_a2) < n_seeds and checked < 50 * n_seeds * seed_batch:
        a2v = torch.randint(0, 2**32, (seed_batch,), dtype=torch.int64, device=device)
        a3v = torch.randint(0, 2**32, (seed_batch,), dtype=torch.int64, device=device)
        mask = (w9g_fn(a2v, a3v) & LO) == target_lo
        if mask.any():
            idx = torch.where(mask)[0][:n_seeds - len(found_a2)]
            found_a2.extend(a2v[idx].cpu().tolist())
            found_a3.extend(a3v[idx].cpu().tolist())
        checked += seed_batch
        del a2v, a3v, mask
    return found_a2[:n_seeds], found_a3[:n_seeds]

def run_ctx(tbl, hash_bytes, known_a, R=19, max_iter=8, batch=2**24, K_seeds=4):
    a4=known_a[4]; a5=known_a[5]; a6=known_a[6]
    a7=known_a[7]; a8=known_a[8]; a9=known_a[9]; a10=known_a[10]; a11=known_a[11]
    T2_7=(big_sigma0(a6)+maj(a6,a5,a4))&M; T1_7=(a7-T2_7)&M
    T2_8=(big_sigma0(a7)+maj(a7,a6,a5))&M; T1_8=(a8-T2_8)&M; e8=(a4+T1_8)&M
    T2_9=(big_sigma0(a8)+maj(a8,a7,a6))&M; T1_9=(a9-T2_9)&M; e9=(a5+T1_9)&M
    T2_10=(big_sigma0(a9)+maj(a9,a8,a7))&M; T1_10=(a10-T2_10)&M
    T2_11=(big_sigma0(a10)+maj(a10,a9,a8))&M; T1_11=(a11-T2_11)&M
    e10=(a6+T1_10)&M
    CONST_10=(T1_10-big_sigma1(e9)-K[10])&M
    W11_base=(T1_11-T1_7-big_sigma1(e10)-ch(e10,e9,e8)-K[11])&M
    W9_base=(T1_9-big_sigma1(e8)-K[9])&M
    S0a4=big_sigma0(a4); S0a5=big_sigma0(a5)

    def w9g(a2v,a3v):
        T16=(a6-S0a5-gmaj(a5,a4,a3v))&M; e6=(a2v+T16)&M
        e7=(a3v+T1_7)&M; T15=(a5-S0a4-gmaj(a4,a3v,a2v))&M
        return (W9_base-T15-gch(e8,e7,e6))&M

    W9_init=int(w9g(torch.tensor(0,dtype=torch.int64,device=device),
                    torch.tensor(0,dtype=torch.int64,device=device)).item())
    W9_lo_tgt=W9_init&LO; W9_hi_tgt_ctx=(W9_init>>16)&LO

    # Pre-compute W9_lo-consistent seed starts (~3.3× more W9_lo events per compute)
    if K_seeds > 0:
        seeds_a2, seeds_a3 = find_w9_seeds(w9g, W9_init, n_seeds=max(K_seeds*50, 200))
        seeds_a2 = seeds_a2[:K_seeds]; seeds_a3 = seeds_a3[:K_seeds]
        if len(seeds_a2) < K_seeds:
            # Fallback: pad with (0,0) if not enough seeds found
            seeds_a2 += [0] * (K_seeds - len(seeds_a2))
            seeds_a3 += [0] * (K_seeds - len(seeds_a3))
    else:
        seeds_a2, seeds_a3 = [0], [0]  # original (0,0) start

    ka0=dict(known_a)
    for r in [0,1,2,3]: ka0[r]=0
    ke0=compute_e_from_a(ka0,R); kW0=recover_W(ka0,ke0,R)
    W14=kW0.get(14,0); W15=kW0.get(15,0)
    W16r=kW0.get(16,0); W17r=kW0.get(17,0); W18r=kW0.get(18,0)

    def cs(x): return torch.tensor(x,dtype=torch.int64,device=device)
    g_e8=cs(e8); g_e9=cs(e9); g_T1_7=cs(T1_7)
    g_C10=cs(CONST_10); g_W9b=cs(W9_base)
    g_S0a4=cs(S0a4); g_S0a5=cs(S0a5)
    g_a4=cs(a4); g_a5=cs(a5); g_a6=cs(a6)
    g_W11b=cs(W11_base); g_W9i=cs(W9_init)
    g_W9lo=cs(W9_lo_tgt); g_W9hi=cs(W9_hi_tgt_ctx)

    stats=dict(c0=0,conv=0,lo=0,hi=0,cons=0,verified=0)
    residuals=[]; found=None

    for bs in range(0,2**32,batch):
        a0=torch.arange(bs,bs+batch,dtype=torch.int64,device=device)
        e0v=(a0+CONST_E0)&M; W0v=(a0+C0_CONST)&M
        gv=(-gS0(a0)-gmaj(a0,a_m1,a_m2)-e_m3-gS1(e0v)-gch(e0v,e_m1,e_m2)-K[1])&M
        F0=(W16r-small_sigma1(W14)-gv-g_W9i-a0-C0_CONST)&M
        W1v=tlookup(tbl,F0); alive=W1v>=0; n0=alive.sum().item()
        if n0==0: continue
        stats['c0']+=n0
        a0a=a0[alive]; gva=gv[alive]; e0a=e0v[alive]; W0a=W0v[alive]; W1a=W1v[alive]
        a1a=(W1a-gva)&M
        e1a=(a_m3+a1a-gS0(a0a)-gmaj(a0a,a_m1,a_m2))&M
        T2_2a=(gS0(a1a)+gmaj(a1a,a0a,a_m1))&M
        F12a=(-T2_2a-e_m2-gS1(e1a)-gch(e1a,e0a,e_m1)-K[2])&M
        # K-seed iteration: run max_iter C1→C2 steps from each seed start
        for sk in range(len(seeds_a2)):
            a2c=torch.full((n0,),int(seeds_a2[sk]),dtype=torch.int64,device=device)
            a3c=torch.full((n0,),int(seeds_a3[sk]),dtype=torch.int64,device=device)

            for it in range(max_iter):
                T16a=(g_a6-g_S0a5-gmaj(g_a5,g_a4,a3c))&M
                Da3=(T16a+gch(g_e9,g_e8,(a3c+g_T1_7)&M))&M
                T_c1=(W17r-small_sigma1(W15)-W1a-g_C10-F12a+Da3)&M
                W2v=tlookup(tbl,T_c1); h1=W2v>=0
                W2v=torch.where(h1,W2v,torch.zeros_like(W2v)); a2n=(W2v-F12a)&M
                W16s=(small_sigma1(W14)+g_W9i+gs0(W1a)+W0a)&M
                W16s_corr=(W16s-a1a)&M  # w9g omits a1; correct: W16_real = W16s - a1
                e2v=(a_m2+a2n-T2_2a)&M
                T2_3v=(gS0(a2n)+gmaj(a2n,a1a,a0a))&M
                F23v=(-T2_3v-e_m1-gS1(e2v)-gch(e2v,e1a,e0a)-K[3])&M
                T_c2=(W18r-gs1(W16s_corr)-g_W11b-W2v-F23v)&M  # use corrected W16
                W3v=tlookup(tbl,T_c2); h2=W3v>=0
                W3v=torch.where(h2,W3v,torch.zeros_like(W3v)); a3n=(W3v-F23v)&M
                both=h1&h2
                a2n=torch.where(both,a2n,a2c); a3n=torch.where(both,a3n,a3c)
                conv=both&(a2n==a2c)&(a3n==a3c)
                if conv.any():
                    stats['conv']+=conv.sum().item()
                    a2f=a2c[conv]; a3f=a3c[conv]
                    a0f=a0a[conv]; a1f=a1a[conv]; W1f=W1a[conv]; gvf=gva[conv]; W0f=W0a[conv]
                    T16f=(g_a6-g_S0a5-gmaj(g_a5,g_a4,a3f))&M
                    e6f=(a2f+T16f)&M; e7f=(a3f+g_T1_7)&M
                    T15f=(g_a5-g_S0a4-gmaj(g_a4,a3f,a2f))&M
                    W9f=(g_W9b-T15f-gch(g_e8,e7f,e6f))&M

                    # Lo filter
                    lo=(W9f&LO)==g_W9lo; stats['lo']+=lo.sum().item()

                    if lo.any():
                        # Record W9_hi residuals — THE KEY DIAGNOSTIC
                        W9f_hi=((W9f[lo]>>16)&LO).cpu().numpy()
                        resid=((W9f_hi.astype(int)-int(W9_hi_tgt_ctx))%(1<<16)).tolist()
                        residuals.extend(resid)

                        # Hi filter
                        hi=(((W9f[lo])>>16)&LO)==g_W9hi; stats['hi']+=hi.sum().item()

                        if hi.any():
                            # W9 full match — C0+C1+C2 all satisfied
                            a0_v=a0f[lo][hi]; a1_v=a1f[lo][hi]
                            a2_v=a2f[lo][hi]; a3_v=a3f[lo][hi]
                            W9_v=W9f[lo][hi]; gv_v=gvf[lo][hi]

                            cons = hi  # all hi_pass consistent by construction
                            stats['cons']+=hi.sum().item()
                            F_rc=(W16r-small_sigma1(W14)-gv_v-W9_v-a0_v-C0_CONST)&M
                            W1_rc=tlookup(tbl,F_rc)
                            recheck=(W1_rc==W1f[lo][hi])
                            n_mismatch=(~recheck).sum().item()
                            if n_mismatch>0:
                                print(f'  [WARN] {n_mismatch} hi_pass recheck mismatch', flush=True)

                            for i in range(min(cons.sum().item(),20)):
                                ci=torch.where(cons)[0][i].item()
                                kav=dict(known_a)
                                kav.update({0:int(a0_v[ci]),1:int(a1_v[ci]),
                                           2:int(a2_v[ci]),3:int(a3_v[ci])})
                                kev=compute_e_from_a(kav,R); kWv=recover_W(kav,kev,R)
                                mw=[kWv.get(r,0) for r in range(16)]
                                if verify_preimage(mw,hash_bytes,R):
                                    stats['verified']+=1; found=mw
                                    print(f"\n{'*'*65}")
                                    print(f"*** 19-ROUND SHA-256 PREIMAGE FOUND ***")
                                    print(f"  hash={hash_bytes.hex()}")
                                    print(f"  a0=0x{int(a0_v[ci]):08x} a1=0x{int(a1_v[ci]):08x}")
                                    print(f"  a2=0x{int(a2_v[ci]):08x} a3=0x{int(a3_v[ci]):08x}")
                                    print(f"  msg={[hex(w) for w in mw]}", flush=True)
                                    out_path = os.path.join(os.getcwd(), 'PREIMAGE_FOUND.txt')
                                    with open(out_path,'w') as f:
                                        f.write(f"hash={hash_bytes.hex()}\n")
                                        f.write(f"msg_words={[hex(w) for w in mw]}\n")
                                    print(f"  saved → {out_path}", flush=True)
                                    return stats, residuals, found

                a2c=torch.where(both&~conv,a2n,a2c)
                a3c=torch.where(both&~conv,a3n,a3c)

    return stats, residuals, found

def analyze_residuals(all_residuals, n_ctx):
    if not all_residuals:
        print("No lo_pass residuals collected yet.", flush=True)
        return
    arr=np.array(all_residuals)
    n=len(arr)
    zero=np.sum(arr==0)
    expected_zero=n/65536
    hist,_=np.histogram(arr,bins=256,range=(0,65536))
    exp_per_bin=n/256
    chi2=np.sum((hist-exp_per_bin)**2/exp_per_bin) if exp_per_bin>0 else 0

    print(f"\n{'='*65}")
    print(f"W9_hi RESIDUAL HISTOGRAM ({n_ctx} contexts, {n} lo_pass events)")
    print(f"{'='*65}")
    print(f"  hi==target (residual=0): {zero}   expected if uniform: {expected_zero:.2f}")
    print(f"  Chi-squared (256 bins):  {chi2:.1f}  (uniform baseline: ~255)")
    near_zero=np.sum(arr<256)
    print(f"  Residual in [0,255]:     {near_zero}  ({near_zero/n*100:.2f}%,  expected {256/65536*100:.2f}%)")

    if chi2 > 350:
        print("\n  RESULT: SIGNIFICANT NON-UNIFORMITY DETECTED")
        print("  Conclusion: structural anti-correlation between lo and hi.")
        print("  The 1/2^16 model is WRONG. True lambda is lower.")
        print("  Budget for 5000+ contexts, or revisit the model.")
    elif chi2 < 200:
        print("\n  RESULT: SIGNIFICANTLY BELOW UNIFORM BASELINE")
        print("  Something is wrong — check for bugs in the residual calc.")
    else:
        print("\n  RESULT: CONSISTENT WITH UNIFORM")
        print("  The 1/2^16 model holds. Zero hits is bad luck.")
        print("  72% at 2000 contexts is a valid estimate.")
    print(f"{'='*65}\n", flush=True)

def main():
    import argparse
    ap=argparse.ArgumentParser(description='Oracle-free R=19 SHA-256 preimage solver (H100)')
    ap.add_argument('--hash','-t', default=None,
        help='Target hash as 64 hex chars. If omitted, a random target is generated each context.')
    ap.add_argument('--contexts','-n', type=int, default=2000,
        help='Max contexts to try (default 2000).')
    ap.add_argument('--max-iter', type=int, default=8,
        help='C1/C2 iteration limit (default 8).')
    ap.add_argument('--k-seeds', type=int, default=4,
        help='W9_lo-consistent seed pairs per context (default 4).')
    args=ap.parse_args()

    tbl=build_table()
    R=19; n_ctx=args.contexts; max_iter=args.max_iter; K_seeds=args.k_seeds

    fixed_hash=None
    if args.hash:
        h=args.hash.strip().replace(' ','')
        if len(h)!=64: ap.error('--hash must be exactly 64 hex characters')
        fixed_hash=bytes.fromhex(h)
        print(f"Target hash: {fixed_hash.hex()}")
    print(f"Running up to {n_ctx} contexts, max_iter={max_iter}, K_seeds={K_seeds}")
    print(f"  (K_seeds=4 → ~3.3× more W9_lo events per compute vs (0,0) start)")
    print(f"{'ctx':>4} {'c0':>14} {'conv':>10} {'lo':>8} {'hi':>6} {'cons':>5} {'ver':>4} {'t':>6}")
    print("-"*65, flush=True)

    all_residuals=[]; totals=dict(c0=0,conv=0,lo=0,hi=0,cons=0,verified=0)
    t_wall=0; t0_total=time.time()
    REPORT_EVERY=100  # print residual analysis every 100 contexts

    for ci in range(n_ctx):
        if fixed_hash is not None:
            hb=fixed_hash
        else:
            msg=os.urandom(55)
            trace=sha256_full_trace(msg,num_rounds=R); hb=trace.final_hash
        kab,_=backward_chain(hb,R); ka=dict(kab)
        for r in range(4,R-8):
            if r not in ka: ka[r]=int.from_bytes(os.urandom(4),'big')
        t0=time.time()
        stats,residuals,found=run_ctx(tbl,hb,ka,R,max_iter,K_seeds=K_seeds)
        el=time.time()-t0; t_wall+=el
        all_residuals.extend(residuals)

        print(f"{ci:>4} {stats['c0']:>14,} {stats['conv']:>10,} {stats['lo']:>8,} "
              f"{stats['hi']:>6} {stats['cons']:>5} {stats['verified']:>4} {el:>5.0f}s",flush=True)
        for k in totals: totals[k]+=stats[k]

        if found:
            tt=time.time()-t0_total
            print(f"\nTotal time: {tt:.1f}s ({tt/60:.1f} min)", flush=True)
            return

        # Periodic residual analysis
        if (ci+1) % REPORT_EVERY == 0:
            analyze_residuals(all_residuals, ci+1)

    # Final analysis
    analyze_residuals(all_residuals, n_ctx)
    tt=time.time()-t0_total
    print(f"\nCompleted {n_ctx} contexts in {tt:.0f}s ({tt/3600:.2f}h)")
    print(f"Totals: {totals}")
    out_npy = os.path.join(os.getcwd(), 'w9_hi_residuals_final.npy')
    np.save(out_npy, np.array(all_residuals))
    print(f"Residuals saved → {out_npy}")

if __name__=='__main__':
    main()
