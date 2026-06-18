"""
Bit-flip sensitivity matrix for the block-2 near-collision.

For each of 16×32=512 bit positions in N[0..15]:
  flip that bit, evaluate HW(ΔH2), compare to baseline.

Also evaluates the 16×32 matrix starting from the BASELINE (W_B1),
so we can see which bits most reduced the objective during coord descent.

Outputs: 512-entry sensitivity table + heatmap-style summary.
"""
import os
os.environ['CUDA_PATH'] = '/usr'
import cupy as cp, numpy as np, sys
sys.path.insert(0, '/home/administrator/sha/sha256')
from utils import MASK32 as M, H0 as H0_cpu, K as K_cpu, small_sigma0 as s0
from utils import small_sigma1 as s1, big_sigma0 as S0, big_sigma1 as S1, ch, maj

DELTA = 0x0011e034

W_B1 = [0x6ac550a3,0x604e4a61,0x3d096d9d,0xc0b47203,0xe3127bf5,0x59c5c56a,
         0x8dd4cf0d,0xede08902,0x1eae72a3,0x001c2641,0x195659ea,0x0e39df0d,
         0xb0a6feb0,0x9562b0a6,0xa56abdfe,0x82627b41]

# Best block-2 from coordinate descent
N_BEST = [0x254b45f2, 0xae6bc5c0, 0x531ffc39,
          0xc0b47203, 0xe3127bf5, 0x59c5c56a,
          0x8dd4cf0d, 0xede08902, 0x1eae72a3,
          0x001c2641, 0x195659ea, 0x0e39df0d,
          0xb0a6feb0, 0x9562b0a6, 0xa56abdfe, 0x82627b41]

def sched(w16):
    ws = list(w16)
    for r in range(16, 64):
        ws.append((s1(ws[r-2]) + ws[r-7] + s0(ws[r-15]) + ws[r-16]) & M)
    return ws

def compress(ws, iv):
    a,b,c,d,e,f,g,h = iv
    for r in range(64):
        T1 = (h + S1(e) + ch(e,f,g) + K_cpu[r] + ws[r]) & M
        T2 = (S0(a) + maj(a,b,c)) & M
        a,b,c,d,e,f,g,h = (T1+T2)&M, a,b,c, (d+T1)&M, e,f,g
    return [a,b,c,d,e,f,g,h]

W_B1p = list(W_B1); W_B1p[0]=(W_B1p[0]+DELTA)&M; W_B1p[9]=(W_B1p[9]-DELTA)&M
H1    = compress(sched(W_B1),  H0_cpu)
H1p   = compress(sched(W_B1p), H0_cpu)

CUDA_SRC = r"""
#define ROTR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define SIG0(x) (ROTR32(x,7)^ROTR32(x,18)^((x)>>3))
#define SIG1(x) (ROTR32(x,17)^ROTR32(x,19)^((x)>>10))
#define BIG0(x) (ROTR32(x,2)^ROTR32(x,13)^ROTR32(x,22))
#define BIG1(x) (ROTR32(x,6)^ROTR32(x,11)^ROTR32(x,25))
#define CH(e,f,g)  (((e)&(f))^((~(e))&(g)))
#define MAJ(a,b,c) (((a)&(b))^((a)&(c))^((b)&(c)))

/* One thread per (word_idx, bit_idx) pair: 512 threads total.
   Flips bit bit_idx of word word_idx and evaluates HW(ΔH2). */
extern "C" __global__ void sensitivity(
    const unsigned int* base_words,
    unsigned int*       out_hw,
    const unsigned int* K_arr,
    const unsigned int* H1_arr,
    const unsigned int* H1p_arr,
    unsigned int        delta
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 512) return;

    int word_idx = tid / 32;
    int bit_idx  = tid % 32;

    unsigned int W[16], Wp[16];
    for (int i = 0; i < 16; i++) { W[i] = base_words[i]; Wp[i] = base_words[i]; }
    W[word_idx]  ^= (1u << bit_idx);   /* flip one bit */
    Wp[word_idx] ^= (1u << bit_idx);
    Wp[0] += delta;
    Wp[9] -= delta;

    unsigned int WW[64], WWp[64];
    for (int i = 0; i < 16; i++) { WW[i] = W[i]; WWp[i] = Wp[i]; }
    for (int r = 16; r < 64; r++) {
        WW[r]  = SIG1(WW[r-2])  + WW[r-7]  + SIG0(WW[r-15])  + WW[r-16];
        WWp[r] = SIG1(WWp[r-2]) + WWp[r-7] + SIG0(WWp[r-15]) + WWp[r-16];
    }

    unsigned int a=H1_arr[0],b=H1_arr[1],cc=H1_arr[2],d=H1_arr[3];
    unsigned int e=H1_arr[4],f=H1_arr[5],g=H1_arr[6],h=H1_arr[7];
    for (int r = 0; r < 64; r++) {
        unsigned int T1=h+BIG1(e)+CH(e,f,g)+K_arr[r]+WW[r];
        unsigned int T2=BIG0(a)+MAJ(a,b,cc);
        h=g;g=f;f=e;e=d+T1;d=cc;cc=b;b=a;a=T1+T2;
    }
    unsigned int ap=H1p_arr[0],bp=H1p_arr[1],ccp=H1p_arr[2],dp=H1p_arr[3];
    unsigned int ep=H1p_arr[4],fp=H1p_arr[5],gp=H1p_arr[6],hp=H1p_arr[7];
    for (int r = 0; r < 64; r++) {
        unsigned int T1=hp+BIG1(ep)+CH(ep,fp,gp)+K_arr[r]+WWp[r];
        unsigned int T2=BIG0(ap)+MAJ(ap,bp,ccp);
        hp=gp;gp=fp;fp=ep;ep=dp+T1;dp=ccp;ccp=bp;bp=ap;ap=T1+T2;
    }
    unsigned int tot = __popc(ap-a)+__popc(bp-b)+__popc(ccp-cc)+__popc(dp-d)
                     + __popc(ep-e)+__popc(fp-f)+__popc(gp-g)+__popc(hp-h);
    out_hw[tid] = tot;
}
"""

import time
print("Compiling...", flush=True)
mod    = cp.RawModule(code=CUDA_SRC, name_expressions=['sensitivity'])
kernel = mod.get_function('sensitivity')
K_gpu  = cp.array(K_cpu, dtype=cp.uint32)
H1_gpu = cp.array(H1,    dtype=cp.uint32)
H1p_gpu= cp.array(H1p,   dtype=cp.uint32)

def run_sensitivity(words, label):
    base_gpu = cp.array(words, dtype=cp.uint32)
    hw_gpu   = cp.zeros(512, dtype=cp.uint32)
    kernel((2,), (256,), (base_gpu, hw_gpu, K_gpu, H1_gpu, H1p_gpu, cp.uint32(DELTA)))
    cp.cuda.runtime.deviceSynchronize()
    hw = hw_gpu.get().reshape(16, 32)

    def ref_hw(w):
        wp = list(w); wp[0]=(wp[0]+DELTA)&M; wp[9]=(wp[9]-DELTA)&M
        H2  = compress(sched(list(w)), H1)
        H2p = compress(sched(wp), H1p)
        return sum(bin((H2p[i]-H2[i])&M).count('1') for i in range(8))

    baseline = ref_hw(words)
    print(f"\n{'='*70}", flush=True)
    print(f"Sensitivity matrix: {label}  (baseline HW = {baseline}/256)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Entries = HW after flipping that bit. diff = HW - baseline.", flush=True)
    print(f"  Negative diff = that flip HELPS (reduces HW).", flush=True)
    print(f"  Zero diff     = insensitive bit.", flush=True)
    print(flush=True)

    # Per-word stats
    print(f"{'Word':>6}  {'min':>4}  {'max':>4}  {'mean':>5}  {'#improve':>8}  {'#neutral':>8}  {'#worsen':>7}", flush=True)
    for wi in range(16):
        diffs = hw[wi].astype(int) - baseline
        n_imp = int((diffs < 0).sum())
        n_neu = int((diffs == 0).sum())
        n_wor = int((diffs > 0).sum())
        marker = " *** CHANGED" if list(words)[wi] != W_B1[wi] else ""
        print(f"  N[{wi:2d}]  {diffs.min():>4}  {diffs.max():>4}  {diffs.mean():>5.1f}  "
              f"{n_imp:>8}  {n_neu:>8}  {n_wor:>7}{marker}", flush=True)

    print(flush=True)
    # Most improvable bits
    flat = hw.flatten().astype(int) - baseline
    sorted_idx = np.argsort(flat)
    print("  Top-15 most helpful single-bit flips:", flush=True)
    for k in range(min(15, 512)):
        idx = sorted_idx[k]
        wi, bi = idx//32, idx%32
        d = flat[idx]
        if d >= 0:
            break
        print(f"    N[{wi:2d}] bit {bi:2d}  → {baseline+d}/256  (Δ={d:+d})", flush=True)

    # Most insensitive words (zero-diff bit count)
    print(f"\n  Per-word neutral-bit count (higher = more insensitive):", flush=True)
    for wi in range(16):
        n_zero = int((hw[wi].astype(int) == baseline).sum())
        bar = '#' * (n_zero // 2)
        print(f"    N[{wi:2d}]  {n_zero:2d}/32 neutral  {bar}", flush=True)

    return hw, baseline

t0 = time.time()
hw_best, base_best = run_sensitivity(N_BEST, "best block-2 (coord descent, 75/256)")
hw_wb1,  base_wb1  = run_sensitivity(W_B1,   "block-1 message  (baseline, 134/256)")
print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)

# Cross-comparison: which bits matter most going from 134→75?
print(f"\n{'='*70}", flush=True)
print("Cross-comparison: bits with largest improvement across both states", flush=True)
print(f"{'='*70}", flush=True)
# Bits that are consistently low-HW in best state vs high-HW in baseline
diff_matrix = hw_wb1.astype(int) - hw_best.astype(int)  # >0 means flip helped
flat_diff = diff_matrix.flatten()
top = np.argsort(-flat_diff)[:20]
for idx in top:
    wi, bi = idx//32, idx%32
    bit_val_b1   = (W_B1[wi] >> bi) & 1
    bit_val_best = (N_BEST[wi] >> bi) & 1
    if flat_diff[idx] <= 0:
        break
    print(f"  N[{wi:2d}] bit {bi:2d}  W_B1_bit={bit_val_b1}  best_bit={bit_val_best}  "
          f"improvement={flat_diff[idx]:+d}", flush=True)
