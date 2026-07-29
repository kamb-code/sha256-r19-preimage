"""
Coordinate descent on block-1 message itself.

We've been FIXED to W_B1 as the block-1 message (chosen for preimage attack).
For the near-collision, there's no reason to use W_B1.
Sweep each of 16 block-1 words over all 2^32 to minimise |ΔH1|
where H1  = compress(sched(W),   H0)
      H1' = compress(sched(W'),  H0)
      W'[0] = W[0]+δ, W'[9] = W[9]-δ, all others equal.

This may find |ΔH1| << 95/256 with a different block-1 message.
Then run block-2 coord descent from the new H1/H1'.
"""
import os
os.environ['CUDA_PATH'] = '/usr'
import cupy as cp, numpy as np, sys, time
sys.path.insert(0, '/home/administrator/sha/sha256')
from utils import MASK32 as M, H0 as H0_cpu, K as K_cpu, small_sigma0 as s0
from utils import small_sigma1 as s1, big_sigma0 as S0, big_sigma1 as S1, ch, maj

DELTA = 0x0011e034

W_B1 = [0x6ac550a3,0x604e4a61,0x3d096d9d,0xc0b47203,0xe3127bf5,0x59c5c56a,
         0x8dd4cf0d,0xede08902,0x1eae72a3,0x001c2641,0x195659ea,0x0e39df0d,
         0xb0a6feb0,0x9562b0a6,0xa56abdfe,0x82627b41]

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

def ref_hw1(W16):
    Wp = list(W16); Wp[0]=(Wp[0]+DELTA)&M; Wp[9]=(Wp[9]-DELTA)&M
    H1  = compress(sched(W16), H0_cpu)
    H1p = compress(sched(Wp),  H0_cpu)
    return sum(bin((H1p[i]-H1[i])&M).count('1') for i in range(8))

def ref_hw2(N16, H1, H1p):
    Np = list(N16); Np[0]=(Np[0]+DELTA)&M; Np[9]=(Np[9]-DELTA)&M
    H2  = compress(sched(N16), H1)
    H2p = compress(sched(Np),  H1p)
    return sum(bin((H2p[i]-H2[i])&M).count('1') for i in range(8))

# Kernel: sweep one word over 2^32 evaluating block-1 HW (standard IV = H0)
CUDA_B1 = r"""
#define ROTR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define SIG0(x) (ROTR32(x,7)^ROTR32(x,18)^((x)>>3))
#define SIG1(x) (ROTR32(x,17)^ROTR32(x,19)^((x)>>10))
#define BIG0(x) (ROTR32(x,2)^ROTR32(x,13)^ROTR32(x,22))
#define BIG1(x) (ROTR32(x,6)^ROTR32(x,11)^ROTR32(x,25))
#define CH(e,f,g)  (((e)&(f))^((~(e))&(g)))
#define MAJ(a,b,c) (((a)&(b))^((a)&(c))^((b)&(c)))

extern "C" __global__ void sweep_b1(
    const unsigned int* base_words,
    int                 word_idx,
    unsigned int        chunk_start,
    unsigned int        chunk_size,
    unsigned int*       out_min_hw,
    unsigned int*       out_min_val,
    const unsigned int* K_arr,
    const unsigned int* H0_arr,
    unsigned int        delta
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= chunk_size) return;

    unsigned int val = chunk_start + tid;
    unsigned int W[16], Wp[16];
    for (int i = 0; i < 16; i++) { W[i] = base_words[i]; Wp[i] = base_words[i]; }
    W[word_idx]  = val;
    Wp[word_idx] = val;
    Wp[0] += delta;
    Wp[9] -= delta;

    unsigned int WW[64], WWp[64];
    for (int i = 0; i < 16; i++) { WW[i] = W[i]; WWp[i] = Wp[i]; }
    for (int r = 16; r < 64; r++) {
        WW[r]  = SIG1(WW[r-2])  + WW[r-7]  + SIG0(WW[r-15])  + WW[r-16];
        WWp[r] = SIG1(WWp[r-2]) + WWp[r-7] + SIG0(WWp[r-15]) + WWp[r-16];
    }

    unsigned int a=H0_arr[0],b=H0_arr[1],cc=H0_arr[2],d=H0_arr[3];
    unsigned int e=H0_arr[4],f=H0_arr[5],g=H0_arr[6],h=H0_arr[7];
    for (int r = 0; r < 64; r++) {
        unsigned int T1=h+BIG1(e)+CH(e,f,g)+K_arr[r]+WW[r];
        unsigned int T2=BIG0(a)+MAJ(a,b,cc);
        h=g;g=f;f=e;e=d+T1;d=cc;cc=b;b=a;a=T1+T2;
    }
    unsigned int ap=H0_arr[0],bp=H0_arr[1],ccp=H0_arr[2],dp=H0_arr[3];
    unsigned int ep=H0_arr[4],fp=H0_arr[5],gp=H0_arr[6],hp=H0_arr[7];
    for (int r = 0; r < 64; r++) {
        unsigned int T1=hp+BIG1(ep)+CH(ep,fp,gp)+K_arr[r]+WWp[r];
        unsigned int T2=BIG0(ap)+MAJ(ap,bp,ccp);
        hp=gp;gp=fp;fp=ep;ep=dp+T1;dp=ccp;ccp=bp;bp=ap;ap=T1+T2;
    }
    unsigned int tot = __popc(ap-a)+__popc(bp-b)+__popc(ccp-cc)+__popc(dp-d)
                     + __popc(ep-e)+__popc(fp-f)+__popc(gp-g)+__popc(hp-h);
    out_min_hw[tid]  = tot;
    out_min_val[tid] = val;
}
"""

# Kernel: sweep one word of block-2 over 2^32
CUDA_B2 = r"""
#define ROTR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define SIG0(x) (ROTR32(x,7)^ROTR32(x,18)^((x)>>3))
#define SIG1(x) (ROTR32(x,17)^ROTR32(x,19)^((x)>>10))
#define BIG0(x) (ROTR32(x,2)^ROTR32(x,13)^ROTR32(x,22))
#define BIG1(x) (ROTR32(x,6)^ROTR32(x,11)^ROTR32(x,25))
#define CH(e,f,g)  (((e)&(f))^((~(e))&(g)))
#define MAJ(a,b,c) (((a)&(b))^((a)&(c))^((b)&(c)))

extern "C" __global__ void sweep_b2(
    const unsigned int* base_words,
    int                 word_idx,
    unsigned int        chunk_start,
    unsigned int        chunk_size,
    unsigned int*       out_min_hw,
    unsigned int*       out_min_val,
    const unsigned int* K_arr,
    const unsigned int* H1_arr,
    const unsigned int* H1p_arr,
    unsigned int        delta
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= chunk_size) return;

    unsigned int val = chunk_start + tid;
    unsigned int W[16], Wp[16];
    for (int i = 0; i < 16; i++) { W[i] = base_words[i]; Wp[i] = base_words[i]; }
    W[word_idx]  = val;
    Wp[word_idx] = val;
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
    out_min_hw[tid]  = tot;
    out_min_val[tid] = val;
}
"""

print("Compiling kernels...", flush=True)
mod_b1   = cp.RawModule(code=CUDA_B1, name_expressions=['sweep_b1'])
mod_b2   = cp.RawModule(code=CUDA_B2, name_expressions=['sweep_b2'])
kern_b1  = mod_b1.get_function('sweep_b1')
kern_b2  = mod_b2.get_function('sweep_b2')
K_gpu    = cp.array(K_cpu,  dtype=cp.uint32)
H0_gpu   = cp.array(H0_cpu, dtype=cp.uint32)
BLOCK    = 256
CHUNK    = 1 << 22
N_CHUNKS = (1 << 32) // CHUNK
print("Done.", flush=True)

def sweep_b1_word(cur_words, word_idx):
    base_gpu = cp.array(cur_words, dtype=cp.uint32)
    best_hw, best_val = 256, 0
    for chunk in range(N_CHUNKS):
        cs = np.uint32(chunk * CHUNK)
        hw_gpu  = cp.empty(CHUNK, dtype=cp.uint32)
        val_gpu = cp.empty(CHUNK, dtype=cp.uint32)
        grid = (CHUNK + BLOCK - 1) // BLOCK
        kern_b1((grid,), (BLOCK,), (
            base_gpu, cp.int32(word_idx),
            cp.uint32(cs), cp.uint32(CHUNK),
            hw_gpu, val_gpu, K_gpu, H0_gpu, cp.uint32(DELTA)
        ))
        cp.cuda.runtime.deviceSynchronize()
        idx = int(cp.argmin(hw_gpu))
        bh, bv = int(hw_gpu[idx]), int(val_gpu[idx])
        if bh < best_hw:
            best_hw, best_val = bh, bv
    return best_hw, best_val

def sweep_b2_word(cur_words, word_idx, H1, H1p):
    base_gpu = cp.array(cur_words, dtype=cp.uint32)
    H1_gpu   = cp.array(H1,  dtype=cp.uint32)
    H1p_gpu  = cp.array(H1p, dtype=cp.uint32)
    best_hw, best_val = 256, 0
    for chunk in range(N_CHUNKS):
        cs = np.uint32(chunk * CHUNK)
        hw_gpu  = cp.empty(CHUNK, dtype=cp.uint32)
        val_gpu = cp.empty(CHUNK, dtype=cp.uint32)
        grid = (CHUNK + BLOCK - 1) // BLOCK
        kern_b2((grid,), (BLOCK,), (
            base_gpu, cp.int32(word_idx),
            cp.uint32(cs), cp.uint32(CHUNK),
            hw_gpu, val_gpu, K_gpu, H1_gpu, H1p_gpu, cp.uint32(DELTA)
        ))
        cp.cuda.runtime.deviceSynchronize()
        idx = int(cp.argmin(hw_gpu))
        bh, bv = int(hw_gpu[idx]), int(val_gpu[idx])
        if bh < best_hw:
            best_hw, best_val = bh, bv
    return best_hw, best_val

def coord_descent_b1(start, max_passes=4):
    cur = list(start)
    cur_hw = ref_hw1(cur)
    print(f"\nBlock-1 coord descent  (start HW={cur_hw}/256)", flush=True)
    for p in range(max_passes):
        improved = False
        for wi in range(16):
            bh, bv = sweep_b1_word(cur, wi)
            if bh < cur_hw:
                print(f"  p={p} w[{wi:2d}]  {cur_hw}→{bh}  val=0x{bv:08x}", flush=True)
                cur[wi] = bv; cur_hw = bh; improved = True
        print(f"  Pass {p+1}: {cur_hw}/256", flush=True)
        if not improved:
            break
    return cur, cur_hw

def coord_descent_b2(start, H1, H1p, max_passes=4):
    cur = list(start)
    cur_hw = ref_hw2(cur, H1, H1p)
    print(f"  Block-2 coord descent  (start HW={cur_hw}/256)", flush=True)
    for p in range(max_passes):
        improved = False
        for wi in range(16):
            bh, bv = sweep_b2_word(cur, wi, H1, H1p)
            if bh < cur_hw:
                print(f"    p={p} w[{wi:2d}]  {cur_hw}→{bh}", flush=True)
                cur[wi] = bv; cur_hw = bh; improved = True
        print(f"    Pass {p+1}: {cur_hw}/256", flush=True)
        if not improved:
            break
    return cur, cur_hw

# ── Baseline ─────────────────────────────────────────────────────────────────
print(f"Baseline W_B1: |ΔH1| = {ref_hw1(W_B1)}/256", flush=True)

# ── Block-1 coordinate descent ────────────────────────────────────────────────
print(f"\n{'='*65}", flush=True)
print("Block-1 coordinate descent  (free all 16 words, minimise |ΔH1|)", flush=True)
W_best, dh1 = coord_descent_b1(W_B1)

print(f"\n  Block-1 best: {dh1}/256  (W_B1 was {ref_hw1(W_B1)}/256)", flush=True)
print(f"  Words changed from W_B1:", flush=True)
changed_b1 = [i for i in range(16) if W_best[i] != W_B1[i]]
for i in changed_b1:
    print(f"    W[{i:2d}]: 0x{W_B1[i]:08x} → 0x{W_best[i]:08x}", flush=True)

# Compute H1/H1' for best block-1
W_bestP = list(W_best); W_bestP[0]=(W_bestP[0]+DELTA)&M; W_bestP[9]=(W_bestP[9]-DELTA)&M
H1_best  = compress(sched(W_best),  H0_cpu)
H1p_best = compress(sched(W_bestP), H0_cpu)
dh1_check = sum(bin((H1p_best[i]-H1_best[i])&M).count('1') for i in range(8))
print(f"  CPU verify |ΔH1| = {dh1_check}/256", flush=True)
print(f"  ΔH1 per word: {[bin((H1p_best[i]-H1_best[i])&M).count('1') for i in range(8)]}", flush=True)

# ── Block-2 coord descent using new H1/H1' ────────────────────────────────────
print(f"\n{'='*65}", flush=True)
print("Block-2 coord descent using OPTIMISED block-1 H1/H1'", flush=True)
N_opt, dh2_opt = coord_descent_b2(W_B1, H1_best, H1p_best)

print(f"\n  TWO-BLOCK RESULT with optimised block-1:", flush=True)
print(f"    Block-1: {dh1}/256  Block-2: {dh2_opt}/256", flush=True)
cpu2 = ref_hw2(N_opt, H1_best, H1p_best)
print(f"    CPU verify block-2: {cpu2}/256", flush=True)

# ── Also try multiple random starting points for block-1 ─────────────────────
print(f"\n{'='*65}", flush=True)
print("Block-1 coord descent from random starts", flush=True)
rng = np.random.default_rng(1234)
global_b1_best = dh1
global_W_best  = W_best
global_N_best  = N_opt
global_dh2     = dh2_opt

for trial in range(4):
    start = [int(rng.integers(0, 2**32)) for _ in range(16)]
    W_t, dh1_t = coord_descent_b1(start)

    # Compute H1/H1'
    W_tP = list(W_t); W_tP[0]=(W_tP[0]+DELTA)&M; W_tP[9]=(W_tP[9]-DELTA)&M
    H1_t  = compress(sched(W_t),  H0_cpu)
    H1p_t = compress(sched(W_tP), H0_cpu)

    print(f"  Trial {trial+1}: block-1 = {dh1_t}/256", flush=True)
    if dh1_t <= global_b1_best + 5:
        N_t, dh2_t = coord_descent_b2(W_B1, H1_t, H1p_t)
        print(f"  Trial {trial+1}: block-2 = {dh2_t}/256", flush=True)
        if dh2_t < global_dh2:
            global_dh2 = dh2_t
            global_W_best = W_t
            global_N_best = N_t
            global_b1_best = dh1_t
            print(f"  *** NEW GLOBAL BEST: block-2 = {dh2_t}/256 ***", flush=True)

print(f"\n{'='*65}", flush=True)
print("FINAL RESULTS:", flush=True)
print(f"  Original (W_B1): |ΔH1|={ref_hw1(W_B1)}/256  block-2={75}/256", flush=True)
print(f"  Optimised block-1: |ΔH1|={global_b1_best}/256  block-2={global_dh2}/256", flush=True)
