"""
Coordinate-descent sweep over block-2 message words.

Starting from W_B1, sweep each word i over all 2^32 values (keeping others fixed),
accept the minimum, then repeat passes until convergence.

Block-2 uses H1/H1' as IVs, Nine-Step diff (N'[0]+=δ, N'[9]-=δ) fixed.
GPU: full 2^32 sweep per coordinate in ~5 seconds.
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

W_B1p = list(W_B1); W_B1p[0]=(W_B1p[0]+DELTA)&M; W_B1p[9]=(W_B1p[9]-DELTA)&M
H1    = compress(sched(W_B1),  H0_cpu)
H1p   = compress(sched(W_B1p), H0_cpu)
DH1_bits = sum(bin((H1p[i]-H1[i])&M).count('1') for i in range(8))
print(f"Block-1: |ΔH1| = {DH1_bits}/256 bits", flush=True)

CUDA_SRC = r"""
#define ROTR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define SIG0(x) (ROTR32(x,7)^ROTR32(x,18)^((x)>>3))
#define SIG1(x) (ROTR32(x,17)^ROTR32(x,19)^((x)>>10))
#define BIG0(x) (ROTR32(x,2)^ROTR32(x,13)^ROTR32(x,22))
#define BIG1(x) (ROTR32(x,6)^ROTR32(x,11)^ROTR32(x,25))
#define CH(e,f,g)  (((e)&(f))^((~(e))&(g)))
#define MAJ(a,b,c) (((a)&(b))^((a)&(c))^((b)&(c)))

/* Sweep word idx over all 2^32 values; base[] holds fixed words.
   chunk_start: first value in this launch's chunk.
   chunk_size:  number of values in this chunk.
*/
extern "C" __global__ void sweep_word(
    const unsigned int* base_words,   /* [16] fixed words */
    int                 word_idx,
    unsigned int        chunk_start,
    unsigned int        chunk_size,
    unsigned int*       out_min_hw,   /* per-thread best HW */
    unsigned int*       out_min_val,  /* corresponding word value */
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

print("Compiling kernel...", flush=True)
mod    = cp.RawModule(code=CUDA_SRC, name_expressions=['sweep_word'])
kernel = mod.get_function('sweep_word')
print("Done.", flush=True)

K_gpu   = cp.array(K_cpu, dtype=cp.uint32)
H1_gpu  = cp.array(H1,    dtype=cp.uint32)
H1p_gpu = cp.array(H1p,   dtype=cp.uint32)

BLOCK     = 256
CHUNK     = 1 << 22   # 4M values per launch (avoids timeout)
N_CHUNKS  = (1 << 32) // CHUNK  # 1024 chunks

def sweep_one_word(cur_words, word_idx):
    """Sweep word word_idx over all 2^32 values; return (best_hw, best_val)."""
    base_gpu = cp.array(cur_words, dtype=cp.uint32)
    global_best_hw  = 256
    global_best_val = 0

    for chunk in range(N_CHUNKS):
        chunk_start = np.uint32(chunk * CHUNK)
        hw_gpu  = cp.empty(CHUNK, dtype=cp.uint32)
        val_gpu = cp.empty(CHUNK, dtype=cp.uint32)
        grid = (CHUNK + BLOCK - 1) // BLOCK
        kernel((grid,), (BLOCK,), (
            base_gpu, cp.int32(word_idx),
            cp.uint32(chunk_start), cp.uint32(CHUNK),
            hw_gpu, val_gpu,
            K_gpu, H1_gpu, H1p_gpu, cp.uint32(DELTA)
        ))
        cp.cuda.runtime.deviceSynchronize()
        idx = int(cp.argmin(hw_gpu))
        bh  = int(hw_gpu[idx])
        bv  = int(val_gpu[idx])
        if bh < global_best_hw:
            global_best_hw  = bh
            global_best_val = bv

    return global_best_hw, global_best_val

def ref_eval(N16):
    Np = list(N16); Np[0]=(Np[0]+DELTA)&M; Np[9]=(Np[9]-DELTA)&M
    H2  = compress(sched(N16), H1)
    H2p = compress(sched(Np),  H1p)
    diff = [(H2p[i]-H2[i])&M for i in range(8)]
    return sum(bin(x).count('1') for x in diff)

# ── Initial baseline ──────────────────────────────────────────────────────────
cur_words = list(W_B1)
cur_hw    = ref_eval(cur_words)
print(f"\nBaseline (block-1 message): {cur_hw}/256 bits", flush=True)

# ── Coordinate descent ────────────────────────────────────────────────────────
MAX_PASSES = 6
print(f"\n{'='*65}", flush=True)
print("Coordinate descent: sweep each of 16 words over 2^32", flush=True)
print(f"{'='*65}", flush=True)

for pass_no in range(MAX_PASSES):
    improved = False
    pass_start = time.time()
    print(f"\n--- Pass {pass_no+1} ---", flush=True)

    for wi in range(16):
        t0 = time.time()
        bh, bv = sweep_one_word(cur_words, wi)
        elapsed = time.time() - t0
        marker = ""
        if bh < cur_hw:
            marker = f"  *** IMPROVED {cur_hw}→{bh} ***"
            cur_words[wi] = bv
            cur_hw = bh
            improved = True
        print(f"  w[{wi:2d}]  best={bh:3d}/256  val=0x{bv:08x}  ({elapsed:.1f}s){marker}", flush=True)

    pass_elapsed = time.time() - pass_start
    print(f"  Pass {pass_no+1} done in {pass_elapsed:.0f}s.  Current best: {cur_hw}/256", flush=True)

    if not improved:
        print("  No improvement this pass — converged.", flush=True)
        break

# ── Final report ──────────────────────────────────────────────────────────────
print(f"\n{'='*65}", flush=True)
print(f"FINAL: Coordinate descent best = {cur_hw}/256 bits", flush=True)
print(f"  CPU verification: {ref_eval(cur_words)}/256 bits", flush=True)

Np = list(cur_words); Np[0]=(Np[0]+DELTA)&M; Np[9]=(Np[9]-DELTA)&M
H2  = compress(sched(cur_words), H1)
H2p = compress(sched(Np),  H1p)
diff = [(H2p[i]-H2[i])&M for i in range(8)]
zeros = [i for i in range(8) if diff[i]==0]

print(f"\n  Block-2 N[0..15]:", flush=True)
for i, w in enumerate(cur_words):
    print(f"    N[{i:2d}] = 0x{w:08x}", flush=True)
print(f"\n  ΔH2 zero words: {zeros}", flush=True)
for i in range(8):
    b = bin(diff[i]).count('1')
    z = " ← ZERO" if diff[i]==0 else f"  ({b} bits)"
    print(f"    word[{i}] = 0x{diff[i]:08x}{z}", flush=True)
