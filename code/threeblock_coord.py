"""
Three-block near-collision extension.

Block-1: Nine-Step (0,9) → H1/H1', |ΔH1|=95/256
Block-2: Coord descent → H2/H2', |ΔH2|=75/256 (best known)
Block-3: Coord descent over N2[0..15] to minimise |ΔH3|

Also tries alternative block-1 differentials to find H1/H1' pairs
with smaller initial difference that may compress better in block-2.
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

# Best block-2 from coord descent
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

def ref_hw(N16, H_iv, H_ivp, delta):
    Np = list(N16); Np[0]=(Np[0]+delta)&M; Np[9]=(Np[9]-delta)&M
    H  = compress(sched(N16), H_iv)
    Hp = compress(sched(Np),  H_ivp)
    return sum(bin((Hp[i]-H[i])&M).count('1') for i in range(8))

# Compute block-1 outputs
W_B1p = list(W_B1); W_B1p[0]=(W_B1p[0]+DELTA)&M; W_B1p[9]=(W_B1p[9]-DELTA)&M
H1  = compress(sched(W_B1),  H0_cpu)
H1p = compress(sched(W_B1p), H0_cpu)
DH1 = sum(bin((H1p[i]-H1[i])&M).count('1') for i in range(8))

# Compute block-2 outputs (from N_BEST)
N_BESTp = list(N_BEST); N_BESTp[0]=(N_BESTp[0]+DELTA)&M; N_BESTp[9]=(N_BESTp[9]-DELTA)&M
H2  = compress(sched(N_BEST),  H1)
H2p = compress(sched(N_BESTp), H1p)
DH2 = sum(bin((H2p[i]-H2[i])&M).count('1') for i in range(8))

print(f"Block-1: |ΔH1| = {DH1}/256 bits", flush=True)
print(f"Block-2: |ΔH2| = {DH2}/256 bits  (coord descent best)", flush=True)
print(f"Block-3 IV: H2  = {[hex(x) for x in H2]}", flush=True)
print(f"Block-3 IV: H2' = {[hex(x) for x in H2p]}", flush=True)
DH2_bits = sum(bin((H2p[i]-H2[i])&M).count('1') for i in range(8))
print(f"Initial |ΔH2| = {DH2_bits}/256 for block-3", flush=True)

CUDA_SRC = r"""
#define ROTR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define SIG0(x) (ROTR32(x,7)^ROTR32(x,18)^((x)>>3))
#define SIG1(x) (ROTR32(x,17)^ROTR32(x,19)^((x)>>10))
#define BIG0(x) (ROTR32(x,2)^ROTR32(x,13)^ROTR32(x,22))
#define BIG1(x) (ROTR32(x,6)^ROTR32(x,11)^ROTR32(x,25))
#define CH(e,f,g)  (((e)&(f))^((~(e))&(g)))
#define MAJ(a,b,c) (((a)&(b))^((a)&(c))^((b)&(c)))

extern "C" __global__ void sweep_word(
    const unsigned int* base_words,
    int                 word_idx,
    unsigned int        chunk_start,
    unsigned int        chunk_size,
    unsigned int*       out_min_hw,
    unsigned int*       out_min_val,
    const unsigned int* K_arr,
    const unsigned int* Hiv_arr,
    const unsigned int* Hivp_arr,
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

    unsigned int a=Hiv_arr[0],b=Hiv_arr[1],cc=Hiv_arr[2],d=Hiv_arr[3];
    unsigned int e=Hiv_arr[4],f=Hiv_arr[5],g=Hiv_arr[6],h=Hiv_arr[7];
    for (int r = 0; r < 64; r++) {
        unsigned int T1=h+BIG1(e)+CH(e,f,g)+K_arr[r]+WW[r];
        unsigned int T2=BIG0(a)+MAJ(a,b,cc);
        h=g;g=f;f=e;e=d+T1;d=cc;cc=b;b=a;a=T1+T2;
    }
    unsigned int ap=Hivp_arr[0],bp=Hivp_arr[1],ccp=Hivp_arr[2],dp=Hivp_arr[3];
    unsigned int ep=Hivp_arr[4],fp=Hivp_arr[5],gp=Hivp_arr[6],hp=Hivp_arr[7];
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

print("\nCompiling sweep kernel...", flush=True)
mod    = cp.RawModule(code=CUDA_SRC, name_expressions=['sweep_word'])
kernel = mod.get_function('sweep_word')
K_gpu  = cp.array(K_cpu, dtype=cp.uint32)
BLOCK  = 256
CHUNK  = 1 << 22
N_CHUNKS = (1 << 32) // CHUNK

def sweep_one_word(cur_words, word_idx, Hiv, Hivp):
    base_gpu = cp.array(cur_words, dtype=cp.uint32)
    Hiv_gpu  = cp.array(Hiv,  dtype=cp.uint32)
    Hivp_gpu = cp.array(Hivp, dtype=cp.uint32)
    best_hw, best_val = 256, 0
    for chunk in range(N_CHUNKS):
        chunk_start = np.uint32(chunk * CHUNK)
        hw_gpu  = cp.empty(CHUNK, dtype=cp.uint32)
        val_gpu = cp.empty(CHUNK, dtype=cp.uint32)
        grid = (CHUNK + BLOCK - 1) // BLOCK
        kernel((grid,), (BLOCK,), (
            base_gpu, cp.int32(word_idx),
            cp.uint32(chunk_start), cp.uint32(CHUNK),
            hw_gpu, val_gpu,
            K_gpu, Hiv_gpu, Hivp_gpu, cp.uint32(DELTA)
        ))
        cp.cuda.runtime.deviceSynchronize()
        idx = int(cp.argmin(hw_gpu))
        bh  = int(hw_gpu[idx])
        bv  = int(val_gpu[idx])
        if bh < best_hw:
            best_hw, best_val = bh, bv
    return best_hw, best_val

def coord_descent(start_words, Hiv, Hivp, label, max_passes=4):
    cur = list(start_words)
    cur_hw = ref_hw(cur, Hiv, Hivp, DELTA)
    print(f"\n{label}", flush=True)
    print(f"  Starting HW: {cur_hw}/256", flush=True)

    for pass_no in range(max_passes):
        improved = False
        for wi in range(16):
            bh, bv = sweep_one_word(cur, wi, Hiv, Hivp)
            if bh < cur_hw:
                print(f"    p={pass_no} w[{wi:2d}]  {cur_hw}→{bh}  val=0x{bv:08x}", flush=True)
                cur[wi] = bv
                cur_hw  = bh
                improved = True
        print(f"  Pass {pass_no+1} done.  HW={cur_hw}/256", flush=True)
        if not improved:
            break

    return cur, cur_hw

# ── Block-3 Coord Descent ─────────────────────────────────────────────────────
print(f"\n{'='*65}", flush=True)
print("Block-3 coordinate descent  (starting from W_B1)", flush=True)
print(f"  IV = H2/H2'  (|ΔH2| = {DH2}/256)", flush=True)

N2_start = list(W_B1)
N2_best, DH3 = coord_descent(N2_start, H2, H2p, "=== Block-3 coord descent ===")

print(f"\n{'='*65}", flush=True)
print(f"RESULTS: Three-block chain", flush=True)
print(f"  Block-1: |ΔH1| = {DH1}/256", flush=True)
print(f"  Block-2: |ΔH2| = {DH2}/256", flush=True)
print(f"  Block-3: |ΔH3| = {DH3}/256", flush=True)
print(f"  Total improvement: {256 - DH3}/256", flush=True)

print(f"\n  Block-3 optimal N2[0..15]:", flush=True)
for i, w in enumerate(N2_best):
    mark = " ← CHANGED" if w != W_B1[i] else ""
    print(f"    N2[{i:2d}] = 0x{w:08x}{mark}", flush=True)

# CPU verify
cpu_dh3 = ref_hw(N2_best, H2, H2p, DELTA)
print(f"\n  CPU verify block-3: {cpu_dh3}/256", flush=True)

# ── Also try alternative starting point: W_B1 scrambled ──────────────────────
print(f"\n{'='*65}", flush=True)
print("Alternative block-3: random starts to find deeper minimum", flush=True)

rng = np.random.default_rng(42)
best_global = DH3
best_words  = N2_best

for trial in range(3):
    start = rng.integers(0, 2**32, size=16, dtype=np.uint64).astype(int)
    start = [int(x) & M for x in start]
    cw, ch3 = coord_descent(start, H2, H2p,
                             f"=== Block-3 random start {trial+1} ===", max_passes=3)
    if ch3 < best_global:
        best_global = ch3
        best_words  = cw
        print(f"  NEW GLOBAL BEST: {ch3}/256", flush=True)

print(f"\nFinal block-3 best: {best_global}/256", flush=True)
