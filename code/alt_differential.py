"""
Alternative block-1 differentials.

The Nine-Step (0,9) is the only pair achieving an 8-round dead zone.
But OTHER pairs have partial dead zones and may give LOWER |ΔH1|
depending on the specific message, enabling better two-block results.

Tests:
1. Reversed Nine-Step: ΔW[9]=+δ, ΔW[0]=-δ  (same dead zone, different H1/H1')
2. Best partial-dead pairs from schedule analysis: (13,4), (7,8), (12,3), (4,13)
3. For each, run coordinate descent on block-2 to find best 2-block HW

For each differential:
  a) Compute H1/H1' using W_B1 with that differential applied
  b) Run coordinate descent over 16 block-2 words
  c) Report best 2-block near-collision HW
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

def hw_total(a, b):
    return sum(bin((a[i]-b[i])&M).count('1') for i in range(8))

def ref_hw_block2(N16, H1, H1p, delta, pos_plus, pos_minus):
    Np = list(N16)
    Np[pos_plus]  = (Np[pos_plus]  + delta) & M
    Np[pos_minus] = (Np[pos_minus] - delta) & M
    H2  = compress(sched(N16), H1)
    H2p = compress(sched(Np),  H1p)
    return sum(bin((H2p[i]-H2[i])&M).count('1') for i in range(8))

CUDA_SRC = r"""
#define ROTR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define SIG0(x) (ROTR32(x,7)^ROTR32(x,18)^((x)>>3))
#define SIG1(x) (ROTR32(x,17)^ROTR32(x,19)^((x)>>10))
#define BIG0(x) (ROTR32(x,2)^ROTR32(x,13)^ROTR32(x,22))
#define BIG1(x) (ROTR32(x,6)^ROTR32(x,11)^ROTR32(x,25))
#define CH(e,f,g)  (((e)&(f))^((~(e))&(g)))
#define MAJ(a,b,c) (((a)&(b))^((a)&(c))^((b)&(c)))

extern "C" __global__ void sweep_word_gen(
    const unsigned int* base_words,
    int                 word_idx,
    unsigned int        chunk_start,
    unsigned int        chunk_size,
    unsigned int*       out_min_hw,
    unsigned int*       out_min_val,
    const unsigned int* K_arr,
    const unsigned int* H1_arr,
    const unsigned int* H1p_arr,
    unsigned int        delta,
    int                 pos_plus,
    int                 pos_minus
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= chunk_size) return;

    unsigned int val = chunk_start + tid;

    unsigned int W[16], Wp[16];
    for (int i = 0; i < 16; i++) { W[i] = base_words[i]; Wp[i] = base_words[i]; }
    W[word_idx]  = val;
    Wp[word_idx] = val;
    Wp[pos_plus]  += delta;
    Wp[pos_minus] -= delta;

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

print("Compiling...", flush=True)
mod    = cp.RawModule(code=CUDA_SRC, name_expressions=['sweep_word_gen'])
kernel = mod.get_function('sweep_word_gen')
K_gpu  = cp.array(K_cpu, dtype=cp.uint32)
BLOCK  = 256
CHUNK  = 1 << 22
N_CHUNKS = (1 << 32) // CHUNK

def sweep_one_word(cur_words, word_idx, H1, H1p, pos_plus, pos_minus):
    base_gpu = cp.array(cur_words, dtype=cp.uint32)
    Hiv_gpu  = cp.array(H1,  dtype=cp.uint32)
    Hivp_gpu = cp.array(H1p, dtype=cp.uint32)
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
            K_gpu, Hiv_gpu, Hivp_gpu, cp.uint32(DELTA),
            cp.int32(pos_plus), cp.int32(pos_minus)
        ))
        cp.cuda.runtime.deviceSynchronize()
        idx = int(cp.argmin(hw_gpu))
        bh, bv = int(hw_gpu[idx]), int(val_gpu[idx])
        if bh < best_hw:
            best_hw, best_val = bh, bv
    return best_hw, best_val

def coord_descent_gen(start_words, H1, H1p, pos_plus, pos_minus, label, max_passes=4):
    cur    = list(start_words)
    cur_hw = ref_hw_block2(cur, H1, H1p, DELTA, pos_plus, pos_minus)
    print(f"\n{label}  (baseline={cur_hw}/256)", flush=True)
    for pass_no in range(max_passes):
        improved = False
        for wi in range(16):
            bh, bv = sweep_one_word(cur, wi, H1, H1p, pos_plus, pos_minus)
            if bh < cur_hw:
                print(f"  p={pass_no} w[{wi:2d}]  {cur_hw}→{bh}", flush=True)
                cur[wi] = bv; cur_hw = bh; improved = True
        print(f"  Pass {pass_no+1}: {cur_hw}/256", flush=True)
        if not improved:
            break
    return cur, cur_hw

# ── Test each differential ────────────────────────────────────────────────────
# (pos_plus, pos_minus, label, description)
differentials = [
    (0, 9,  "Nine-Step  (0,9) ORIGINAL",   "δ@W[0], -δ@W[9]"),
    (9, 0,  "Reversed   (9,0)",             "δ@W[9], -δ@W[0]"),
    (13,4,  "Pair (13,4) best partial",     "δ@W[13],-δ@W[4]"),
    (4,13,  "Pair (4,13) reversed",         "δ@W[4], -δ@W[13]"),
    (7, 8,  "Pair (7,8)  6 dead words",     "δ@W[7], -δ@W[8]"),
    (8, 7,  "Pair (8,7)  reversed",         "δ@W[8], -δ@W[7]"),
    (12,3,  "Pair (12,3) 5 dead words",     "δ@W[12],-δ@W[3]"),
    (3,12,  "Pair (3,12) reversed",         "δ@W[3], -δ@W[12]"),
]

results = []
for (pp, pm, label, desc) in differentials:
    print(f"\n{'='*65}", flush=True)
    print(f"Differential: {label}  ({desc})", flush=True)

    # Compute block-1 H1/H1' with this differential
    Wp = list(W_B1)
    Wp[pp] = (Wp[pp] + DELTA) & M
    Wp[pm] = (Wp[pm] - DELTA) & M
    H1  = compress(sched(W_B1), H0_cpu)
    H1p = compress(sched(Wp),   H0_cpu)
    dh1 = hw_total(H1p, H1)
    print(f"  |ΔH1| = {dh1}/256", flush=True)

    # Coord descent on block-2
    N16, best_hw = coord_descent_gen(W_B1, H1, H1p, pp, pm,
                                     f"Block-2 coord descent [{label}]", max_passes=3)
    results.append((label, dh1, best_hw, pp, pm, N16))

print(f"\n{'='*65}", flush=True)
print("SUMMARY: All differentials", flush=True)
print(f"{'Differential':35}  {'|ΔH1|':6}  {'Block-2 best':12}", flush=True)
for label, dh1, dh2, pp, pm, N16 in results:
    marker = " *** BEST" if dh2 == min(r[2] for r in results) else ""
    print(f"  {label:33}  {dh1:3d}/256   {dh2:3d}/256{marker}", flush=True)

# Report best
best_result = min(results, key=lambda x: x[2])
label, dh1, dh2, pp, pm, N16 = best_result
print(f"\nBest differential: {label}", flush=True)
print(f"  Block-1: {dh1}/256  Block-2: {dh2}/256", flush=True)
Np = list(N16); Np[pp]=(Np[pp]+DELTA)&M; Np[pm]=(Np[pm]-DELTA)&M
H1_b = compress(sched(list(W_B1)), H0_cpu)
H1p_b= compress(sched([list(W_B1)[i] if i!=pp and i!=pm else
                        (list(W_B1)[i]+DELTA)&M if i==pp else
                        (list(W_B1)[i]-DELTA)&M for i in range(16)]), H0_cpu)
cpu = hw_total(compress(sched(Np),  H1p_b),
               compress(sched(N16), H1_b))
print(f"  CPU verify: {cpu}/256", flush=True)
