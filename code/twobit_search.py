"""
Exhaustive 2-bit and 3-bit flip search around the 75/256 local minimum.

At N_BEST (75/256), every single bit flip worsens by ≥28.
This searches all C(512,2)=130,816 two-bit pairs for any joint improvement.
Then all C(512,3)≈22M three-bit triples.

GPU: evaluate all pairs/triples in a single batched kernel launch.
"""
import os
os.environ['CUDA_PATH'] = '/usr'
import cupy as cp, numpy as np, sys, time, itertools
sys.path.insert(0, '/home/administrator/sha/sha256')
from utils import MASK32 as M, H0 as H0_cpu, K as K_cpu, small_sigma0 as s0
from utils import small_sigma1 as s1, big_sigma0 as S0, big_sigma1 as S1, ch, maj

DELTA = 0x0011e034

W_B1 = [0x6ac550a3,0x604e4a61,0x3d096d9d,0xc0b47203,0xe3127bf5,0x59c5c56a,
         0x8dd4cf0d,0xede08902,0x1eae72a3,0x001c2641,0x195659ea,0x0e39df0d,
         0xb0a6feb0,0x9562b0a6,0xa56abdfe,0x82627b41]

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

/* Each thread evaluates one candidate with bits specified by flip_pairs[tid*6].
   flip_pairs: [wi0, bi0, wi1, bi1, wi2, bi2] per candidate (wi2/bi2 = 255 = skip for 2-flip)
*/
extern "C" __global__ void multiflip(
    const unsigned int* base_words,
    const unsigned char* flip_pairs,   /* N * 6 bytes */
    unsigned int*        out_hw,
    int                  N,
    const unsigned int*  K_arr,
    const unsigned int*  H1_arr,
    const unsigned int*  H1p_arr,
    unsigned int         delta
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    const unsigned char* fp = flip_pairs + tid * 6;

    unsigned int W[16], Wp[16];
    for (int i = 0; i < 16; i++) { W[i] = base_words[i]; Wp[i] = base_words[i]; }

    /* apply flips */
    for (int k = 0; k < 3; k++) {
        unsigned char wi = fp[k*2];
        unsigned char bi = fp[k*2+1];
        if (wi == 255) break;
        W[wi]  ^= (1u << bi);
        Wp[wi] ^= (1u << bi);
    }
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
    unsigned int ep=H1p_arr[4],fp2=H1p_arr[5],gp=H1p_arr[6],hp=H1p_arr[7];
    for (int r = 0; r < 64; r++) {
        unsigned int T1=hp+BIG1(ep)+CH(ep,fp2,gp)+K_arr[r]+WWp[r];
        unsigned int T2=BIG0(ap)+MAJ(ap,bp,ccp);
        hp=gp;gp=fp2;fp2=ep;ep=dp+T1;dp=ccp;ccp=bp;bp=ap;ap=T1+T2;
    }
    unsigned int tot = __popc(ap-a)+__popc(bp-b)+__popc(ccp-cc)+__popc(dp-d)
                     + __popc(ep-e)+__popc(fp2-f)+__popc(gp-g)+__popc(hp-h);
    out_hw[tid] = tot;
}
"""

print("Compiling...", flush=True)
mod    = cp.RawModule(code=CUDA_SRC, name_expressions=['multiflip'])
kernel = mod.get_function('multiflip')
K_gpu  = cp.array(K_cpu, dtype=cp.uint32)
H1_gpu = cp.array(H1,    dtype=cp.uint32)
H1p_gpu= cp.array(H1p,   dtype=cp.uint32)
base_gpu = cp.array(N_BEST, dtype=cp.uint32)
BLOCK  = 256

def ref_eval(N16):
    Np = list(N16); Np[0]=(Np[0]+DELTA)&M; Np[9]=(Np[9]-DELTA)&M
    H2  = compress(sched(N16), H1)
    H2p = compress(sched(Np),  H1p)
    return sum(bin((H2p[i]-H2[i])&M).count('1') for i in range(8))

baseline = ref_eval(N_BEST)
print(f"Baseline: {baseline}/256", flush=True)

# ── 2-bit flip search ─────────────────────────────────────────────────────────
print(f"\n{'='*65}", flush=True)
print("2-bit flip exhaustive search  (C(512,2) = 130,816 pairs)", flush=True)

bits = [(wi, bi) for wi in range(16) for bi in range(32)]  # 512 bit positions
pairs = list(itertools.combinations(range(512), 2))
N2 = len(pairs)
print(f"  {N2} pairs", flush=True)

fp_arr = np.zeros((N2, 6), dtype=np.uint8)
fp_arr[:, 4] = 255  # 3rd flip = skip
fp_arr[:, 5] = 255
for k, (i, j) in enumerate(pairs):
    fp_arr[k, 0] = i // 32
    fp_arr[k, 1] = i % 32
    fp_arr[k, 2] = j // 32
    fp_arr[k, 3] = j % 32

fp_gpu = cp.asarray(fp_arr)
hw_gpu = cp.zeros(N2, dtype=cp.uint32)
grid   = (N2 + BLOCK - 1) // BLOCK

t0 = time.time()
kernel((grid,), (BLOCK,), (base_gpu, fp_gpu.view(cp.uint8), hw_gpu, cp.int32(N2),
                            K_gpu, H1_gpu, H1p_gpu, cp.uint32(DELTA)))
cp.cuda.runtime.deviceSynchronize()
hw2 = hw_gpu.get()
elapsed = time.time() - t0

best2 = int(hw2.min())
bidx2 = int(hw2.argmin())
print(f"  Done in {elapsed:.2f}s.  Best 2-bit flip: {best2}/256  (pair idx {bidx2})", flush=True)

improvements2 = [(pairs[k], hw2[k]) for k in range(N2) if hw2[k] < baseline]
print(f"  Improving pairs (< {baseline}): {len(improvements2)}", flush=True)
if improvements2:
    improvements2.sort(key=lambda x: x[1])
    print(f"  Top improving pairs:", flush=True)
    for (i,j), hw in improvements2[:20]:
        wi0, bi0 = i//32, i%32
        wi1, bi1 = j//32, j%32
        print(f"    N[{wi0:2d}]b{bi0:2d} + N[{wi1:2d}]b{bi1:2d}  → {hw}/256  (Δ={hw-baseline:+d})", flush=True)

# ── 3-bit flip search ─────────────────────────────────────────────────────────
print(f"\n{'='*65}", flush=True)
print("3-bit flip search  (C(512,3) = 21,845,056 triples)", flush=True)

N3 = 512*511*510 // 6
print(f"  {N3:,} triples — batched in chunks of 2^22", flush=True)

CHUNK3 = 1 << 22  # 4M per launch
best3_hw  = baseline
best3_trio = None
triples_done = 0
t0 = time.time()

# Generate triples lazily
def gen_triple_chunk(start_triple_idx, chunk_size):
    """Generate up to chunk_size triples starting at start_triple_idx.
    Returns np.array of shape (actual_size, 6) dtype uint8."""
    rows = []
    count = 0
    # Enumerate triples in lexicographic order — use fast numpy combo
    # We'll sample them
    for i, j, k in itertools.islice(itertools.combinations(range(512), 3),
                                    start_triple_idx, start_triple_idx + chunk_size):
        rows.append([i//32, i%32, j//32, j%32, k//32, k%32])
        count += 1
        if count >= chunk_size:
            break
    return np.array(rows, dtype=np.uint8) if rows else None

# Faster: pregenerate all indices as numpy
print("  Generating all triple indices... (may take ~30s)", flush=True)
t_gen = time.time()
all_trios = np.array(list(itertools.combinations(range(512), 3)), dtype=np.uint16)
print(f"  Generated {len(all_trios):,} triples in {time.time()-t_gen:.1f}s", flush=True)

fp3 = np.zeros((len(all_trios), 6), dtype=np.uint8)
fp3[:, 0] = (all_trios[:, 0] // 32).astype(np.uint8)
fp3[:, 1] = (all_trios[:, 0] % 32).astype(np.uint8)
fp3[:, 2] = (all_trios[:, 1] // 32).astype(np.uint8)
fp3[:, 3] = (all_trios[:, 1] % 32).astype(np.uint8)
fp3[:, 4] = (all_trios[:, 2] // 32).astype(np.uint8)
fp3[:, 5] = (all_trios[:, 2] % 32).astype(np.uint8)
del all_trios

CHUNK = 1 << 22
n_chunks = (len(fp3) + CHUNK - 1) // CHUNK
print(f"  Evaluating in {n_chunks} GPU chunks of {CHUNK:,}...", flush=True)

t0 = time.time()
improvements3 = []
global_best3  = baseline

for ci in range(n_chunks):
    start = ci * CHUNK
    end   = min(start + CHUNK, len(fp3))
    sz    = end - start
    fp_c  = cp.asarray(fp3[start:end])
    hw_c  = cp.zeros(sz, dtype=cp.uint32)
    grid_c = (sz + BLOCK - 1) // BLOCK
    kernel((grid_c,), (BLOCK,), (base_gpu, fp_c.view(cp.uint8), hw_c, cp.int32(sz),
                                  K_gpu, H1_gpu, H1p_gpu, cp.uint32(DELTA)))
    cp.cuda.runtime.deviceSynchronize()
    hw_np = hw_c.get()
    bh = int(hw_np.min())
    if bh < global_best3:
        global_best3 = bh
        bi3 = int(hw_np.argmin()) + start
        row = fp3[bi3]
        print(f"  New best: {bh}/256  (chunk {ci}/{n_chunks}, trio "
              f"N[{row[0]}]b{row[1]}+N[{row[2]}]b{row[3]}+N[{row[4]}]b{row[5]})", flush=True)
    # Collect all improvements
    idx_imp = np.where(hw_np < baseline)[0]
    for k in idx_imp:
        improvements3.append((start+k, int(hw_np[k])))
    del fp_c, hw_c

    if (ci+1) % 5 == 0:
        done = end
        rate = done / (time.time()-t0)
        print(f"  Progress: {done:,}/{len(fp3):,}  best={global_best3}  "
              f"{rate:.0e} eval/s  {(len(fp3)-done)/rate:.0f}s left", flush=True)

elapsed3 = time.time()-t0
print(f"\n3-bit search done in {elapsed3:.1f}s.", flush=True)
print(f"Best 3-bit flip: {global_best3}/256", flush=True)
print(f"Improving triples: {len(improvements3)}", flush=True)

if improvements3:
    improvements3.sort(key=lambda x: x[1])
    print("Top improvements:", flush=True)
    for idx, hw in improvements3[:20]:
        row = fp3[idx]
        print(f"  N[{row[0]}]b{row[1]} + N[{row[2]}]b{row[3]} + N[{row[4]}]b{row[5]}  "
              f"→ {hw}/256  (Δ={hw-baseline:+d})", flush=True)
        # Verify on CPU
        ww = list(N_BEST)
        for (wi,bi) in [(row[0],row[1]),(row[2],row[3]),(row[4],row[5])]:
            ww[wi] ^= (1 << bi)
        cpu_hw = ref_eval(ww)
        print(f"    CPU verify: {cpu_hw}/256", flush=True)
