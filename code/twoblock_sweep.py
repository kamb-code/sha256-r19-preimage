"""
Two-block near-collision sweep using the Nine-Step differential on BOTH blocks.

Block 1: (M, M') with ΔW[0]=+δ, ΔW[9]=-δ  →  ΔH1 (our best: 95/256 bits, word5=0)
Block 2: (N, N') with the same differential, starting from H1 and H1' respectively.

The schedule differential in block 2 is the same Nine-Step structure, but the
starting state differs by ΔH1. We sweep N[15] to find the minimum ΔH2 = H2'-H2.

Goal: Total two-block near-collision Hamming weight = bits(ΔH2).
"""
import os
os.environ['CUDA_PATH'] = '/usr'
import cupy as cp, numpy as np, sys, time
sys.path.insert(0, '/home/administrator/sha/sha256')
from utils import MASK32 as M, H0 as H0_cpu, K as K_cpu, small_sigma0 as s0, small_sigma1 as s1
from utils import big_sigma0 as S0, big_sigma1 as S1, ch, maj

DELTA = 0x0011e034

# Block 1 message (best one-word zero: word5=0, total=95/256)
W_BLOCK1 = [0x6ac550a3,0x604e4a61,0x3d096d9d,0xc0b47203,
             0xe3127bf5,0x59c5c56a,0x8dd4cf0d,0xede08902,
             0x1eae72a3,0x001c2641,0x195659ea,0x0e39df0d,
             0xb0a6feb0,0x9562b0a6,0xa56abdfe,0x82627b41]

# Compute H1 and H1' from block 1
def sched(w):
    ws = list(w)
    for r in range(16, 64):
        ws.append((s1(ws[r-2]) + ws[r-7] + s0(ws[r-15]) + ws[r-16]) & M)
    return ws

def compress(ws, IV):
    a,b,c,d,e,f,g,h = IV
    for r in range(64):
        T1 = (h + S1(e) + ch(e,f,g) + K_cpu[r] + ws[r]) & M
        T2 = (S0(a) + maj(a,b,c)) & M
        a,b,c,d,e,f,g,h = (T1+T2)&M, a,b,c, (d+T1)&M, e,f,g
    return [a,b,c,d,e,f,g,h]

W_BLOCK1p = list(W_BLOCK1)
W_BLOCK1p[0] = (W_BLOCK1p[0] + DELTA) & M
W_BLOCK1p[9] = (W_BLOCK1p[9] - DELTA) & M

H1  = compress(sched(W_BLOCK1),  H0_cpu)
H1p = compress(sched(W_BLOCK1p), H0_cpu)
DH1 = [(H1p[i] - H1[i]) & M for i in range(8)]
dh1_bits = sum(bin(x).count('1') for x in DH1)

print("Block 1 results:")
print(f"  H1  = {[hex(x) for x in H1]}")
print(f"  H1' = {[hex(x) for x in H1p]}")
print(f"  ΔH1 = {[hex(x) for x in DH1]}")
print(f"  |ΔH1| = {dh1_bits}/256 bits  (word5=0: {DH1[5]==0})")

# Block 2: use same W[0..14] structure, sweep W[15]
W_BLOCK2_BASE = list(W_BLOCK1[:15])  # same W[0..14] as block 1

CUDA_SRC = r"""
#define ROTR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define SIG0(x) (ROTR32(x,7)^ROTR32(x,18)^((x)>>3))
#define SIG1(x) (ROTR32(x,17)^ROTR32(x,19)^((x)>>10))
#define BIG0(x) (ROTR32(x,2)^ROTR32(x,13)^ROTR32(x,22))
#define BIG1(x) (ROTR32(x,6)^ROTR32(x,11)^ROTR32(x,25))
#define CH(e,f,g)  (((e)&(f))^((~(e))&(g)))
#define MAJ(a,b,c) (((a)&(b))^((a)&(c))^((b)&(c)))

extern "C" __global__ void twoblock_sweep(
    unsigned int* best_bits,
    unsigned int* hit_count, unsigned long long* hit_w15, unsigned char* hit_word,
    unsigned int* near_count, unsigned long long* near_w15, unsigned int* near_bits,
    int max_hits, unsigned int threshold,
    unsigned long long start, unsigned long long count,
    const unsigned int* W0_14_arr,
    const unsigned int* K_arr,
    const unsigned int* H1_arr,   // IV for M side (block 2)
    const unsigned int* H1p_arr,  // IV for M' side (block 2)
    unsigned int delta
) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    unsigned int W15 = (unsigned int)(start + tid);

    // Block 2 message schedule: N and N' = N + Nine-Step diff
    unsigned int W[64], Wp[64];
    for (int i=0;i<14;i++) { W[i]=W0_14_arr[i]; Wp[i]=W0_14_arr[i]; }
    W[14]=W0_14_arr[14]; Wp[14]=W0_14_arr[14];
    W[15]=W15; Wp[15]=W15;
    Wp[0] += delta;
    Wp[9] -= delta;
    for (int r=16;r<64;r++) {
        W[r]  = SIG1(W[r-2])  + W[r-7]  + SIG0(W[r-15])  + W[r-16];
        Wp[r] = SIG1(Wp[r-2]) + Wp[r-7] + SIG0(Wp[r-15]) + Wp[r-16];
    }

    // Compress with H1 as IV (M side)
    unsigned int a=H1_arr[0],b=H1_arr[1],cc=H1_arr[2],d=H1_arr[3];
    unsigned int e=H1_arr[4],f=H1_arr[5],g=H1_arr[6],h=H1_arr[7];
    for (int r=0;r<64;r++) {
        unsigned int T1=h+BIG1(e)+CH(e,f,g)+K_arr[r]+W[r];
        unsigned int T2=BIG0(a)+MAJ(a,b,cc);
        h=g;g=f;f=e;e=d+T1;d=cc;cc=b;b=a;a=T1+T2;
    }

    // Compress with H1' as IV, N' as message (M' side)
    unsigned int ap=H1p_arr[0],bp=H1p_arr[1],ccp=H1p_arr[2],dp=H1p_arr[3];
    unsigned int ep=H1p_arr[4],fp=H1p_arr[5],gp=H1p_arr[6],hp=H1p_arr[7];
    for (int r=0;r<64;r++) {
        unsigned int T1=hp+BIG1(ep)+CH(ep,fp,gp)+K_arr[r]+Wp[r];
        unsigned int T2=BIG0(ap)+MAJ(ap,bp,ccp);
        hp=gp;gp=fp;fp=ep;ep=dp+T1;dp=ccp;ccp=bp;bp=ap;ap=T1+T2;
    }

    unsigned int S[8]  = {a,b,cc,d,e,f,g,h};
    unsigned int Sp[8] = {ap,bp,ccp,dp,ep,fp,gp,hp};

    // ΔH2 = H2' - H2 (includes IV addition: H2 = state + H1, H2' = state' + H1')
    // For compression function output INCLUDING chaining: H2 = state + H1
    // We want Δ(H2) = Δstate + ΔH1
    unsigned int total = 0;
    for (int i=0;i<8;i++) {
        unsigned int dstate = (Sp[i] + H1p_arr[i]) - (S[i] + H1_arr[i]);
        total += __popc(dstate);
    }

    atomicMin(best_bits, total);

    if (total <= threshold) {
        unsigned int idx = atomicAdd(near_count, 1u);
        if ((int)idx < max_hits) {
            near_w15[idx] = (unsigned long long)(start+tid);
            near_bits[idx] = total;
        }
    }

    // Also check compression-function-only differential (no IV addition)
    unsigned int total_cf = 0;
    for (int i=0;i<8;i++) total_cf += __popc(Sp[i]-S[i]);

    for (int i=0;i<8;i++) {
        unsigned int dcf = Sp[i]-S[i];
        if (dcf == 0) {
            unsigned int idx = atomicAdd(hit_count, 1u);
            if ((int)idx < max_hits) {
                hit_w15[idx] = (unsigned long long)(start+tid);
                hit_word[idx] = (unsigned char)i;
            }
        }
    }
}
"""

print("\nCompiling kernel...")
mod    = cp.RawModule(code=CUDA_SRC, name_expressions=['twoblock_sweep'])
kernel = mod.get_function('twoblock_sweep')
print("Done.")

K_gpu   = cp.array(K_cpu,  dtype=cp.uint32)
H1_gpu  = cp.array(H1,     dtype=cp.uint32)
H1p_gpu = cp.array(H1p,    dtype=cp.uint32)
W014_gpu = cp.array(W_BLOCK2_BASE, dtype=cp.uint32)

MAX_HITS = 4096
BLOCK    = 256
TOTAL    = 1 << 32
BATCH    = 80_000_000

# Benchmark
print("\nBenchmarking...")
best_bits  = cp.array([256], dtype=cp.uint32)
hit_count  = cp.zeros(1, dtype=cp.uint32)
hit_w15    = cp.zeros(MAX_HITS, dtype=cp.uint64)
hit_word   = cp.zeros(MAX_HITS, dtype=cp.uint8)
near_count = cp.zeros(1, dtype=cp.uint32)
near_w15   = cp.zeros(MAX_HITS, dtype=cp.uint64)
near_bits  = cp.zeros(MAX_HITS, dtype=cp.uint32)

# Threshold: aim for < 75 bits (vs 95 for block 1)
THRESHOLD = 75

t0 = time.time()
grid = (BATCH + BLOCK - 1) // BLOCK
kernel((grid,), (BLOCK,), (
    best_bits, hit_count, hit_w15, hit_word,
    near_count, near_w15, near_bits,
    cp.int32(MAX_HITS), cp.uint32(THRESHOLD),
    cp.uint64(0), cp.uint64(BATCH),
    W014_gpu, K_gpu, H1_gpu, H1p_gpu, cp.uint32(DELTA)
))
cp.cuda.runtime.deviceSynchronize()
t1 = time.time()
rate = BATCH / (t1 - t0)
print(f"  {BATCH/1e6:.0f}M pairs in {t1-t0:.2f}s = {rate:.2e}/s")
print(f"  Full 2^32 ETA: {TOTAL/rate:.1f}s")

# Full sweep
print(f"\n{'='*65}")
print(f"FULL TWO-BLOCK SWEEP  (N[0..14] = same as block 1, N[15] swept)")
print(f"Threshold: {THRESHOLD} bits (vs block-1 best: {dh1_bits}/256)")
print(f"{'='*65}")

best_bits  = cp.array([256], dtype=cp.uint32)
hit_count  = cp.zeros(1, dtype=cp.uint32)
hit_w15    = cp.zeros(MAX_HITS, dtype=cp.uint64)
hit_word   = cp.zeros(MAX_HITS, dtype=cp.uint8)
near_count = cp.zeros(1, dtype=cp.uint32)
near_w15   = cp.zeros(MAX_HITS, dtype=cp.uint64)
near_bits  = cp.zeros(MAX_HITS, dtype=cp.uint32)

pos = 0
t_start = time.time()
bn = 0

while pos < TOTAL:
    end   = min(pos + BATCH, TOTAL)
    batch = end - pos
    grid  = (batch + BLOCK - 1) // BLOCK
    kernel((grid,), (BLOCK,), (
        best_bits, hit_count, hit_w15, hit_word,
        near_count, near_w15, near_bits,
        cp.int32(MAX_HITS), cp.uint32(THRESHOLD),
        cp.uint64(pos), cp.uint64(batch),
        W014_gpu, K_gpu, H1_gpu, H1p_gpu, cp.uint32(DELTA)
    ))
    cp.cuda.runtime.deviceSynchronize()

    bb = int(best_bits[0])
    nhits = int(hit_count[0])
    nnear = int(near_count[0])
    elapsed = time.time() - t_start
    pct = 100 * end / TOTAL
    eta = (TOTAL - end) / (end / elapsed)

    if bn % 5 == 0:
        print(f"  {pct:.1f}%  best={bb}/256  hits={nhits}  near={nnear}  ETA={eta:.0f}s")

    pos = end
    bn += 1

elapsed = time.time() - t_start
bb    = int(best_bits[0])
nhits = int(hit_count[0])
nnear = int(near_count[0])

print(f"\nSweep done: 2^32 in {elapsed:.1f}s ({elapsed/60:.1f}min)")
print(f"Best two-block Hamming weight (CF only): {bb}/256")
print(f"Zero-word hits: {nhits}")
print(f"Near-misses (≤{THRESHOLD}): {nnear}")

if nhits > 0:
    hw = hit_w15[:nhits].get().tolist()
    wd = hit_word[:nhits].get().tolist()
    print(f"\nZero-word hits:")
    for w15, wi in zip(hw, wd):
        print(f"  N[15]=0x{w15:08x}  word[{wi}]=0")

if nnear > 0:
    nw = near_w15[:min(nnear,20)].get().tolist()
    nb = near_bits[:min(nnear,20)].get().tolist()
    print(f"\nBest near-misses:")
    for w15, bits in sorted(zip(nw, nb), key=lambda x: x[1])[:10]:
        print(f"  N[15]=0x{w15:08x}  total={bits}/256")

print(f"\n{'='*65}")
print(f"COMPARISON")
print(f"  Block 1 (single block, CF):  {dh1_bits}/256 bits")
print(f"  Block 2 (CF output diff):    {bb}/256 bits")
print(f"  Improvement from two blocks: {dh1_bits - bb:+d} bits")
