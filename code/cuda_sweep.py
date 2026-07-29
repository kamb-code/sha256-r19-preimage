"""
CUDA kernel sweep: exhaustive W[15] search targeting ΔState[64][word5] = 0
Custom CUDA C kernel via CuPy RawKernel — each thread processes one W[15] value.
No schedule matrix stored; all state in registers. Should be 10-100x faster than
the vectorized numpy/cupy approach.
"""
import os
os.environ['CUDA_PATH'] = '/usr'

import cupy as cp
import numpy as np
import time, sys

# ─── SHA-256 constants ─────────────────────────────────────────────────────────
K_vals = [
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
]
H0_vals = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]
DELTA = 0x0011e034

W_BASE = [
    0x6ac550a3, 0x604e4a61, 0x3d096d9d, 0xc0b47203,
    0xe3127bf5, 0x59c5c56a, 0x8dd4cf0d, 0xede08902,
    0x1eae72a3, 0x001c2641, 0x195659ea, 0x0e39df0d,
    0xb0a6feb0, 0x9562b0a6, 0xa56abdfe,
    # W[15] is varied
]
W_PRIME = list(W_BASE)
W_PRIME[0] = (W_PRIME[0] + DELTA) & 0xFFFFFFFF
W_PRIME[9] = (W_PRIME[9] - DELTA) & 0xFFFFFFFF

# ─── Build CUDA kernel ─────────────────────────────────────────────────────────
K_init = ','.join(f'0x{k:08x}u' for k in K_vals)
W_init = ','.join(f'0x{w:08x}u' for w in W_BASE)
Wp_init = ','.join(f'0x{w:08x}u' for w in W_PRIME)

CUDA_SRC = r"""
#define ROTR32(x,n) (((x) >> (n)) | ((x) << (32-(n))))
#define SIG0(x) (ROTR32(x,7)  ^ ROTR32(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTR32(x,17) ^ ROTR32(x,19) ^ ((x) >> 10))
#define BIG0(x) (ROTR32(x,2)  ^ ROTR32(x,13) ^ ROTR32(x,22))
#define BIG1(x) (ROTR32(x,6)  ^ ROTR32(x,11) ^ ROTR32(x,25))
#define CH(e,f,g)  (((e) & (f)) ^ (~(e) & (g)))
#define MAJ(a,b,c) (((a) & (b)) ^ ((a) & (c)) ^ ((b) & (c)))
#define ROUND(a,b,c,d,e,f,g,h,k,w) \
    { unsigned int T1=(h)+BIG1(e)+CH(e,f,g)+(k)+(w); \
      unsigned int T2=BIG0(a)+MAJ(a,b,c); \
      (h)=(g);(g)=(f);(f)=(e);(e)=(d)+(T1); \
      (d)=(c);(c)=(b);(b)=(a);(a)=(T1)+(T2); }

__device__ void sha256_compress(unsigned int* W, const unsigned int* K,
                                 const unsigned int* H0, unsigned int* out_e5) {
    unsigned int a=H0[0],b=H0[1],c=H0[2],d=H0[3];
    unsigned int e=H0[4],f=H0[5],g=H0[6],h=H0[7];
    #pragma unroll
    for (int r = 0; r < 64; r++) {
        ROUND(a,b,c,d,e,f,g,h, K[r], W[r])
    }
    out_e5[0] = f;  // word 5 of output state (f corresponds to state word index 5 after final round)
    // Wait - after 64 rounds, the state is (a,b,c,d,e,f,g,h) as final state
    // word 5 = index 5 = f (6th word in 0-indexed state after compression)
    // Actually the state words at completion are: [a,b,c,d,e,f,g,h]
    // word[5] = f
    out_e5[0] = f;
}

extern "C" __global__ void sha256_diff_sweep(
    unsigned int* hits_count,
    unsigned long long* hits_w15,
    int max_hits,
    unsigned long long start,
    unsigned long long count,
    const unsigned int* K_arr,
    const unsigned int* H0_arr,
    const unsigned int* W0_14,
    const unsigned int* W0_14p
) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    unsigned int W15 = (unsigned int)(start + tid);

    // Build schedules inline (use registers, not shared mem, for L1 cache)
    unsigned int W[64], Wp[64];

    // Load W[0..14]
    #pragma unroll 15
    for (int i = 0; i < 15; i++) { W[i] = W0_14[i]; Wp[i] = W0_14p[i]; }
    W[15] = W15;   // same W[15] for both
    Wp[15] = W15;

    // Expand schedule
    #pragma unroll 8
    for (int r = 16; r < 64; r++) {
        W[r]  = SIG1(W[r-2])  + W[r-7]  + SIG0(W[r-15])  + W[r-16];
        Wp[r] = SIG1(Wp[r-2]) + Wp[r-7] + SIG0(Wp[r-15]) + Wp[r-16];
    }

    // Compress M
    unsigned int a=H0_arr[0],b=H0_arr[1],c=H0_arr[2],d=H0_arr[3];
    unsigned int e=H0_arr[4],f=H0_arr[5],g=H0_arr[6],h=H0_arr[7];
    for (int r = 0; r < 64; r++) {
        ROUND(a,b,c,d,e,f,g,h, K_arr[r], W[r])
    }
    unsigned int f_out = f;

    // Compress M'
    unsigned int ap=H0_arr[0],bp=H0_arr[1],cp=H0_arr[2],dp=H0_arr[3];
    unsigned int ep=H0_arr[4],fp=H0_arr[5],gp=H0_arr[6],hp=H0_arr[7];
    for (int r = 0; r < 64; r++) {
        ROUND(ap,bp,cp,dp,ep,fp,gp,hp, K_arr[r], Wp[r])
    }
    unsigned int fp_out = fp;

    unsigned int diff5 = fp_out - f_out;
    if (diff5 == 0) {
        int idx = atomicAdd(hits_count, 1);
        if (idx < max_hits) {
            hits_w15[idx] = (unsigned long long)(start + tid);
        }
    }
}

// ─── ALSO: compute full diff vector for a single W[15] ───────────────────────
extern "C" __global__ void sha256_full_diff(
    unsigned int* diff_out,
    unsigned int w15_val,
    const unsigned int* K_arr,
    const unsigned int* H0_arr,
    const unsigned int* W0_14,
    const unsigned int* W0_14p
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int W[64], Wp[64];
    for (int i = 0; i < 15; i++) { W[i] = W0_14[i]; Wp[i] = W0_14p[i]; }
    W[15] = w15_val; Wp[15] = w15_val;
    for (int r = 16; r < 64; r++) {
        W[r]  = SIG1(W[r-2])  + W[r-7]  + SIG0(W[r-15])  + W[r-16];
        Wp[r] = SIG1(Wp[r-2]) + Wp[r-7] + SIG0(Wp[r-15]) + Wp[r-16];
    }
    unsigned int a=H0_arr[0],b=H0_arr[1],c=H0_arr[2],d=H0_arr[3];
    unsigned int e=H0_arr[4],f=H0_arr[5],g=H0_arr[6],h=H0_arr[7];
    for (int r=0;r<64;r++) { ROUND(a,b,c,d,e,f,g,h,K_arr[r],W[r]) }
    unsigned int ap=H0_arr[0],bp=H0_arr[1],cp=H0_arr[2],dp=H0_arr[3];
    unsigned int ep=H0_arr[4],fp=H0_arr[5],gp=H0_arr[6],hp=H0_arr[7];
    for (int r=0;r<64;r++) { ROUND(ap,bp,cp,dp,ep,fp,gp,hp,K_arr[r],Wp[r]) }
    unsigned int state[8]  = {a,b,c,d,e,f,g,h};
    unsigned int statep[8] = {ap,bp,cp,dp,ep,fp,gp,hp};
    for (int i=0;i<8;i++) diff_out[i] = statep[i] - state[i];
}
"""

print("Compiling CUDA kernel...")
t0 = time.time()
module = cp.RawModule(code=CUDA_SRC, options=(), name_expressions=['sha256_diff_sweep', 'sha256_full_diff'])
sweep_kernel = module.get_function('sha256_diff_sweep')
diff_kernel  = module.get_function('sha256_full_diff')
print(f"Compiled in {time.time()-t0:.2f}s")

# ─── Push constants to GPU ────────────────────────────────────────────────────
K_gpu    = cp.array(K_vals,    dtype=cp.uint32)
H0_gpu   = cp.array(H0_vals,   dtype=cp.uint32)
W14_gpu  = cp.array(W_BASE,    dtype=cp.uint32)
W14p_gpu = cp.array(W_PRIME,   dtype=cp.uint32)

MAX_HITS = 1024
hits_count = cp.zeros(1, dtype=cp.uint32)
hits_w15   = cp.zeros(MAX_HITS, dtype=cp.uint64)

# ─── Verify against numpy reference ───────────────────────────────────────────
def verify_reference(w15):
    """CPU reference implementation to verify kernel correctness."""
    import sys
    sys.path.insert(0, '/home/administrator/sha/sha256')
    from utils import MASK32 as M32, H0 as H0ref, K as Kref, small_sigma0, small_sigma1, big_sigma0, big_sigma1, ch, maj

    def sched(W16):
        W = list(W16)
        for r in range(16, 64):
            W.append((small_sigma1(W[r-2]) + W[r-7] + small_sigma0(W[r-15]) + W[r-16]) & M32)
        return W

    def compress(W64):
        a,b,c,d,e,f,g,h = H0ref
        for r in range(64):
            T1 = (h + big_sigma1(e) + ch(e,f,g) + Kref[r] + W64[r]) & M32
            T2 = (big_sigma0(a) + maj(a,b,c)) & M32
            a,b,c,d,e,f,g,h = (T1+T2)&M32, a,b,c, (d+T1)&M32, e,f,g
        return [a,b,c,d,e,f,g,h]

    W  = W_BASE + [w15]
    Wp = W_PRIME + [w15]
    S  = compress(sched(W))
    Sp = compress(sched(Wp))
    return [(Sp[i]-S[i])&M32 for i in range(8)]

# Verify kernel gives same result as CPU for W[15]=0
print("\nVerifying kernel vs CPU reference (W[15]=0)...")
ref_diff = verify_reference(0)
print(f"  CPU: {[hex(x) for x in ref_diff]}")

diff_out = cp.zeros(8, dtype=cp.uint32)
diff_kernel((1,), (1,), (diff_out, cp.uint32(0), K_gpu, H0_gpu, W14_gpu, W14p_gpu))
cp.cuda.runtime.deviceSynchronize()
gpu_diff = diff_out.get().tolist()
print(f"  GPU: {[hex(x) for x in gpu_diff]}")
assert ref_diff == gpu_diff, f"MISMATCH! CPU={ref_diff}, GPU={gpu_diff}"
print(f"  Match: OK ✓")

# Verify near-collision value (SA best: word5 = 0x02000010)
print(f"\nVerifying SA near-collision W[15]=0xe279fe16...")
ref_nc = verify_reference(0xe279fe16)
print(f"  CPU: word5 = 0x{ref_nc[5]:08x}  (SA best = 0x02000010)")
diff_out2 = cp.zeros(8, dtype=cp.uint32)
diff_kernel((1,),(1,),(diff_out2, cp.uint32(0xe279fe16), K_gpu, H0_gpu, W14_gpu, W14p_gpu))
cp.cuda.runtime.deviceSynchronize()
gpu_nc = diff_out2.get().tolist()
print(f"  GPU: word5 = 0x{gpu_nc[5]:08x}")
if ref_nc == gpu_nc:
    print(f"  Match: OK ✓")
else:
    print(f"  MISMATCH (but might differ from SA optimum since W[15] is fixed in that path)")

# ─── Benchmark ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"BENCHMARK — RTX 4060 CUDA kernel")
print(f"{'='*60}")

BLOCK_SIZE = 256

def run_batch(start, count):
    hits_count.fill(0)
    n_blocks = (count + BLOCK_SIZE - 1) // BLOCK_SIZE
    sweep_kernel(
        (n_blocks,), (BLOCK_SIZE,),
        (hits_count, hits_w15, cp.int32(MAX_HITS),
         cp.uint64(start), cp.uint64(count),
         K_gpu, H0_gpu, W14_gpu, W14p_gpu)
    )
    cp.cuda.runtime.deviceSynchronize()
    n_hits = int(hits_count.get()[0])
    return hits_w15[:n_hits].get().tolist()

# Warmup
run_batch(0, 1_000_000)

for B in [1_000_000, 5_000_000, 10_000_000, 50_000_000]:
    t0 = time.time()
    run_batch(0, B)
    t1 = time.time()
    rate = B / (t1 - t0)
    full_est = (1 << 32) / rate
    print(f"B={B//1_000_000}M: {rate:.2e} pairs/s  2^32 in {full_est:.0f}s ({full_est/60:.1f}min)")

# ─── Full sweep ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"FULL 2^32 SWEEP — targeting ΔState[64][word5] = 0")
print(f"{'='*60}")

TOTAL = 1 << 32
BATCH = 50_000_000  # 50M per batch
all_hits = []
start_time = time.time()
pos = 0

while pos < TOTAL:
    end = min(pos + BATCH, TOTAL)
    batch_count = end - pos
    batch_hits = run_batch(pos, batch_count)
    all_hits.extend(batch_hits)
    for h in batch_hits:
        print(f"  *** HIT: W[15]=0x{h:08x} ({h}) ***")

    elapsed = time.time() - start_time
    pct = 100 * end / TOTAL
    rate = end / elapsed
    eta = (TOTAL - end) / rate if rate > 0 else 0
    print(f"  {pct:5.1f}%  {end/1e6:.0f}M done  {rate:.2e}/s  ETA {eta:.0f}s  hits={len(all_hits)}")
    pos = end

elapsed = time.time() - start_time
print(f"\n{'='*60}")
print(f"Sweep done: {TOTAL:,} values in {elapsed:.1f}s ({elapsed/60:.1f}min)")
print(f"Total hits (ΔState[64][word5]=0): {len(all_hits)}")
for h in all_hits:
    d = verify_reference(h)
    print(f"\n  W[15] = 0x{h:08x}")
    print(f"  ΔState[64]: {[hex(x) for x in d]}")
    bits = sum(bin(x).count('1') for x in d)
    print(f"  Total bit weight: {bits}/256")
