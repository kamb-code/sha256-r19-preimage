"""
GPU-accelerated simulated annealing for multi-word partial collision.
Evaluates batches of SA neighbor candidates on RTX 4060 via CUDA.
Target: zero multiple words of ΔState[64] under the Nine-Step differential.
"""
import os
os.environ['CUDA_PATH'] = '/usr'

import cupy as cp
import numpy as np
import random, time, sys

DELTA = np.uint64(0x0011e034)
M32 = np.uint64(0xFFFFFFFF)

K_vals = [0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]
H0_vals = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]

# W[0..9] fixed; W[10..15] are our 6 SA free variables
W_FIXED_0_9 = [0x6ac550a3,0x604e4a61,0x3d096d9d,0xc0b47203,0xe3127bf5,0x59c5c56a,0x8dd4cf0d,0xede08902,0x1eae72a3,0x001c2641]
# Best SA starting point (word5=0, total=95 bits):
W_START = [0x195659ea,0x0e39df0d,0xb0a6feb0,0x9562b0a6,0xa56abdfe,0x82627b41]

CUDA_SRC = r"""
#define M 0xFFFFFFFFu
#define ROTR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define SIG0(x) (ROTR32(x,7)^ROTR32(x,18)^((x)>>3))
#define SIG1(x) (ROTR32(x,17)^ROTR32(x,19)^((x)>>10))
#define BIG0(x) (ROTR32(x,2)^ROTR32(x,13)^ROTR32(x,22))
#define BIG1(x) (ROTR32(x,6)^ROTR32(x,11)^ROTR32(x,25))
#define CH(e,f,g) (((e)&(f))^(~(e)&(g)))
#define MAJ(a,b,c) (((a)&(b))^((a)&(c))^((b)&(c)))
#define ROUND(a,b,c,d,e,f,g,h,k,w) { \
    unsigned int T1=(h)+BIG1(e)+CH(e,f,g)+(k)+(w); \
    unsigned int T2=BIG0(a)+MAJ(a,b,c); \
    (h)=(g);(g)=(f);(f)=(e);(e)=(d)+(T1); \
    (d)=(c);(c)=(b);(b)=(a);(a)=(T1)+(T2); }

// popcount for 32-bit
__device__ int pc32(unsigned int x) { return __popc(x); }

// Evaluate batch of (W[10..15]) candidates; return popcount of each diff word
// Each thread = one candidate
extern "C" __global__ void eval_batch(
    const unsigned int* cands,   // shape [B, 6] — W[10..15] for each candidate
    unsigned int* diff_out,      // shape [B, 8] — ΔState[64] for each candidate
    unsigned int* cost_out,      // shape [B] — weighted cost
    int B,
    const unsigned int* W_fixed, // shape [10] — W[0..9]
    const unsigned int* K,       // shape [64]
    const unsigned int* H0,      // shape [8]
    unsigned int delta,
    int n_target,                // number of target words
    const int* target_words      // indices of words to target (primary)
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B) return;

    const unsigned int* row = cands + tid * 6;

    // Build W[0..15]
    unsigned int W[16], Wp[16];
    for (int i = 0; i < 10; i++) { W[i] = W_fixed[i]; Wp[i] = W_fixed[i]; }
    for (int i = 0; i < 6;  i++) { W[i+10] = row[i]; Wp[i+10] = row[i]; }
    // Apply differential: W'[0]+=delta, W'[9]-=delta
    Wp[0] += delta;
    Wp[9] -= delta;

    // Expand schedule
    unsigned int WW[64], WWp[64];
    for (int i = 0; i < 16; i++) { WW[i] = W[i]; WWp[i] = Wp[i]; }
    for (int r = 16; r < 64; r++) {
        WW[r]  = SIG1(WW[r-2])  + WW[r-7]  + SIG0(WW[r-15])  + WW[r-16];
        WWp[r] = SIG1(WWp[r-2]) + WWp[r-7] + SIG0(WWp[r-15]) + WWp[r-16];
    }

    // Compress M
    unsigned int a=H0[0],b=H0[1],cc=H0[2],d=H0[3];
    unsigned int e=H0[4],f=H0[5],g=H0[6],h=H0[7];
    for (int r=0;r<64;r++) { ROUND(a,b,cc,d,e,f,g,h,K[r],WW[r]) }

    // Compress M'
    unsigned int ap=H0[0],bp=H0[1],ccp=H0[2],dp=H0[3];
    unsigned int ep=H0[4],fp=H0[5],gp=H0[6],hp=H0[7];
    for (int r=0;r<64;r++) { ROUND(ap,bp,ccp,dp,ep,fp,gp,hp,K[r],WWp[r]) }

    // Compute diff
    unsigned int S[8]  = {a,b,cc,d,e,f,g,h};
    unsigned int Sp[8] = {ap,bp,ccp,dp,ep,fp,gp,hp};
    unsigned int diff[8];
    for (int i=0;i<8;i++) diff[i] = Sp[i] - S[i];
    for (int i=0;i<8;i++) diff_out[tid*8+i] = diff[i];

    // Compute cost: target words × 32 + other words × 1
    unsigned int cost = 0;
    bool in_target[8] = {false,false,false,false,false,false,false,false};
    for (int j=0;j<n_target;j++) in_target[target_words[j]] = true;
    for (int i=0;i<8;i++) {
        int w = in_target[i] ? 32 : 1;
        cost += w * pc32(diff[i]);
    }
    cost_out[tid] = cost;
}
"""

print("Compiling GPU SA kernel...")
t0 = time.time()
mod = cp.RawModule(code=CUDA_SRC, name_expressions=['eval_batch'])
eval_kernel = mod.get_function('eval_batch')
print(f"Compiled in {time.time()-t0:.2f}s")

K_gpu     = cp.array(K_vals,         dtype=cp.uint32)
H0_gpu    = cp.array(H0_vals,        dtype=cp.uint32)
Wfixed_gpu = cp.array(W_FIXED_0_9,   dtype=cp.uint32)

BLOCK = 256

def batch_eval(cands_np, target_words=(4,5)):
    """Evaluate a batch of W[10..15] candidates. Returns (diff_matrix, cost_vec)."""
    B = len(cands_np)
    cands_gpu = cp.array(cands_np.astype(np.uint32).reshape(-1), dtype=cp.uint32)
    diff_gpu  = cp.zeros(B * 8, dtype=cp.uint32)
    cost_gpu  = cp.zeros(B,     dtype=cp.uint32)
    tw_np     = np.array(target_words, dtype=np.int32)
    tw_gpu    = cp.array(tw_np, dtype=cp.int32)

    n_blocks = (B + BLOCK - 1) // BLOCK
    eval_kernel(
        (n_blocks,), (BLOCK,),
        (cands_gpu, diff_gpu, cost_gpu,
         cp.int32(B), Wfixed_gpu, K_gpu, H0_gpu,
         cp.uint32(0x0011e034),
         cp.int32(len(target_words)), tw_gpu)
    )
    cp.cuda.runtime.deviceSynchronize()
    diff_np = diff_gpu.get().reshape(B, 8).astype(np.uint32)
    cost_np = cost_gpu.get().astype(np.int32)
    return diff_np, cost_np

# ── Verify ──────────────────────────────────────────────────────────────────
print("\nVerifying GPU SA kernel vs reference...")
test_cands = np.array([W_START], dtype=np.uint32)
diff_m, cost_m = batch_eval(test_cands, target_words=(4,5))
print(f"  diff[4]=0x{diff_m[0,4]:08x} diff[5]=0x{diff_m[0,5]:08x}  cost={cost_m[0]}")
assert diff_m[0,5] == 0, "word5 should be 0 for starting config"
print(f"  Verification OK ✓")

# ── Benchmark batch evaluation ───────────────────────────────────────────────
print(f"\nBenchmark batch evaluation:")
for B in [10_000, 50_000, 200_000]:
    rng_cands = np.tile(W_START, (B, 1)).astype(np.int64)
    rng_cands += np.random.randint(0, 0x1000, (B, 6), dtype=np.int64)
    rng_cands = (rng_cands % (1<<32)).astype(np.uint32)
    t0 = time.time()
    _, _ = batch_eval(rng_cands)
    t1 = time.time()
    rate = B / (t1 - t0)
    print(f"  B={B}: {t1-t0:.3f}s = {rate:.2e} evals/s")

# ── Parallel SA: each iteration evaluates N_NEIGHBORS candidates ─────────────
print(f"\n{'='*60}")
print("GPU-ACCELERATED SA — targeting word4=word5=0")
print(f"{'='*60}")

N_NEIGHBORS = 50_000  # candidates per SA step
TARGET_WORDS = (4, 5)
MAX_SECONDS = 120

random.seed(0)
np.random.seed(0)

W_cur = np.array(W_START, dtype=np.uint32)
diff_cur, cost_arr = batch_eval(np.array([W_cur]))
cost_cur = int(cost_arr[0])

W_best = W_cur.copy()
diff_best = diff_cur[0].copy()
cost_best = cost_cur

T = 512.0
T_min = 0.1
t_start = time.time()
step = 0
last_print = t_start
improvements = 0

print(f"Initial state: word4=0x{diff_cur[0,4]:08x} word5=0x{diff_cur[0,5]:08x} cost={cost_cur}")
print(f"Running SA with {N_NEIGHBORS:,} parallel neighbors per step for {MAX_SECONDS}s...")

while T > T_min and time.time() - t_start < MAX_SECONDS:
    # Generate N_NEIGHBORS perturbations of W_cur
    # Use varied perturbation magnitudes
    neighbor_cands = np.tile(W_cur, (N_NEIGHBORS, 1)).astype(np.int64)
    magnitudes = np.concatenate([
        np.random.randint(-0xFFFF, 0x10000,     (N_NEIGHBORS//4, 6)),
        np.random.randint(-0x100,  0x100,        (N_NEIGHBORS//4, 6)),
        np.random.randint(-1, 2,                 (N_NEIGHBORS//4, 6)),  # ±1 or 0
        np.random.randint(-(1<<31), (1<<31),     (N_NEIGHBORS//4, 6)),  # full range
    ])
    # Each candidate changes 1, 2, or all 6 words
    n_change = np.random.choice([1,1,1,2,3,6], N_NEIGHBORS)
    for i in range(N_NEIGHBORS):
        idxs = np.random.choice(6, n_change[i], replace=False)
        for idx in idxs:
            neighbor_cands[i, idx] += magnitudes[i, idx]
    neighbor_cands = (neighbor_cands % (1 << 32)).astype(np.uint32)

    diff_n, cost_n = batch_eval(neighbor_cands, TARGET_WORDS)

    # Pick best neighbor
    best_idx = int(np.argmin(cost_n))
    cost_new = int(cost_n[best_idx])
    diff_new = diff_n[best_idx]

    # SA acceptance
    dE = cost_new - cost_cur
    if dE < 0 or random.random() < 2**(-dE / max(T, 0.001)):
        W_cur = neighbor_cands[best_idx].copy()
        diff_cur = diff_new[np.newaxis, :]
        cost_cur = cost_new
        if cost_cur < cost_best:
            cost_best = cost_cur
            W_best = W_cur.copy()
            diff_best = diff_new.copy()
            improvements += 1

    T *= 0.999  # cool per step (not per candidate)
    step += 1

    now = time.time()
    if now - last_print >= 10:
        bits = sum(bin(int(x)).count('1') for x in diff_best)
        bits4 = bin(int(diff_best[4])).count('1')
        bits5 = bin(int(diff_best[5])).count('1')
        print(f"  t={now-t_start:.0f}s T={T:.2f} step={step} improvements={improvements} "
              f"word4=0x{diff_best[4]:08x}({bits4}b) word5=0x{diff_best[5]:08x}({bits5}b) "
              f"total={bits} cost={cost_best}")
        last_print = now

    # Check for 2-word zero
    if diff_best[4] == 0 and diff_best[5] == 0:
        print(f"\n*** TWO-WORD PARTIAL COLLISION FOUND at step {step}! ***")
        break
    # Check for 1-additional-word zero (word4=0 added)
    if diff_best[4] == 0 and improvements > 0:
        print(f"\n+++ word4 also zero at step {step}! (word5 may have changed) +++")

elapsed = time.time() - t_start
print(f"\nSA complete: {step} steps ({step*N_NEIGHBORS:,} total evals) in {elapsed:.1f}s")
print(f"Rate: {step*N_NEIGHBORS/elapsed:.2e} evals/s")

# ── Final result ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("BEST RESULT")
print(f"{'='*60}")
diff_final, _ = batch_eval(np.array([W_best]))
diff_final = diff_final[0]
bits_final = sum(bin(int(x)).count('1') for x in diff_final)
zero_words = [i for i in range(8) if diff_final[i] == 0]
print(f"W[10..15] = {[hex(int(x)) for x in W_best]}")
print(f"ΔState[64] = {[hex(int(x)) for x in diff_final]}")
print(f"Zero words: {zero_words}")
print(f"Total Hamming weight: {bits_final}/256")

# Per-word bit counts
for i in range(8):
    b = bin(int(diff_final[i])).count('1')
    mark = " ← TARGET" if i in TARGET_WORDS else ""
    print(f"  word[{i}]: 0x{diff_final[i]:08x}  {b:2d} bits{mark}")

# Build full message pair if successful or best partial
W_full = W_FIXED_0_9 + [int(x) for x in W_best]
Wp_full = list(W_full)
Wp_full[0] = (Wp_full[0] + 0x0011e034) & 0xFFFFFFFF
Wp_full[9] = (Wp_full[9] - 0x0011e034) & 0xFFFFFFFF

print(f"\nFull message M[0..15]:  {[hex(w) for w in W_full]}")
print(f"Full message M'[0..15]: {[hex(w) for w in Wp_full]}")
print(f"ΔW[0] = +0x{0x0011e034:x}, ΔW[9] = -0x{0x0011e034:x} (Nine-Step differential)")
