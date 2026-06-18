"""
Deep search for minimum near-collision HW across:
1. Block-3 random starts (N_STARTS) — extends three-block chain
2. (13,4) differential two-block to extract optimal N16
3. (13,4) three-block with random starts
4. Remaining alt differentials: (7,8), (8,7), (12,3), (3,12)

Runs as a single process to avoid GPU contention.
"""
import os
os.environ['CUDA_PATH'] = '/usr'
import cupy as cp, numpy as np, sys, time
sys.path.insert(0, '/home/administrator/sha/sha256')
from utils import MASK32 as M, H0 as H0_cpu, K as K_cpu, small_sigma0 as s0
from utils import small_sigma1 as s1, big_sigma0 as S0, big_sigma1 as S1, ch, maj

DELTA = 0x0011e034
N_STARTS = 12       # random starts per block-3 experiment

W_B1 = [0x6ac550a3,0x604e4a61,0x3d096d9d,0xc0b47203,0xe3127bf5,0x59c5c56a,
         0x8dd4cf0d,0xede08902,0x1eae72a3,0x001c2641,0x195659ea,0x0e39df0d,
         0xb0a6feb0,0x9562b0a6,0xa56abdfe,0x82627b41]

# Best block-2 (nine-step) from coord descent
N_BEST_NINESTEP = [0x254b45f2, 0xae6bc5c0, 0x531ffc39,
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

def hw_total(a, b):
    return sum(bin((a[i]-b[i])&M).count('1') for i in range(8))

def ref_hw_b3(N16, H_iv, H_ivp, delta, pos_plus=0, pos_minus=9):
    Np = list(N16)
    Np[pos_plus]  = (Np[pos_plus]  + delta) & M
    Np[pos_minus] = (Np[pos_minus] - delta) & M
    H  = compress(sched(N16), H_iv)
    Hp = compress(sched(Np),  H_ivp)
    return sum(bin((Hp[i]-H[i])&M).count('1') for i in range(8))

def ref_hw_gen(N16, H_iv, H_ivp, delta, pos_plus, pos_minus):
    Np = list(N16)
    Np[pos_plus]  = (Np[pos_plus]  + delta) & M
    Np[pos_minus] = (Np[pos_minus] - delta) & M
    H  = compress(sched(N16), H_iv)
    Hp = compress(sched(Np),  H_ivp)
    return sum(bin((Hp[i]-H[i])&M).count('1') for i in range(8))

# ── CUDA kernels ──────────────────────────────────────────────────────────────
CUDA_SRC = r"""
#define ROTR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define SIG0(x) (ROTR32(x,7)^ROTR32(x,18)^((x)>>3))
#define SIG1(x) (ROTR32(x,17)^ROTR32(x,19)^((x)>>10))
#define BIG0(x) (ROTR32(x,2)^ROTR32(x,13)^ROTR32(x,22))
#define BIG1(x) (ROTR32(x,6)^ROTR32(x,11)^ROTR32(x,25))
#define CH(e,f,g)  (((e)&(f))^((~(e))&(g)))
#define MAJ(a,b,c) (((a)&(b))^((a)&(c))^((b)&(c)))

/* Generic sweep: differential at (pos_plus, pos_minus) */
extern "C" __global__ void sweep_gen(
    const unsigned int* base_words, int word_idx,
    unsigned int chunk_start, unsigned int chunk_size,
    unsigned int* out_hw, unsigned int* out_val,
    const unsigned int* K_arr, const unsigned int* H_arr, const unsigned int* Hp_arr,
    unsigned int delta, int pos_plus, int pos_minus
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= chunk_size) return;
    unsigned int val = chunk_start + tid;

    unsigned int W[16], Wp[16];
    for (int i = 0; i < 16; i++) { W[i] = base_words[i]; Wp[i] = base_words[i]; }
    W[word_idx] = val; Wp[word_idx] = val;
    Wp[pos_plus]  += delta;
    Wp[pos_minus] -= delta;

    unsigned int WW[64], WWp[64];
    for (int i = 0; i < 16; i++) { WW[i] = W[i]; WWp[i] = Wp[i]; }
    for (int r = 16; r < 64; r++) {
        WW[r]  = SIG1(WW[r-2])  + WW[r-7]  + SIG0(WW[r-15])  + WW[r-16];
        WWp[r] = SIG1(WWp[r-2]) + WWp[r-7] + SIG0(WWp[r-15]) + WWp[r-16];
    }

    unsigned int a=H_arr[0],b=H_arr[1],cc=H_arr[2],d=H_arr[3];
    unsigned int e=H_arr[4],f=H_arr[5],g=H_arr[6],h=H_arr[7];
    for (int r = 0; r < 64; r++) {
        unsigned int T1=h+BIG1(e)+CH(e,f,g)+K_arr[r]+WW[r];
        unsigned int T2=BIG0(a)+MAJ(a,b,cc);
        h=g;g=f;f=e;e=d+T1;d=cc;cc=b;b=a;a=T1+T2;
    }
    unsigned int ap=Hp_arr[0],bp=Hp_arr[1],ccp=Hp_arr[2],dp=Hp_arr[3];
    unsigned int ep=Hp_arr[4],fp=Hp_arr[5],gp=Hp_arr[6],hp=Hp_arr[7];
    for (int r = 0; r < 64; r++) {
        unsigned int T1=hp+BIG1(ep)+CH(ep,fp,gp)+K_arr[r]+WWp[r];
        unsigned int T2=BIG0(ap)+MAJ(ap,bp,ccp);
        hp=gp;gp=fp;fp=ep;ep=dp+T1;dp=ccp;ccp=bp;bp=ap;ap=T1+T2;
    }
    unsigned int tot = __popc(ap-a)+__popc(bp-b)+__popc(ccp-cc)+__popc(dp-d)
                     + __popc(ep-e)+__popc(fp-f)+__popc(gp-g)+__popc(hp-h);
    out_hw[tid] = tot; out_val[tid] = val;
}
"""

print("Compiling CUDA kernel...", flush=True)
mod    = cp.RawModule(code=CUDA_SRC, name_expressions=['sweep_gen'])
ker    = mod.get_function('sweep_gen')
K_gpu  = cp.array(K_cpu, dtype=cp.uint32)
BLOCK  = 256
CHUNK  = 1 << 22
N_CHUNKS = (1 << 32) // CHUNK

def sweep_one_word(cur, wi, H_iv, H_ivp, pos_plus, pos_minus):
    base_gpu = cp.array(cur, dtype=cp.uint32)
    H_gpu    = cp.array(H_iv,  dtype=cp.uint32)
    Hp_gpu   = cp.array(H_ivp, dtype=cp.uint32)
    best_hw, best_val = 256, 0
    for chunk in range(N_CHUNKS):
        cs = np.uint32(chunk * CHUNK)
        hw_gpu  = cp.empty(CHUNK, dtype=cp.uint32)
        val_gpu = cp.empty(CHUNK, dtype=cp.uint32)
        grid = (CHUNK + BLOCK - 1) // BLOCK
        ker((grid,), (BLOCK,), (
            base_gpu, cp.int32(wi), cs, np.uint32(CHUNK),
            hw_gpu, val_gpu, K_gpu, H_gpu, Hp_gpu,
            cp.uint32(DELTA), cp.int32(pos_plus), cp.int32(pos_minus)
        ))
        cp.cuda.runtime.deviceSynchronize()
        idx = int(cp.argmin(hw_gpu))
        bh, bv = int(hw_gpu[idx]), int(val_gpu[idx])
        if bh < best_hw:
            best_hw, best_val = bh, bv
    return best_hw, best_val

def coord_descent(start, H_iv, H_ivp, pos_plus, pos_minus, label, max_passes=3):
    cur = list(start)
    cur_hw = ref_hw_gen(cur, H_iv, H_ivp, DELTA, pos_plus, pos_minus)
    print(f"\n{label}  (start={cur_hw}/256)", flush=True)
    for p in range(max_passes):
        improved = False
        for wi in range(16):
            bh, bv = sweep_one_word(cur, wi, H_iv, H_ivp, pos_plus, pos_minus)
            if bh < cur_hw:
                print(f"  p={p} w[{wi:2d}]  {cur_hw}→{bh}  0x{bv:08x}", flush=True)
                cur[wi] = bv; cur_hw = bh; improved = True
        print(f"  Pass {p+1}: {cur_hw}/256", flush=True)
        if not improved:
            break
    return cur, cur_hw

# ── Block-1 nine-step IVs ──────────────────────────────────────────────────────
W_B1p = list(W_B1); W_B1p[0]=(W_B1p[0]+DELTA)&M; W_B1p[9]=(W_B1p[9]-DELTA)&M
H1_ns  = compress(sched(W_B1),  H0_cpu)
H1p_ns = compress(sched(W_B1p), H0_cpu)

# Block-2 nine-step outputs (from N_BEST)
N_BESTp = list(N_BEST_NINESTEP)
N_BESTp[0]=(N_BESTp[0]+DELTA)&M; N_BESTp[9]=(N_BESTp[9]-DELTA)&M
H2_ns  = compress(sched(N_BEST_NINESTEP), H1_ns)
H2p_ns = compress(sched(N_BESTp),          H1p_ns)
print(f"Nine-step H2/H2' |ΔH2|={hw_total(H2p_ns, H2_ns)}/256", flush=True)

# ── SECTION 1: Block-3 extended random starts (nine-step IV) ─────────────────
print(f"\n{'='*65}", flush=True)
print(f"SECTION 1: Block-3 extended random starts  (nine-step IV, {N_STARTS} starts)", flush=True)
print(f"  IV |ΔH2| = {hw_total(H2p_ns, H2_ns)}/256", flush=True)

rng = np.random.default_rng(1337)
best_global_b3 = 73  # known best from previous run
best_words_b3  = None

# Warm start: exact 73/256 state from threeblock_coord random start 3
# (start values reconstructed from seed=42, trial=2; w[0] and w[15] optimised)
known_best_b3 = [
    0xce9d0d07, 0x8df944c3, 0xe34cbfaf, 0x105653e3,
    0xdbb8fa6e, 0xd3dfa2f0, 0x46d9b92c, 0xa1b4c210,
    0x2a4c73f5, 0xc21209c3, 0xb355791f, 0x5ac236be,
    0x11633489, 0xf87faa6d, 0x7218933c, 0x5091c8d5,
]
cpu_check = ref_hw_b3(known_best_b3, H2_ns, H2p_ns, DELTA)
print(f"\n  Warm start (previous best 73/256) CPU verify: {cpu_check}/256", flush=True)
if cpu_check == 73:
    best_words_b3 = known_best_b3
    print(f"  Confirmed 73/256, using as warm start", flush=True)

for trial in range(N_STARTS):
    start = [int(x) & M for x in rng.integers(0, 2**32, size=16, dtype=np.uint64)]
    cw, ch = coord_descent(start, H2_ns, H2p_ns, 0, 9, f"=== B3-NS trial {trial+1}/{N_STARTS} ===")
    if ch < best_global_b3:
        best_global_b3 = ch
        best_words_b3  = cw
        print(f"  *** NEW GLOBAL BEST: {ch}/256 ***", flush=True)
    print(f"  Running best: {best_global_b3}/256", flush=True)

print(f"\nSection 1 best (block-3 nine-step): {best_global_b3}/256", flush=True)

# ── SECTION 2: (13,4) two-block coord descent — get optimal N16 and H2 ────────
print(f"\n{'='*65}", flush=True)
print(f"SECTION 2: (13,4) two-block coord descent — extract optimal N16", flush=True)

# (13,4): ΔW[13]=+δ, ΔW[4]=-δ
W_B1p_1304 = list(W_B1); W_B1p_1304[13]=(W_B1p_1304[13]+DELTA)&M; W_B1p_1304[4]=(W_B1p_1304[4]-DELTA)&M
H1_1304  = compress(sched(W_B1),       H0_cpu)
H1p_1304 = compress(sched(W_B1p_1304), H0_cpu)
dh1_1304 = hw_total(H1p_1304, H1_1304)
print(f"  (13,4) |ΔH1| = {dh1_1304}/256", flush=True)

N16_1304, dh2_1304 = coord_descent(W_B1, H1_1304, H1p_1304, 13, 4,
                                    "Block-2 (13,4) coord", max_passes=3)

print(f"\n  (13,4) best block-2: {dh2_1304}/256", flush=True)
print(f"  Optimal N16 (13,4):", flush=True)
for i, w in enumerate(N16_1304):
    mark = " ← CHANGED" if w != W_B1[i] else ""
    print(f"    N[{i:2d}] = 0x{w:08x}{mark}", flush=True)

# Compute H2/H2' for (13,4)
N16p_1304 = list(N16_1304)
N16p_1304[13]=(N16p_1304[13]+DELTA)&M; N16p_1304[4]=(N16p_1304[4]-DELTA)&M
H2_1304  = compress(sched(N16_1304),  H1_1304)
H2p_1304 = compress(sched(N16p_1304), H1p_1304)
print(f"  (13,4) H2/H2' |ΔH2| = {hw_total(H2p_1304, H2_1304)}/256", flush=True)

# Verify dh2_1304 CPU
cpu_v = ref_hw_gen(N16_1304, H1_1304, H1p_1304, DELTA, 13, 4)
print(f"  CPU verify: {cpu_v}/256", flush=True)

# ── SECTION 3: Block-3 extended random starts ((13,4) IV) ────────────────────
print(f"\n{'='*65}", flush=True)
print(f"SECTION 3: Block-3 random starts  ((13,4) IV, {N_STARTS} starts)", flush=True)
print(f"  IV |ΔH2| = {hw_total(H2p_1304, H2_1304)}/256", flush=True)

best_b3_1304 = 999
best_words_b3_1304 = None

# Also try W_B1 as starting point
cw0, ch0 = coord_descent(W_B1, H2_1304, H2p_1304, 13, 4, "=== B3-1304 from W_B1 ===", max_passes=3)
if ch0 < best_b3_1304:
    best_b3_1304 = ch0; best_words_b3_1304 = cw0
    print(f"  W_B1 start: {ch0}/256", flush=True)

for trial in range(N_STARTS):
    start = [int(x) & M for x in rng.integers(0, 2**32, size=16, dtype=np.uint64)]
    cw, ch = coord_descent(start, H2_1304, H2p_1304, 13, 4, f"=== B3-1304 trial {trial+1}/{N_STARTS} ===")
    if ch < best_b3_1304:
        best_b3_1304 = ch; best_words_b3_1304 = cw
        print(f"  *** NEW BEST (13,4) B3: {ch}/256 ***", flush=True)
    print(f"  Running best: {best_b3_1304}/256", flush=True)

print(f"\nSection 3 best (block-3 (13,4)): {best_b3_1304}/256", flush=True)

# ── SECTION 4: Remaining alt differentials ───────────────────────────────────
print(f"\n{'='*65}", flush=True)
print("SECTION 4: Remaining alt differentials  (4,13), (7,8), (8,7), (12,3), (3,12)", flush=True)

remaining = [
    (4, 13, "Pair (4,13) reversed",   "δ@W[4],-δ@W[13]"),
    (7,  8, "Pair (7,8) 6 dead",      "δ@W[7],-δ@W[8]"),
    (8,  7, "Pair (8,7) reversed",    "δ@W[8],-δ@W[7]"),
    (12, 3, "Pair (12,3) 5 dead",     "δ@W[12],-δ@W[3]"),
    (3, 12, "Pair (3,12) reversed",   "δ@W[3],-δ@W[12]"),
]

alt_results = []
for (pp, pm, label, desc) in remaining:
    print(f"\n{'='*65}", flush=True)
    print(f"  {label}  ({desc})", flush=True)
    Wp = list(W_B1); Wp[pp]=(Wp[pp]+DELTA)&M; Wp[pm]=(Wp[pm]-DELTA)&M
    H1  = compress(sched(W_B1), H0_cpu)
    H1p = compress(sched(Wp),   H0_cpu)
    dh1 = hw_total(H1p, H1)
    print(f"  |ΔH1| = {dh1}/256", flush=True)
    N16, dh2 = coord_descent(W_B1, H1, H1p, pp, pm, f"Block-2 ({pp},{pm}) coord", max_passes=3)
    alt_results.append((label, dh1, dh2, pp, pm, N16))

print(f"\n{'='*65}", flush=True)
print("SECTION 4 SUMMARY", flush=True)
print(f"  {'Differential':30}  |ΔH1|  Block-2", flush=True)
for label, dh1, dh2, pp, pm, _ in alt_results:
    print(f"  {label:30}  {dh1:3d}    {dh2:3d}/256", flush=True)

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}", flush=True)
print("FINAL SUMMARY — All experiments", flush=True)
print(f"  Nine-step two-block:   75/256  (coord descent from W_B1)", flush=True)
print(f"  (13,4)  two-block:     {dh2_1304}/256  (coord descent from W_B1)", flush=True)
print(f"  Three-block nine-step: {best_global_b3}/256  (best of {N_STARTS+1} starts)", flush=True)
print(f"  Three-block (13,4):    {best_b3_1304}/256  (best of {N_STARTS+1} starts)", flush=True)
for label, dh1, dh2, pp, pm, _ in alt_results:
    print(f"  {label[:20]:20}:   {dh2:3d}/256", flush=True)

# Verify the best three-block results
print(f"\n  CPU verifications:", flush=True)
if best_words_b3 and best_global_b3 <= 73:
    v = ref_hw_b3(best_words_b3, H2_ns, H2p_ns, DELTA)
    print(f"  Block-3 nine-step best: {v}/256 (CPU verify)", flush=True)
    print(f"  Words:", flush=True)
    for i, w in enumerate(best_words_b3):
        mark = " ← CHANGED" if w != W_B1[i] else ""
        print(f"    N2[{i:2d}] = 0x{w:08x}{mark}", flush=True)

if best_words_b3_1304 and best_b3_1304 <= 74:
    v = ref_hw_gen(best_words_b3_1304, H2_1304, H2p_1304, DELTA, 13, 4)
    print(f"  Block-3 (13,4) best:    {v}/256 (CPU verify)", flush=True)
    print(f"  Words:", flush=True)
    for i, w in enumerate(best_words_b3_1304):
        mark = " ← CHANGED" if w != W_B1[i] else ""
        print(f"    N2[{i:2d}] = 0x{w:08x}{mark}", flush=True)
