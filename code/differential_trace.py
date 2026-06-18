"""
Round-by-round state differential trace for block-2 near-collision.

For the best-known block-2 message (75/256), trace the internal state
difference Δstate[r] = state'[r] - state[r] through all 64 rounds.

Also:
- Identify which rounds amplify vs suppress the differential
- Find the "softest" rounds (lowest HW amplification)
- Analyse per-output-word contributions
- Check if any round achieves Δstate[r]=0 (a "local collision")

Second pass: same analysis for the W_B1 baseline (134/256) to compare.
"""
import os
os.environ['CUDA_PATH'] = '/usr'
import sys, numpy as np
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

def hw(x):
    return bin(x & M).count('1')

def compress_trace(ws, iv):
    """Returns list of (a,b,c,d,e,f,g,h) after each round (64 entries)."""
    a,b,c,d,e,f,g,h = iv
    trace = []
    for r in range(64):
        T1 = (h + S1(e) + ch(e,f,g) + K_cpu[r] + ws[r]) & M
        T2 = (S0(a) + maj(a,b,c)) & M
        a,b,c,d,e,f,g,h = (T1+T2)&M, a,b,c, (d+T1)&M, e,f,g
        trace.append((a,b,c,d,e,f,g,h))
    return trace

W_B1p = list(W_B1); W_B1p[0]=(W_B1p[0]+DELTA)&M; W_B1p[9]=(W_B1p[9]-DELTA)&M

def compress(ws, iv):
    a,b,c,d,e,f,g,h = iv
    for r in range(64):
        T1 = (h + S1(e) + ch(e,f,g) + K_cpu[r] + ws[r]) & M
        T2 = (S0(a) + maj(a,b,c)) & M
        a,b,c,d,e,f,g,h = (T1+T2)&M, a,b,c, (d+T1)&M, e,f,g
    return [a,b,c,d,e,f,g,h]

H1  = compress(sched(W_B1),  H0_cpu)
H1p = compress(sched(W_B1p), H0_cpu)

# Initial IV difference for block-2
delta_iv = [(H1p[i] - H1[i]) & M for i in range(8)]
print(f"ΔH1 (block-2 IV difference): {sum(hw(x) for x in delta_iv)}/256 bits", flush=True)
print(f"  Per word: {[hw(x) for x in delta_iv]}", flush=True)

def analyse_trace(N16, label):
    Np = list(N16); Np[0]=(Np[0]+DELTA)&M; Np[9]=(Np[9]-DELTA)&M
    ws  = sched(N16)
    wsp = sched(Np)

    # Schedule HW
    sched_hw = [hw((wsp[r]-ws[r])&M) for r in range(64)]
    print(f"\n{'='*70}", flush=True)
    print(f"TRACE: {label}", flush=True)
    print(f"  Schedule ΔW HW: {sched_hw[:16]}  (free words)", flush=True)
    print(f"  Schedule ΔW HW: {sched_hw[16:24]}  (dead zone)", flush=True)
    print(f"  Schedule ΔW HW: {sched_hw[24:32]}", flush=True)
    print(f"  Schedule ΔW HW: {sched_hw[32:40]}", flush=True)
    print(f"  Schedule ΔW HW: {sched_hw[40:48]}", flush=True)
    print(f"  Schedule ΔW HW: {sched_hw[48:56]}", flush=True)
    print(f"  Schedule ΔW HW: {sched_hw[56:64]}", flush=True)

    t1 = compress_trace(ws,  H1)
    t2 = compress_trace(wsp, H1p)

    # State difference after each round
    # State = (a,b,c,d,e,f,g,h); diff per word
    round_hws = []
    for r in range(64):
        diff = [(t2[r][i] - t1[r][i]) & M for i in range(8)]
        total = sum(hw(x) for x in diff)
        round_hws.append(total)

    print(f"\n  Round-by-round state diff HW (0=local collision):", flush=True)
    for r in range(0, 64, 8):
        row = round_hws[r:r+8]
        print(f"    r={r:2d}-{r+7:2d}: {row}", flush=True)

    # Min and max rounds
    min_r = int(np.argmin(round_hws))
    max_r = int(np.argmax(round_hws))
    print(f"\n  Min HW at round {min_r}: {round_hws[min_r]}/256", flush=True)
    print(f"  Max HW at round {max_r}: {round_hws[max_r]}/256", flush=True)

    # Final HW
    final = [(t2[63][i] - t1[63][i]) & M for i in range(8)]
    total = sum(hw(x) for x in final)
    print(f"  Final output HW: {total}/256  (= |ΔH2|)", flush=True)

    # Per output word
    print(f"\n  Per-output-word ΔH2:", flush=True)
    for i in range(8):
        v = final[i]
        print(f"    H2[{i}]: 0x{v:08x}  ({hw(v)} bits)", flush=True)

    # Transitions: where does HW increase the most? (amplification rounds)
    deltas = [round_hws[r+1] - round_hws[r] for r in range(63)]
    amp_r  = sorted(range(63), key=lambda r: -deltas[r])[:5]
    sup_r  = sorted(range(63), key=lambda r:  deltas[r])[:5]
    print(f"\n  Top-5 amplifying round transitions:", flush=True)
    for r in amp_r:
        print(f"    r={r}→{r+1}:  {round_hws[r]}→{round_hws[r+1]}  ({deltas[r]:+d})", flush=True)
    print(f"  Top-5 suppressing round transitions:", flush=True)
    for r in sup_r:
        print(f"    r={r}→{r+1}:  {round_hws[r]}→{round_hws[r+1]}  ({deltas[r]:+d})", flush=True)

    return round_hws

hw_best = analyse_trace(N_BEST, "best block-2  (coord descent, 75/256)")
hw_base = analyse_trace(W_B1,   "block-1 message  (baseline, 134/256)")

# Compare: which rounds benefit most from coord descent message vs baseline?
print(f"\n{'='*70}", flush=True)
print("COMPARISON: round-by-round HW reduction  (baseline - best)", flush=True)
print("Positive = coord descent message suppressed differential at that round", flush=True)
diffs_rr = [hw_base[r] - hw_best[r] for r in range(64)]
for r in range(0, 64, 8):
    row = diffs_rr[r:r+8]
    print(f"  r={r:2d}-{r+7:2d}: {row}", flush=True)

max_benefit_r = int(np.argmax(diffs_rr))
print(f"\n  Most suppressed round: {max_benefit_r}  (diff = {diffs_rr[max_benefit_r]:+d})", flush=True)
