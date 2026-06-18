"""
Exhaustive schedule differential analysis.

Part 1: For all C(16,2)=120 pairs (i<j) with ΔW[i]=+δ, ΔW[j]=-δ (and reversed),
        compute the exact schedule differential ΔW[16..63] for δ = DELTA.
        Also sweep δ over 2^32 to find best (min total schedule HW) per pair.

Part 2: For single-position differentials (ΔW[i]=+δ), compute schedule diff.

Part 3: Multi-point differentials — all 16 "shifted Nine-Step" variants
        (i, i+9 mod 16) and all adjacent pairs.

Part 4: GPU exhaustive — for each of the 120 pairs (i<j), sweep δ over
        all 2^32 values, recording the minimum total HW of ΔW[16..63].
        This finds the "best delta" for each word pair.

Goal: discover if any pair beyond (0,9) creates an extended algebraic dead zone.
"""
import os
os.environ['CUDA_PATH'] = '/usr'
import numpy as np, sys, time
sys.path.insert(0, '/home/administrator/sha/sha256')
from utils import MASK32 as M, small_sigma0 as s0, small_sigma1 as s1

DELTA = 0x0011e034

def schedule_diff(dw16, delta=None):
    """Compute ΔW[0..63] given initial 16 deltas. Returns list of 64 values."""
    dw = list(dw16)
    for r in range(16, 64):
        v = (s1(dw[r-2]) + dw[r-7] + s0(dw[r-15]) + dw[r-16]) & M
        dw.append(v)
    return dw

def hw(x):
    return bin(x & M).count('1')

def total_hw(dws, start=16, end=64):
    return sum(hw(dws[r]) for r in range(start, end))

DELTA = 0x0011e034

print("="*70, flush=True)
print("PART 1: All 240 pair differentials (i,j) with ΔW[i]=+δ, ΔW[j]=-δ", flush=True)
print(f"  δ = DELTA = 0x{DELTA:08x}", flush=True)
print("="*70, flush=True)

results = []
for i in range(16):
    for j in range(16):
        if i == j:
            continue
        dw = [0]*16
        dw[i] = DELTA
        dw[j] = (-DELTA) & M
        ds = schedule_diff(dw)
        dead16_23 = sum(1 for r in range(16,24) if ds[r]==0)
        dead16_63 = sum(1 for r in range(16,64) if ds[r]==0)
        tot_hw = sum(hw(ds[r]) for r in range(16,64))
        sched_hw_16_23 = sum(hw(ds[r]) for r in range(16,24))
        results.append((i, j, dead16_23, dead16_63, sched_hw_16_23, tot_hw, ds))

# Sort by total schedule HW (low = most interesting)
results.sort(key=lambda x: x[5])

print(f"\n{'Pair':>10}  {'dead16-23':>10}  {'dead16-63':>10}  {'HW[16-23]':>10}  {'HW[16-63]':>10}", flush=True)
for i,j,d8,d48,h8,h48,ds in results[:30]:
    marker = " *** NINE-STEP" if (i,j)==(0,9) else ""
    marker += " <-- PERFECT DEAD ZONE" if d8==8 else ""
    print(f"  ({i:2d},{j:2d})      {d8:>10}  {d48:>10}  {h8:>10}  {h48:>10}{marker}", flush=True)

# Find pairs with dead16_23 == 8 (full algebraic dead zone)
perfect = [(i,j,ds) for i,j,d8,d48,h8,h48,ds in results if d8==8]
print(f"\nPairs with ALL W[16..23]=0 (full dead zone): {len(perfect)}", flush=True)
for i,j,ds in perfect:
    print(f"  ({i},{j}): W[24..31] = {[hex(ds[r]) for r in range(24,32)]}", flush=True)

print(flush=True)
print("="*70, flush=True)
print("PART 2: Single-position differentials ΔW[i]=+δ (no cancellation)", flush=True)
print("="*70, flush=True)
singles = []
for i in range(16):
    dw = [0]*16; dw[i] = DELTA
    ds = schedule_diff(dw)
    h8 = sum(hw(ds[r]) for r in range(16,24))
    h48 = sum(hw(ds[r]) for r in range(16,64))
    n0 = sum(1 for r in range(16,64) if ds[r]==0)
    singles.append((i, h8, h48, n0))

singles.sort(key=lambda x: x[2])
print(f"  {'i':>3}  {'HW[16-23]':>10}  {'HW[16-63]':>10}  {'zeros':>6}", flush=True)
for i,h8,h48,n0 in singles:
    print(f"  {i:>3}  {h8:>10}  {h48:>10}  {n0:>6}", flush=True)

print(flush=True)
print("="*70, flush=True)
print("PART 3: Multi-point differentials — shifted Nine-Step variants", flush=True)
print("="*70, flush=True)

# All (k, k+9) pairs for k=0..6  (the "Nine-Step family")
print("\n  Nine-Step family (k, k+9):", flush=True)
for k in range(7):
    i, j = k, k+9
    dw = [0]*16; dw[i]=DELTA; dw[j]=(-DELTA)&M
    ds = schedule_diff(dw)
    d8 = sum(1 for r in range(16,24) if ds[r]==0)
    h8 = sum(hw(ds[r]) for r in range(16,24))
    h48 = sum(hw(ds[r]) for r in range(16,64))
    # Which W[16..23] are zero?
    zeros = [r-16 for r in range(16,24) if ds[r]==0]
    print(f"  ({i:2d},{j:2d})  dead_zone={zeros}  HW[16-23]={h8}  HW[16-63]={h48}", flush=True)

# Double Nine-Step: (0,9) + (1,10) simultaneously
print("\n  Double / triple Nine-Step combinations:", flush=True)
combos = [
    [(0,9),(1,10)],
    [(0,9),(2,11)],
    [(0,9),(1,10),(2,11)],
    [(0,9),(3,12)],
    [(0,9),(1,10),(3,12)],
]
for combo in combos:
    dw = [0]*16
    for (i,j) in combo:
        dw[i] = (dw[i] + DELTA) & M
        dw[j] = (dw[j] - DELTA) & M
    ds = schedule_diff(dw)
    d8 = sum(1 for r in range(16,24) if ds[r]==0)
    h8 = sum(hw(ds[r]) for r in range(16,24))
    h48 = sum(hw(ds[r]) for r in range(16,64))
    zeros_16_24 = [r for r in range(16,24) if ds[r]==0]
    label = '+'.join(f'({i},{j})' for i,j in combo)
    print(f"  {label:30s}  dead={zeros_16_24}  HW[16-23]={h8}  HW[16-63]={h48}", flush=True)

print(flush=True)
print("="*70, flush=True)
print("PART 4: For each pair (i<j), find δ that minimises HW[W[16..23]]", flush=True)
print("  (CPU exhaustive over δ limited to 2^20 for speed — GPU below)", flush=True)
print("="*70, flush=True)

SAMPLE_DELTA = 1 << 20  # 1M samples for CPU
best_per_pair = {}

for i in range(16):
    for j in range(i+1, 16):
        best_h8 = 999
        best_d  = 0
        for d in range(1, SAMPLE_DELTA):
            dw = [0]*16; dw[i]=d; dw[j]=(-d)&M
            ds = schedule_diff(dw)
            h8 = sum(hw(ds[r]) for r in range(16,24))
            if h8 < best_h8:
                best_h8 = h8
                best_d = d
                if h8 == 0:
                    break
        best_per_pair[(i,j)] = (best_h8, best_d)

# Sort by best_h8
sorted_pairs = sorted(best_per_pair.items(), key=lambda x: x[1][0])
print(f"\n  {'Pair':>10}  {'best HW[16-23]':>14}  {'best δ':>12}", flush=True)
for (i,j),(bh,bd) in sorted_pairs[:20]:
    marker = " *** ZERO" if bh==0 else ""
    print(f"  ({i:2d},{j:2d})       {bh:>14}  0x{bd:08x}{marker}", flush=True)

zero_pairs = [(p,v) for p,v in sorted_pairs if v[0]==0]
print(f"\n  Pairs achieving HW[16-23]=0 with some δ: {len(zero_pairs)}", flush=True)
for (i,j),(bh,bd) in zero_pairs[:20]:
    dw = [0]*16; dw[i]=bd; dw[j]=(-bd)&M
    ds = schedule_diff(dw)
    h48 = sum(hw(ds[r]) for r in range(16,64))
    print(f"  ({i:2d},{j:2d})  δ=0x{bd:08x}  HW[16-23]=0  HW[16-63]={h48}", flush=True)

print(flush=True)
print("="*70, flush=True)
print("PART 5: What happens at W[25]?  (Nine-Step reappearance at -δ)", flush=True)
print("="*70, flush=True)

# Verify known Nine-Step: W[25] = -δ
dw = [0]*16; dw[0]=DELTA; dw[9]=(-DELTA)&M
ds = schedule_diff(dw)
print(f"\n  (0,9) Nine-Step differential:", flush=True)
for r in range(16, 33):
    v = ds[r]
    zero = " ← ZERO" if v==0 else (f"  ← -δ = 0x{(-DELTA)&M:08x}" if v==(-DELTA)&M else f"  ({hw(v)} bits)")
    print(f"    W[{r:2d}] = 0x{v:08x}{zero}", flush=True)

# Check other pairs for similar W[25]=-δ reappearance
print(f"\n  Pairs where W[25]=-δ (like Nine-Step):", flush=True)
for i in range(16):
    for j in range(i+1, 16):
        dw = [0]*16; dw[i]=DELTA; dw[j]=(-DELTA)&M
        ds = schedule_diff(dw)
        if ds[25] == (-DELTA)&M:
            h8  = sum(hw(ds[r]) for r in range(16,24))
            print(f"    ({i},{j}): W[25]=-δ, HW[16-23]={h8}", flush=True)
        dw = [0]*16; dw[j]=DELTA; dw[i]=(-DELTA)&M
        ds = schedule_diff(dw)
        if ds[25] == (-DELTA)&M:
            h8  = sum(hw(ds[r]) for r in range(16,24))
            print(f"    ({j},{i}): W[25]=-δ, HW[16-23]={h8}", flush=True)
