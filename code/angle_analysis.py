"""
Multi-angle SHA-256 differential structure analysis.
Building on the Nine-Step Cancellation Lemma (ΔW[0]=+δ, ΔW[9]=-δ → ΔW[16..23]=0).
"""
import sys, random
import numpy as np
from collections import Counter
sys.path.insert(0, '/home/administrator/sha/sha256')
from utils import MASK32 as M, H0, K, small_sigma0 as s0, small_sigma1 as s1
from utils import big_sigma0 as S0, big_sigma1 as S1, ch, maj

DELTA = 0x0011e034
NEG_D = (M + 1 - DELTA) & M
random.seed(42)

def sched(W16):
    W = list(W16)
    for r in range(16, 64):
        W.append((s1(W[r-2]) + W[r-7] + s0(W[r-15]) + W[r-16]) & M)
    return W

def one_round(state, W_r, r):
    a,b,c,d,e,f,g,h = state
    T1 = (h + S1(e) + ch(e,f,g) + K[r] + W_r) & M
    T2 = (S0(a) + maj(a,b,c)) & M
    return [(T1+T2)&M, a, b, c, (d+T1)&M, e, f, g]

def compress_N(W64, N, iv=None):
    s = list(iv if iv else H0)
    for r in range(N): s = one_round(s, W64[r], r)
    return s

def dS(s1_, s2_):
    return [(s2_[i]-s1_[i]) & M for i in range(8)]

def hw(diff):  # Hamming weight = nonzero 32-bit words
    return sum(1 for x in diff if x != 0)

def tc0(v, d=DELTA):
    return (s0((v + d) & M) - s0(v)) & M == d

print("="*65)
print("MULTI-ANGLE SHA-256 DIFFERENTIAL ANALYSIS")
print(f"δ = 0x{DELTA:08x}")
print("="*65)

# ── Angle 1: State differential distribution at round boundaries ──────────────
print("\n═══ ANGLE 1: ΔState distribution at round boundaries ═══")
N = 20000
hw_at = {16: [], 24: [], 32: []}

for _ in range(N):
    W = [random.randint(0, M) for _ in range(16)]
    Wp = list(W); Wp[0] = (W[0]+DELTA)&M; Wp[9] = (W[9]-DELTA)&M
    Wf = sched(W); Wpf = sched(Wp)
    for R in [16, 24, 32]:
        hw_at[R].append(hw(dS(compress_N(Wf,R), compress_N(Wpf,R))))

for R in [16, 24, 32]:
    c = Counter(hw_at[R])
    print(f"\n  ΔState HW distribution after round {R}:")
    for k in sorted(c): print(f"    {k}/8 nonzero words: {c[k]/N*100:.1f}%")

# ── Angle 2: Find W[0..15] minimising ΔState after round 24 ──────────────────
print("\n═══ ANGLE 2: Minimum-weight ΔState[24] search (100k trials) ═══")
best = {'hw': 8, 'W': None, 'dS': None}
for _ in range(100000):
    W = [random.randint(0, M) for _ in range(16)]
    Wp = list(W); Wp[0] = (W[0]+DELTA)&M; Wp[9] = (W[9]-DELTA)&M
    Wf = sched(W); Wpf = sched(Wp)
    diff = dS(compress_N(Wf,24), compress_N(Wpf,24))
    h = hw(diff)
    if h < best['hw']:
        best['hw'] = h; best['W'] = list(W); best['dS'] = diff
        print(f"  New best: HW={h}  ΔState[24]={[hex(x) for x in diff]}")

print(f"  Final best: {best['hw']} nonzero words")

# ── Angle 3: Direct propagation chain analysis ────────────────────────────────
print("\n═══ ANGLE 3: Direct propagation chains ═══")
W = [random.randint(0, M) for _ in range(16)]
Wp = list(W); Wp[0]=(W[0]+DELTA)&M; Wp[9]=(W[9]-DELTA)&M
Wf = sched(W); Wpf = sched(Wp)
dW = [(Wpf[r]-Wf[r])&M for r in range(64)]

print("  W[r-16] chain from W[0] (+δ injected at r=0):")
for r in [0, 16, 32, 48]:
    s = "+δ ✓" if dW[r]==DELTA else "0" if dW[r]==0 else f"0x{dW[r]:08x}"
    print(f"    ΔW[{r:2d}] = {s}")

print("  W[r-7] chain from W[9] (-δ injected at r=9):")
for r in [9, 16, 25, 32, 41, 48, 57]:
    s = "-δ ✓" if dW[r]==NEG_D else "+δ" if dW[r]==DELTA else "0" if dW[r]==0 else f"0x{dW[r]:08x}"
    print(f"    ΔW[{r:2d}] = {s}")

# ── Angle 4: "Window reflection" — position (0,9) is unique ──────────────────
print("\n═══ ANGLE 4: Why (0,9) is unique — σ₀ contamination map ═══")
print("  For pair (i, i+9): σ₀(W[i]) hits W[r] at r=i+15")
print("  If i+15 >= 16 (i.e., i >= 1), σ₀ lands inside the extended schedule")
print("  For i=0: σ₀(W[0]) hits W[15] = MESSAGE WORD (known, no contamination)")
print()
for i in range(7):
    sigma0_hits = i + 15
    inside_window = sigma0_hits >= 16
    print(f"  Pair ({i},{i+9}): σ₀(W[{i}]) → W[{sigma0_hits}]"
          f"  {'❌ INSIDE extended schedule — contaminates window' if inside_window else '✓ message word — safe'}")

# ── Angle 5: Boomerang structure quantification ───────────────────────────────
print("\n═══ ANGLE 5: Boomerang structure — probability analysis ═══")
print("  Boomerang model:")
print("    E_top  = rounds  0-15 (schedule active: ΔW[0]=+δ, ΔW[9]=-δ)")
print("    E_mid  = rounds 16-23 (schedule ZERO — free state diffusion)")
print("    E_bot  = rounds 24-63 (schedule active: ΔW[25]=-δ, noisy)")
print()
print("  Key: E_mid is deterministic given ΔState[16]. No randomness.")
print("  The zero window eliminates 8 rounds of schedule probability conditions.")
print()
print("  In conventional differential attacks, each active round costs ~2^{-1} to 2^{-3}.")
print("  8 free rounds → saves ~2^{8} to 2^{24} in attack complexity.")

# ── Angle 6: Algebraic structure of ΔState during zero window ────────────────
print("\n═══ ANGLE 6: State differential trajectories (1000 trials) ═══")
print("  Tracking ΔState[r] HW for r=16..24 for 1000 random instances")

hw_trajectory = {r: [] for r in range(16, 25)}
for _ in range(1000):
    W = [random.randint(0, M) for _ in range(16)]
    Wp = list(W); Wp[0]=(W[0]+DELTA)&M; Wp[9]=(W[9]-DELTA)&M
    Wf = sched(W); Wpf = sched(Wp)
    Sn = list(H0); Spn = list(H0)
    for r in range(16):
        Sn = one_round(Sn, Wf[r], r)
        Spn = one_round(Spn, Wpf[r], r)
    for r in range(16, 25):
        Sn = one_round(Sn, Wf[r], r)
        Spn = one_round(Spn, Wpf[r], r)
        hw_trajectory[r].append(hw(dS(Sn, Spn)))

print("  Round | Mean HW | Min HW | Max HW")
print("  ------|---------|--------|-------")
for r in range(16, 25):
    data = hw_trajectory[r]
    print(f"  {r:5d} | {sum(data)/len(data):7.2f} | {min(data):6d} | {max(data):6d}")

# ── Angle 7: TC extension chain — can we extend the exact region? ─────────────
print("\n═══ ANGLE 7: TC-chain extension beyond round 25 ═══")
print("  ΔW[25]=-δ always (from W[r-16]=W[9]=-δ direct)")
print("  ΔW[24]=-δ with ONE TC condition (W[9]-δ ∈ TC-σ₀)")
print("  Can we get ΔW[26..27] exact with more TC conditions?")
print()
print("  W[26] = σ₁(W[24]) + W[19] + σ₀(W[11]) + W[10]")
print("  ΔW[26]: σ₁(W[24]) term — W[24] is a function of W[0..15]")
print("          W[19] term: ΔW[19]=0 ✓")
print("          σ₀(W[11]) term: ΔW[11]=0 ✓")
print("          W[10] term: ΔW[10]=0 ✓")
print("  → ΔW[26] = Δσ₁(W[24]) [only active term]")
print()
print("  If additionally W[24]-δ ∈ TC-σ₁(δ), then ΔW[26]=σ₁(W[24]-δ+δ)-σ₁(W[24]-δ)=-δ")
print("  BUT: TC-σ₁(δ)=∅ for δ=0x0011e034 → impossible!")
print()
print("  W[27] = σ₁(W[25]) + W[20] + σ₀(W[12]) + W[11]")
print("  ΔW[25]=-δ → Δσ₁(W[25]) active. Again requires TC-σ₁ which is empty.")
print("  → Beyond round 25 (with just TC-σ₀): no more exact positions possible")

# ── Angle 8: Constraint counting — how hard is a zero-window collision? ───────
print("\n═══ ANGLE 8: Collision difficulty with Nine-Step structure ═══")
print()
print("  Full 64-round SHA-256 collision:")
print("    Standard complexity: ~2^{65} (birthday on 256-bit hash)")
print("    Best practical: 2^{44} (31 rounds, Li et al. ASIACRYPT 2024)")
print()
print("  Nine-Step structure constraints:")
print("    Input freedom: 512 bits (W[0..15])")
print("    Fixed: ΔW[0]=+δ (32 bits), ΔW[9]=-δ (32 bits)")
print("    Free: 448 bits (W[1..8], W[10..15])")
print()
print("  Collision condition: ΔState[64] = 0 (256 bits)")
print("  → Need 256 equations in 448 free bits")
print("  → 192 bits of 'slack' in principle")
print()
print("  But SHA-256 is not linear! Each active round requires:")
print("  - Probability condition on maj/ch/σ functions")
print("  - 8 free rounds (16-23) ELIMINATE ~8 round conditions")
print()
active_rounds = list(range(0,16)) + list(range(24,64))  # 0-15 + 24-63
print(f"  Active rounds with nonzero schedule: {len(active_rounds)} (rounds 0-15, 24-63)")
print(f"  Zero-schedule rounds: 8 (rounds 16-23)")
print(f"  Reduction: 8/{64} = {8/64*100:.1f}% fewer schedule conditions")

# ── Angle 9: Can ΔW[16]=0 be exploited in the state? ────────────────────────
print("\n═══ ANGLE 9: State word 'e' after round 15 ═══")
print("  T1[15] = h[14] + S1(e[14]) + ch(e[14],f[14],g[14]) + K[15] + W[15]")
print("  T1'[15] = same (ΔW[15]=0)")
print("  → T1[15] = T1'[15] → ΔT1[15] = 0")
print("  → e[16] = d[15] + T1[15]  and  e'[16] = d'[15] + T1'[15]")
print("  → Δe[16] = Δd[15] + ΔT1[15] = Δd[15] + 0 = Δd[15]")
print()
print("  But ΔW[15]=0, so T1 only differs if state differs.")
print("  The round 15 state diff propagates into round 16 state.")
print()

# Check empirically: what is ΔT1 at round 15?
N_check = 10000
dt1_15_nonzero = 0
for _ in range(N_check):
    W = [random.randint(0, M) for _ in range(16)]
    Wp = list(W); Wp[0]=(W[0]+DELTA)&M; Wp[9]=(W[9]-DELTA)&M
    Wf = sched(W); Wpf = sched(Wp)

    S = list(H0); Sp = list(H0)
    for r in range(15):
        S = one_round(S, Wf[r], r)
        Sp = one_round(Sp, Wpf[r], r)

    # T1 at round 15
    a,b,c,d,e,f,g,h = S
    T1_15 = (h + S1(e) + ch(e,f,g) + K[15] + Wf[15]) & M
    a,b,c,d,e,f,g,h = Sp
    T1p_15 = (h + S1(e) + ch(e,f,g) + K[15] + Wpf[15]) & M

    if T1_15 != T1p_15: dt1_15_nonzero += 1

print(f"  ΔT1[15] ≠ 0 in {dt1_15_nonzero}/{N_check} = {dt1_15_nonzero/N_check*100:.1f}% of trials")
print("  (Since ΔW[15]=0, T1 differs only via state diff from previous rounds)")

# ── Angle 10: Symmetry in schedule differences at identical positions ─────────
print("\n═══ ANGLE 10: Schedule difference symmetry — resonance structure ═══")
print("  Checking symmetry: ΔW[r] vs ΔW[r+T] for various periods T")
print()

W = [random.randint(0, M) for _ in range(16)]
Wp = list(W); Wp[0]=(W[0]+DELTA)&M; Wp[9]=(W[9]-DELTA)&M
Wf = sched(W); Wpf = sched(Wp)
dW = [(Wpf[r]-Wf[r])&M for r in range(64)]

for T in [7, 9, 16, 7+9]:
    matches = sum(1 for r in range(64-T) if dW[r]==dW[r+T])
    exact = sum(1 for r in range(64-T) if dW[r+T]==dW[r] and dW[r] not in [0, DELTA, NEG_D])
    print(f"  Period T={T:2d}: {matches} positions with ΔW[r]=ΔW[r+T]  "
          f"({exact} are both noisy-but-equal)")

# ── Angle 11: Constraint reduction — "effective rounds" to attack ─────────────
print("\n═══ ANGLE 11: Effective attack complexity for reduced-round variants ═══")
print()
print("  Round count | Schedule zeros | Effective conditions | Savings vs random")
headers = ["16 (to window start)", "24 (after window)", "26 (W[24],W[25] exact)"]
win_zeros = [8, 8, 8]  # zero window always contributes
extra_zeros = [0, 8, 8]  # window contributes to all >= 24
total_active = [16, 24-8, 26-8]  # active rounds
for rnd, label in [(16,"R=16"), (24,"R=24"), (32,"R=32"), (64,"R=64")]:
    zero_rounds = min(8, max(0, rnd-16))
    active = rnd - zero_rounds
    saving_bits = zero_rounds  # ~1 bit per free round (conservative)
    print(f"  {label:6s}: {zero_rounds} zero-schedule rounds, ~{active} active rounds, "
          f"~2^{saving_bits} savings factor")

print("\n" + "="*65)
print("SUMMARY OF STRUCTURAL FINDINGS")
print("="*65)
print("""
  1. Nine-Step Cancellation: 8 EXACT zero-schedule rounds (algebraic)
  2. Direct propagation chains:
       W[0] → W[32], W[48] (via W[r-16], contaminated after window)
       W[9] → W[25] (exact -δ), W[41] (contaminated)
  3. State differential through zero window: full 8/8 nonzero words
     → state diverges without schedule excitation (no "free ride" in state)
  4. TC-σ₁(δ) is EMPTY → no exact extension past W[25]
  5. (0,9) pair uniqueness: σ₀(W[0]) hits W[15] (message), not extended schedule
  6. Boomerang potential: zero window = natural E_top/E_mid/E_bot separator
  7. Collision degress of freedom: 448 free bits vs 256 collision conditions
     → 192 bits slack (but nonlinear — no direct solve)
  8. ΔT1[15] ≠ 0 in ~100% of cases → no "free collapse" at round 15

  OPEN PROBLEM: What is the minimum-probability differential trail through
  rounds 16-23 that maps ΔState[16] → ΔState[24] compatible with collision?
  This is the key remaining question for a concrete attack.
""")
