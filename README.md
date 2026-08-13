# Oracle-Free 19-Round SHA-256 Preimage — Reproducibility Package

This folder contains everything needed to verify and reproduce the results in
two papers, kept deliberately separate.

> **Main result** — "Oracle-Free Preimage Attack on 19-Round Reduced SHA-256"
> `paper_r19_final.pdf` / `paper_r19_final.tex` (24 pp)
>
> The 19-round preimage attack with three verified preimages, the Nine-Step
> Cancellation Lemma and collision structure, and context screening. This is
> the finished, self-contained result.

> **Companion** — "The 20-Round Barrier for Table-Based Preimage Attacks on
> SHA-256"
> `paper_r20_barrier.pdf` / `paper_r20_barrier.tex` (8 pp)
>
> Why the same construction does not reach 20 rounds. **Exploratory and
> largely negative**: an inventory of which absorptions admit a global table, a
> forced assignment leaving a cyclic residue, a second global table for `a0`, an
> exact linear identity for sigma0, and a reduced-width scale model. Not an
> impossibility result. Read the main paper first; nothing here modifies it.

---

## What is claimed

An oracle-free preimage attack on the **19-round reduced SHA-256 compression function**
initialized with the standard IV. Given only a 32-byte target hash, the solver
finds 16 arbitrary 32-bit message words W[0..15] such that:

```
SHA256_compress_19(IV, W[0..15]) + IV == target_hash
```

No padding constraint is imposed on W[0..15]. This is a claim about the compression
function, not standard padded SHA-256.

Six independently verified preimages are included (`verified_preimages.txt`):
P1-P3 from dedicated end-to-end attacks, P4-P6 recovered during the
preregistered screening campaign. All six check against `code/verify_r19.py`.

---

## Quick verification (no GPU, no lookup table needed)

Requirements: Python 3.8+, no third-party packages.

```bash
# Verify P1
python3 code/verify_r19.py --rounds 19 \
  --hash 1e65261c54255188604f5375091839733de63e966b5e4715658226bf03588447 \
  --words "22f091af ec52d67b 74c33819 a280dc6a b001ff1a 1f2356a5 3eccf108 bd9a2333 \
           abe611d1 6d1e5a20 8041df25 e43d31af aa895a2e 69106ad2 7479fa3a 2a9abb91"

# Verify P2
python3 code/verify_r19.py --rounds 19 \
  --hash fb52f81baed24f8728faf5bbce82c67d510761172fb9876d9e3a72dda351b7ca \
  --words "37e6702f bc20efea 2dd42a3e 501dfbe9 3cacc578 ea2de1c1 11c0f066 0f22be47 \
           2a447d2d 13f0080f 1f33df6b d655d8e6 15730eaa 9bf64950 9f129973 5a964edf"

# Verify P3
python3 code/verify_r19.py --rounds 19 \
  --hash 1bd7ebbdc4d938fb26d19b5dd5caf333de397bd1c745727bd5556baf38ccf977 \
  --words "3ce8fba4 e2fb9661 44730c59 e1cf4bc0 e1a18d93 97658983 67efe2a7 ef260ecb \
           d4c6dbe0 13e9388e 95664a59 4d9e248b 74137862 664815ac 89eae95a cd7dbef5"
```

All three should print `result: OK`.

The verifier (`code/verify_r19.py`) is self-contained and deliberately does not
import any attack code.

---

## Running the solver on your own target hash (requires NVIDIA GPU + CUDA)

Requirements: Python 3.10+, NumPy, tqdm, a CUDA-enabled PyTorch build, NVIDIA GPU
(H100 recommended). A CPU-only PyTorch install will not run the solver.

```bash
pip install numpy tqdm
# Install PyTorch using the CUDA wheel appropriate for your driver/CUDA setup.

# Attack a specific 19-round target hash of your choice:
python3 code/h100_extended.py --hash <64-hex-char-hash>

# Example (reproduces P1):
python3 code/h100_extended.py \
  --hash 1e65261c54255188604f5375091839733de63e966b5e4715658226bf03588447

# Run with default random targets (benchmark / statistical mode):
python3 code/h100_extended.py

# Full options:
python3 code/h100_extended.py --help
```

The solver:
1. Builds the σ₀(u)−u representative table in GPU memory (~16 GB, ~1.8 s on H100).
2. Runs the backward chain on the target hash to fix the high state words.
3. Samples random contexts for the free state words a[4..10].
4. Sweeps a[0] over 2³² values per context using the C0/C1/C2 cancellation chain.
5. Prints any found preimage to stdout and saves it to a `.txt` file.

No precomputed table file is required — the table is rebuilt from scratch each run.
Expected time to first hit on H100: not yet measured. The three recorded
attacks succeeded after 3, 20 and 34 contexts (~25 s per context), but three
observations cannot establish a distribution — the exact 95% interval on that
success rate is [0.011, 0.146]. A larger multi-target campaign is in progress.

**Generate your own target to attack:**
Choose any 16 input words and run the verifier without `--hash`; it will print
the corresponding 19-round target hash. Then pass that hash to the solver.

```bash
python3 code/verify_r19.py --rounds 19 \
  --words "00000000 00000001 00000002 00000003 00000004 00000005 00000006 00000007 \
           00000008 00000009 0000000a 0000000b 0000000c 0000000d 0000000e 0000000f"
```

---

## Does padding matter? Why W[0..15] are unconstrained here

Standard SHA-256 appends a specific padding to each message before hashing:
a `0x80` byte, then zeros, then the 8-byte message bit-length. For a message
shorter than 56 bytes, this all fits in one 512-bit block, so several of the
16 input words are fixed by the padding format.

**This attack does not use standard padding.** W[0..15] are treated as 16
arbitrary 32-bit words — no structure is required. This gives the solver the
maximum possible freedom to satisfy the 19-round equations.

**What this means in practice:**

| Scenario | Applies? |
|---|---|
| "Find W[0..15] s.t. 19-round compress(IV,W)+IV = T" (this paper) | ✅ Yes |
| "Find a padded message s.t. standard SHA-256(msg) = T" (real preimage) | ❌ Not directly |

To attack a padded message preimage you would need to additionally satisfy
the padding constraints (e.g. W[14]=0, W[15]=bit_length, specific 0x80 byte).
That reduces the attacker's free variables from 16 to roughly 13–14, and
propagates constraints into the schedule words (W[16], W[17], ...) that the
attack depends on. A padded-message variant would require extending the method
to handle those fixed values — this is not done here and remains an open problem.

**Update (2026-08-13): the padding constraints turn out to be nearly free.**
See `code/padded_preimage.py`. In the round function `e[r] = a[r-4] + T1[r]`,
so context words a5, a6, a7 enter W13, W14, W15 additively and each padding
word is linear in its context word with coefficient exactly -1. Solving in the
order a7 -> a6 -> a5 pins all three in ONE closed-form pass with no search:

    a_new = a_old + (W_current - W_wanted)

Padding therefore costs three context words, not an exponential factor;
a4, a8, a9, a10 stay free (128 bits). Verified: 200/200 contexts land exactly
on W15=440, W14=0, W13&0xFF=0x80, and the padding is independent of the swept
a0..a3 (1500/1500), which is what makes it survive the sweep.

HOWEVER, no padded preimage has yet been demonstrated. A 167-context H100 run
produced 6,762 lo-passes and zero hi-passes; against the unpadded rate the
expectation was 1.14, so P(0) = 0.32 -- consistent with bad luck, but NOT a
demonstration. The padded contexts yield normally (40.5 lo/ctx against 33.7
unpadded), so there is no sign of degradation. Raw log: `data_padded_run.log`.

**The bottom line:** the method is the same (backward chain + σ₀-differential
table + C0/C1/C2 cancellations); padding is a closed-form three-word solve
rather than an obstacle, but a demonstration artifact is still outstanding.
The security of full 64-round padded SHA-256 is not affected by this result.

---

## File listing

```
publish/
  README.md                  — this file
  paper_r19_final.pdf        — main paper, R=19 (24 pages)
  paper_r19_final.tex        — LaTeX source
  paper_r20_barrier.pdf      — companion, R=20 barrier (8 pages)
  paper_r20_barrier.tex      — LaTeX source
  verified_preimages.txt     — the three verified preimage examples

  code/                      — core reproducibility files:
    verify_r19.py            — standalone verifier (no dependencies beyond stdlib)
    measure_w9_filters.py    — measures the W9 lo/hi filter pass rates that set
                               the complexity constant mu. Deduplicates fixed
                               points globally, tests lo/hi independence at
                               several half-widths, and reports the DIRECT
                               estimator mu = N_LH / C (with a Poisson upper
                               bound when no full match is observed, which is
                               the current situation). Pass --table with the
                               path to the 16 GB sigma0(u)-u table; ~1 h on
                               14 cores for ~1.4 context-equivalents.
    context_screening.py     — measures per-context productivity and its
                               predictiveness. Contexts are ~1500x overdispersed
                               and a cheap 2^20 screening pass predicts yield
                               with Spearman 0.91, giving ~12.7x higher
                               out-of-sample C1/C2 FIXED-POINT yield per
                               attacked context.
    campaign_screening.py    — the preregistered paired campaign that measures
                               what screening is actually worth. 40 targets,
                               240 fully attacked contexts, matched 2^32
                               exposure per arm, no early stop. RESULT: the
                               8.13x fixed-point lift converts to only 3.41x
                               on W9 lo-passes (95% CI [2.13, 5.82], sign test
                               p=3.4e-4) -- conversion is SUBLINEAR. Helped on
                               31/40 targets, hurt on 9. Conversion through the
                               final 32-bit filter to preimages remains
                               unmeasured (screened 3, control 0, p=0.050).
    screening_validation.py  — predictiveness + keep-fraction economics
                               when the best 1% are kept.
    verify_a0_absorber.py    — verifies Proposition (a0 absorber):
                               Gamma(a0) = a0 + gv(a0) = R + W1 - sigma0(W1)
                               is equivalent to the C0 schedule constraint.
                               gv depends only on a0 and the IV, so Gamma^-1 is
                               a GLOBAL table (63.214% coverage). Checked on all
                               three preimages; --full adds the 2^32 scan.
    verify_sigma0_identities.py
                             — checks the sigma0 linear identities of §"Discussion:
                               R=20 Status": rank(sigma0 XOR I) = 31, the kernel
                               {0, 0x27f42515}, and the unique left-null mask
                               lambda = 0xa8a42fe4 satisfying
                               lambda·(sigma0(W) XOR W) = 0 for every W.
                               Run with --full for the exhaustive 2^32 pass and
                               the modular-subtraction bias (2^-15.8).
    h100_extended.py         — production GPU solver (PyTorch + CUDA)
    extended_solver.py       — backward chain + W recovery utilities
    sha256_core.py           — SHA-256 full-trace reference implementation
    utils.py                 — SHA-256 primitives (ROTR, Σ, σ, Ch, Maj, H0, K)

                               auxiliary research scripts for the full-round
                               differential / near-collision sections:
    absorption_analysis.py   — multi-block coordinate descent experiments
    alt_differential.py      — alternate differential experiments
    angle_analysis.py        — differential angle analysis
    block1_coord.py          — single-block coordinate descent
    block2_coord.py          — two-block coordinate descent
    cuda_sweep.py            — CUDA birthday sweep for near-collisions
    deep_search.py           — extended birthday search
    differential_trace.py    — differential trace logging
    final_results.py         — result aggregation
    gpu_sa.py                — GPU simulated annealing (experimental)
    near_collision_result.py — near-collision result logging
    schedule_differential.py — schedule differential analysis
    sensitivity_matrix.py    — sensitivity matrix computation
    threeblock_coord.py      — three-block coordinate descent
    twobit_search.py         — two-bit differential search
    twoblock_sweep.py        — two-block birthday sweep
    zero_window_lemma.py     — zero-window lemma verification
```

The lightweight verification scripts for the displayed full-round
near-collision claims are `final_results.py` and `zero_window_lemma.py`.
The larger GPU/coordinate-descent scripts may require CuPy/CUDA and were kept as
research artifacts rather than polished command-line tools.

---

## Key algebraic identities (see paper §4–§5)

**Lemma 1 (W9 differential).**
Define Ŵ₉ = W₉_sched(a₁=0). Then Ŵ₉ − W₉_real = a₁.

**Proposition 1 (C0 cancellation).**
F₀ = W₁₆_bc − σ₁(W₁₄) − g(a₀) − Ŵ₉ − a₀ − C₀_const = σ₀(W₁) − W₁.
So a₁ cancels and the σ₀-differential table recovers W₁, hence a₁ = W₁ − g(a₀).

**Proposition 2 (C1/C2 cancellations).**
In C1 the target unknown a₂ cancels; in C2 the target unknown a₃ cancels.
Both reduce to σ₀-differential table lookups.

These three identities together make the full 2³² sweep tractable on a single GPU.

---

## Attack model disclaimer

The preimage result concerns only the one-block reduced-round SHA-256
**compression function** (19 of 64 rounds), initialized with the standard IV and
with no padding constraint on the 16 input words. The later differential
experiments concern the full 64-round compression function, also on arbitrary
one-block inputs without a padding constraint. This is **not** a preimage attack
on standard padded SHA-256.

The oracle-free 20-round case remains open (see paper §7).
