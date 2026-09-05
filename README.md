# 19-Round SHA-256 Preimage Attack — Reproducibility Package

This repository contains everything needed to verify and reproduce the results
of two documents, kept deliberately separate.

> **Main paper** — "A Practical Preimage Attack on the 19-Round SHA-256
> Compression Function via a Global σ₀-Difference Table"
> `paper_r19_final.pdf` / `paper_r19_final.tex` (18 pp)
>
> The attack, seven verified preimages (six of random digests, one of the
> all-ones digest), the measured success rate, and the preregistered
> screening campaign. Its cost figures are for random contexts and are
> superseded by the second paper below.

> **Companion note** — "The 20-Round Barrier for Table-Based Preimage Attacks
> on SHA-256"
> `paper_r20_barrier.pdf` / `paper_r20_barrier.tex` (10 pp)
>
> Why the same construction does not reach 20 rounds. **Exploratory and
> largely negative**: an inventory of which absorptions admit a global table
> (every constraint C_j has one, for a_(j+1)), a measurement showing that the
> full-avalanche dependency graph among absorbed words is acyclic at 19 rounds
> and acquires its first cycle at exactly 20 (in the frame that absorbs a4 a
> single edge, a4 -> C0 through W9, 12.2 bits per flipped bit against <= 1.9
> for every feedback edge at 19), a second global table for `a0`, an exact
> linear identity for σ₀, and a reduced-width scale model. Not an
> impossibility result. Read the main paper first; nothing here modifies it.

> **Second paper** — "Context Shaping for Reduced-Round SHA-256
> Compression-Function Preimages: Nineteen Rounds in Milliseconds, and
> Computed Preimages at Twenty Rounds"
> `paper_submask_r20.pdf` / `paper_submask_r20.tex` (15 pp), filing package
> `SUBMISSION.md`
>
> Choosing the context so that a4 = a5 and e8 = e9 = 0xFFFFFFFF removes the
> fixed-point iteration from the 19-round attack and collapses its consistency
> check to a bitwise condition of probability (3/4)^32. Measured cost of a
> 19-round preimage: about 2^15.25 swept a0 (roughly 12 ms of one CPU core)
> against 2^38.3 for the random-context attack above. At 20 rounds the fourth
> schedule constraint remains a flat 2^-32 filter, giving 2^45.4 swept a0 with
> three-root tables; two computed and verified 20-round preimages are reported,
> one for the all-ones digest that Zaikin inverted at 19 rounds. **Supersedes
> the cost figures of the two documents above.** 21 rounds is shown to cost
> this construction the full additional 2^64. Evidence, logs, reviewer reports
> and a sha256 manifest: `experiments/submask_2026-09-03/`.

---

## What is claimed

A preimage attack on the **19-round SHA-256 compression function** in the
standard setting: the chaining value is the fixed IV, only the 32-byte target
digest is given, and the attacker chooses the 512-bit block. The solver finds
16 arbitrary 32-bit words W[0..15] such that

```
SHA256_compress_19(IV, W[0..15]) + IV == target
```

No padding constraint is imposed on W[0..15], so this is a claim about the
compression function, not about length-padded SHA-256 (see "Padding" below).

Seven independently verified preimages are included
(`verified_preimages.txt`): P1–P3 from dedicated end-to-end runs on digests
of random messages, P4–P6 recovered during the preregistered screening
campaign, and P7 for the all-ones digest used as the target by Zaikin. All
seven check against `code/verify_r19.py`.

**Prior work and priority.** SAT-based cryptanalysis (Davydov, Pikhtovnikov,
Kiryanova, Zaikin, 2025) found 17- and 18-round preimages of the zero digest
and reached 19 rounds only in a weakened form. Zaikin's *Constraints* article
(received March 2025, online 23 January 2026, DOI 10.1007/s10601-025-09383-0)
inverted the 19-round compression function first, for the all-ones digest
with all 256 bits fixed, in the same attack model as here: 18 h 33 min on
192 CPU cores (about 3,560 core-hours) with a parameterized Kissat inside
Cube-and-Conquer, via an intermediate problem between rounds 18 and 19. We
make **no priority claim** at 19 rounds. The contribution here is a
different, algebraic route whose cost is 25 to 89 s per attempt on one H100
(one or four seed chains) with roughly one attempt in eighty succeeding in
the four-chain campaign, i.e. about two GPU-hours per arbitrary digest in
expectation, with complete artefacts. **On his own target, the all-ones
digest, this attack found a different preimage (P7 below) after 84 attempts,
123 minutes on one H100**, so it is also a second preimage of that digest.

Full 64-round SHA-256 is not affected by any of this.

---

## Quick verification (no GPU, no lookup table)

Requirements: Python 3.8+, standard library only.

The two 20-round preimages of the second paper (each prints `result: OK`;
`--rounds 19` and `--rounds 21` print `FAIL`):

```bash
# all-ones digest (Zaikin's 19-round target), 20 rounds
python3 code/verify_r19.py --rounds 20 \
  --hash ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff \
  --words "0b0187bf b9aea692 b66effa5 087dca3a 0caea827 1d2f9916 739a224e a87c0eae 7f9ef4a7 b318a7de a8848c61 a7a141a4 11ac114b 952299be 97aaa67f c6acfe57"
# solver-generated digest, 20 rounds
python3 code/verify_r19.py --rounds 20 \
  --hash d962ca30635f9b74ac6c8c1243a1a9cf800e81bc05f1d2e40764c68c795f7388 \
  --words "a36f4238 f2c9204e b5b6653b 070401f7 5928e4f3 fe766be2 52026907 f7ee1812 01344603 ea505012 86cbe6cb 6ce73d0b 5c91de6a 43355e3b ff3d5e88 c1ad0b54"
```

The seven 19-round preimages of the main paper:

```bash
# P1
python3 code/verify_r19.py --rounds 19 \
  --hash 1e65261c54255188604f5375091839733de63e966b5e4715658226bf03588447 \
  --words "22f091af ec52d67b 74c33819 a280dc6a b001ff1a 1f2356a5 3eccf108 bd9a2333 \
           abe611d1 6d1e5a20 8041df25 e43d31af aa895a2e 69106ad2 7479fa3a 2a9abb91"

# P2
python3 code/verify_r19.py --rounds 19 \
  --hash fb52f81baed24f8728faf5bbce82c67d510761172fb9876d9e3a72dda351b7ca \
  --words "37e6702f bc20efea 2dd42a3e 501dfbe9 3cacc578 ea2de1c1 11c0f066 0f22be47 \
           2a447d2d 13f0080f 1f33df6b d655d8e6 15730eaa 9bf64950 9f129973 5a964edf"

# P3
python3 code/verify_r19.py --rounds 19 \
  --hash 1bd7ebbdc4d938fb26d19b5dd5caf333de397bd1c745727bd5556baf38ccf977 \
  --words "3ce8fba4 e2fb9661 44730c59 e1cf4bc0 e1a18d93 97658983 67efe2a7 ef260ecb \
           d4c6dbe0 13e9388e 95664a59 4d9e248b 74137862 664815ac 89eae95a cd7dbef5"

# P4
python3 code/verify_r19.py --rounds 19 \
  --hash c55aecafd68ac4f559bb2b940688f3bc1dabe3ff86a7d03447e8b11a637d7506 \
  --words "3694ee67 a1d020b9 47f08ea3 dc873e2b 1f41c8b1 39018970 d3a2afb2 23f6e6cf \
           80c46545 f7b2b46c 2e44f333 2effe68b 06d80cdf 8525e7d7 7ef008e5 13e56087"

# P5
python3 code/verify_r19.py --rounds 19 \
  --hash a8241289b5980304e9dfd401b1370200ab32b5e8cfaff26c26aabcfc8be89df4 \
  --words "3e0a7d67 df0944a3 4bfe156b bd344ac5 7b7d9587 8984e286 21c995d9 d8e53cdd \
           0a0be79d e02dda1a e17e0f4a e3ff692b 8c782631 97dc4811 5ed68656 dfce314f"

# P6
python3 code/verify_r19.py --rounds 19 \
  --hash 09156002031fed2899bc67d0e419b6ff81b9a0dd4e3fa417e000817adb85b143 \
  --words "164084a5 f97bd8bd bff47431 a8e55b29 948d1160 10eb8f7d d0ea565e 7918b8b9 \
           87527a3a 6baa5685 a1fbc5dc 7548adfc be3fb206 b9738756 b4ca33e3 b8354a95"

# P7 -- the all-ones digest, the target of Zaikin (Constraints, 2026)
python3 code/verify_r19.py --rounds 19 \
  --hash ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff \
  --words "3c6ff45c a5b23f8d dc6b8089 4f232935 79578e14 9c2da339 b22ec37a 9554191b \
           e69661f0 49a43b7a d8c60a2b 88d7588c 6225084d 87c95c4e 23ee573d 8d8fedfa"
```

All seven should print `result: OK`. The verifier is self-contained and
deliberately imports no attack code.

Two CPU checks of the algebra, each under a minute:

```bash
python3 code/edge_weights.py            # heavy-cycle measurement (companion note)
python3 code/padded_preimage.py --self-test
```

---

## Running the solver on your own target (NVIDIA GPU + CUDA)

Requirements: Python 3.10+, NumPy, tqdm, a CUDA build of PyTorch, and a GPU
with at least 24 GB of memory (the published runs used an H100 SXM 80 GB;
see `requirements.txt` for the exact versions recorded in the campaign
manifest). A CPU-only PyTorch install will not run the solver.

```bash
pip install numpy tqdm
# install the PyTorch CUDA wheel matching your driver

python3 code/h100_extended.py --hash <64-hex-char-target>     # attack one target
python3 code/h100_extended.py                                  # random targets, statistics mode
python3 code/h100_extended.py --help
```

The solver builds the σ₀(u)−u table in GPU memory (16 GB, about 2 s on an
H100), runs the backward chain on the target, samples random contexts
a[4..10], and sweeps a[0] over 2³² per context (about 25 s on an H100 with
`K_seeds=0`, about 89 s with `K_seeds=4`). Any preimage found is printed and
saved to a `.txt` file.

**Success rate.** The preregistered 240-context campaign measured
μ̂ = 3/240 = 1.25×10⁻² per context (95 % Poisson interval
[2.6×10⁻³, 3.7×10⁻²]); all three events were in the screened arm and the
control arm gave 0/120. The three dedicated runs succeeded in their 4th, 21st
and 35th contexts (solver indices 3, 20 and 34). Expect tens of contexts, i.e. tens of minutes, per preimage, with a
wide spread.

**Generate a target of your own:** run the verifier without `--hash` and it
prints the 19-round digest of the words you give it.

```bash
python3 code/verify_r19.py --rounds 19 \
  --words "00000000 00000001 00000002 00000003 00000004 00000005 00000006 00000007 \
           00000008 00000009 0000000a 0000000b 0000000c 0000000d 0000000e 0000000f"
```

**Table file for the CPU analysis scripts.** `measure_w9_filters.py`,
`context_screening.py` and `screening_validation.py` read the table from disk:

```bash
python3 code/build_sigma0_table.py --out sigma0_u_table.npy   # 16 GiB, ~3 min, needs ~17 GiB RAM
python3 code/build_sigma0_table.py --check sigma0_u_table.npy
```

---

### The submask family (second paper)

`code/submask_family.py` (CPU, numpy) runs the 19-round attack of the second
paper against the 16 GB table; with no arguments it runs a table-free
algebraic self-test of the paper's two propositions. `code/gpu_submask.py` is
the eager PyTorch sweep and `code/submask_triton.py` the fused kernel used for
both 20-round preimages; the kernel needs Triton (bundled with the torch wheel
recorded in `requirements.txt`) and an 80 GB GPU for three root tables:

```bash
python3 code/submask_triton.py --verify            # planted-preimage check of the kernel
python3 code/submask_triton.py --bench             # timing
python3 code/submask_triton.py --run --rounds 20 --roots 3 --hash <64-hex-char-target>
```

## Padding

Standard SHA-256 pads a message with `0x80`, zeros, and the 8-byte bit
length. For a single-block 55-byte message this fixes W[15] = 440, W[14] = 0
and the low byte of W[13]. The attack as published leaves W[0..15] free, so
its result is about the compression function.

`code/padded_preimage.py` shows that the three padding conditions cost
nothing to impose: in the round function `e[r] = a[r-4] + T1[r]`, so the
context words a5, a6, a7 enter W13, W14, W15 with coefficient exactly −1, and
the closed-form solve `a_new = a_old + (W_current − W_wanted)` in the order
a7 → a6 → a5 pins all three (200/200 contexts; independent of the swept
a0..a3, 1500/1500). Four context words stay free.

**No padded preimage has been demonstrated.** A 167-context H100 run
produced 6,762 lo-passes and zero hi-passes where about 1.1 hi-passes were
expected (P(0) ≈ 0.32), so the run is uninformative rather than negative.
Padded contexts yield normally (40.5 lo-passes per context against 33.7
unpadded). Raw log: `data_padded_run.txt`. The paper therefore claims only
the compression-function result.

---

## File listing

```
publish/
  README.md                  — this file
  LICENSE                    — MIT (code and data); papers are author copyright
  CITATION.cff               — citation metadata
  requirements.txt           — Python dependencies and the recorded versions
  paper_r19_final.pdf/.tex   — main paper (18 pages)
  RESUBMISSION_R19.md        — ePrint filing package for the main paper
  data_screening_validation.json, data_screening_280.txt,
  data_screening_notes_20260808.md — archived screening runs behind §5.7
  paper_r20_barrier.pdf/.tex — companion note (10 pages)
  paper_submask_r20.pdf/.tex — second paper (15 pages); supersedes the cost
                               figures of the two documents above
  SUBMISSION.md              — ePrint filing package for the second paper
  experiments/submask_2026-09-03/ — the second paper's evidence: as-run scripts,
                               logs, GPU run snapshots, witnesses with
                               provenance, reviewer and analysis reports,
                               priority search, sha256 MANIFEST.txt
  verified_preimages.txt     — the seven verified preimages with provenance
  data_campaign_screening.json — raw per-context data and run manifest of the
                               preregistered 240-context campaign (§5.7)
  data_edge_weights.txt      — raw output of code/edge_weights.py (companion §3)
  data_padded_run.txt        — raw log of the 167-context padded run
  data_ones_target_run.txt   — raw solver log of the run that found P7
                               (all-ones digest, 84 contexts, 7,373 s)

  code/                      — scripts behind the paper's claims:
    verify_r19.py            — standalone verifier (stdlib only)
    h100_extended.py         — production GPU solver (PyTorch + CUDA)
    campaign_screening.py    — the preregistered paired campaign (§5.7): 40
                               targets, 240 contexts, matched 2^32 exposure per
                               arm, no early stop. Result: 8.13x fixed-point
                               lift converts to 3.41x on W9 lo-passes (95 % CI
                               [2.13, 5.82]; one-sided sign test p = 3.4e-4);
                               helped on 31/40 targets, hurt on 9; conversion
                               to preimages unmeasured (3 vs 0, p = 0.125).
    context_screening.py     — per-context productivity and its predictiveness
                               (Spearman 0.91 between independent 2^20 samples;
                               top-1 % keep gives 12.99x FIXED-POINT lift, a
                               statement about fixed points only).
    screening_validation.py  — predictiveness and keep-fraction economics.
    measure_w9_filters.py    — W9 lo/hi filter pass rates with global fixed-point
                               deduplication and half-width independence tests;
                               needs the 16 GB table (--table).
    build_sigma0_table.py    — builds/checks the on-disk σ₀(u)−u table.
    edge_weights.py          — the heavy-cycle measurement of the companion note.
    padded_preimage.py       — padding as a closed-form three-word solve, with
                               --self-test (CPU) and --hash (GPU attack).
    verify_a0_absorber.py    — the a0-absorber proposition (companion §2.5):
                               Gamma(a0) = a0 + g(a0) = R + W1 − σ₀(W1) is
                               equivalent to C0; Gamma^-1 is a global table
                               (63.214 % coverage). --full adds the 2^32 scan.
    verify_sigma0_identities.py
                             — the σ₀ identities of companion §2.6:
                               rank(σ₀ XOR I) = 31, kernel {0, 0x27f42515},
                               left-null mask 0xa8a42fe4; --full for the
                               exhaustive pass and the 2^-15.8 bias.
    extended_solver.py       — backward chain and W recovery utilities
    sha256_core.py           — full-trace reference SHA-256
    utils.py                 — SHA-256 primitives

                               research artefacts, NOT part of the paper's
                               claims (see note below):
    alt_differential.py      — exhaustive survey of two-position schedule
                               differences (supports Appendix A's uniqueness
                               statement)
    zero_window_lemma.py     — numerical check of the Nine-Step lemma
    absorption_analysis.py, angle_analysis.py, block1_coord.py,
    block2_coord.py, cuda_sweep.py, deep_search.py, differential_trace.py,
    final_results.py, gpu_sa.py, near_collision_result.py,
    schedule_differential.py, sensitivity_matrix.py, threeblock_coord.py,
    twobit_search.py, twoblock_sweep.py
                             — GPU/CPU near-collision searches built on the
                               Nine-Step difference.
```

**Note on the near-collision scripts.** These searches found message pairs
with one equal output word of full SHA-256 and multi-block output
differences of Hamming weight 72/256. They are kept for completeness and
are **not claimed as results**: over 2³² fixed-difference trials the expected
minimum Hamming weight of an unbiased 256-bit difference is 78, and over the
2³⁷–2³⁹ evaluations of the multi-block searches it is 72–74, so the observed
minima are at the level of chance, and one-word equalities occur at the
birthday rate 2⁻³² per word. The main paper's Appendix A says the same.
Several of these scripts need CuPy and a GPU; `schedule_differential.py`
runs a 2²⁰-step CPU loop before its GPU phase.

---

## Key algebraic facts (main paper §4–§5)

**Proposition (provisional-W₉ form).** With Ŵ₉ computed from the context at
a₂ = a₃ = 0, Ŵ₉ − W₉ = a₁ + ε(a₂, a₃), where ε depends on a₂, a₃ only through
Maj and Ch terms and vanishes at a₂ = a₃ = 0. The attack proceeds as if
ε = 0 and tests ε(a₂, a₃) = 0 exactly afterwards (the "W9 filter"); the
measured pass rates are in §5.6.

**Proposition (C0 cancellation).** Under ε = 0,
F₀ = W₁₆_bc − σ₁(W₁₄) − g(a₀) − Ŵ₉ − W₀ = σ₀(W₁) − W₁, so one table lookup
returns W₁ and a₁ = W₁ − g(a₀).

**Proposition (C1/C2 cancellations).** In C1 the unknown a₂ cancels; in C2
(under ε = 0) the unknown a₃ cancels. Both reduce to table lookups, iterated
to a fixed point.

---

## Attack model

The preimage result concerns the one-block 19-round SHA-256 compression
function with the standard IV and no padding constraint on the 16 input
words. It is a preimage attack in the standard sense for the compression
function (fixed chaining value, only the digest given), not a pseudo-preimage
or free-start result, and not a preimage attack on padded SHA-256. The
companion note's 20-round analysis is superseded by the second paper
(`paper_submask_r20.pdf`), which computes 20-round preimages in the same
model; 21 rounds is shown there to cost this construction the full additional
2^64.

## How to cite

See `CITATION.cff`. Until an ePrint identifier is assigned:

> K. S. Basra. A Practical Preimage Attack on the 19-Round SHA-256
> Compression Function via a Global σ₀-Difference Table. Preprint, 2026.
> https://github.com/kamb-code/sha256-r19-preimage
