# Corrections recorded after the second review (2026-09-04)

The as-run scripts in `scripts/` are preserved unchanged. The following statements
printed by them or made in the project log are WRONG and are corrected here:

* `plant_r20.py` prints "expected found fraction with two roots ~ (2/e)^3 = 0.40".
  The planted preimage's true root is size-biased (it is a root), so the right
  quantity is P(true root among the k stored | u is a root) = E[min(P+1,k)/(P+1)]
  with P ~ Poisson(1), which equals E[min(r,k)] per lookup: 0.632 / 0.896 / 0.976
  for k = 1 / 2 / 3, hence found fractions 0.25 / 0.72 / 0.93 over three lookups.
  Observed: 10/16 (two roots, this run), 145/200 = 0.725 (a reviewer's 200 plants),
  54/200 = 0.27 with one root. The data agreed with the correct formula all along.
* `quantized_table.py` prints that at B = 24 the completion arithmetic is "13x below
  the memory bottleneck". The int32 throughput constant used (6e13/s) is too high;
  with a realistic figure the 256 completions per lookup are NOT free on an H100 or
  an A6000. The exactness of the scheme stands (165,819/165,819 recovered; never
  wrong by construction). A reviewer notes sigma0 is GF(2)-linear and the completion
  bits are disjoint from the stored bits, so sigma0(u) can be formed from a
  precomputed sigma0(i << B), reducing each completion to a few ops; and that
  interleaving the k roots into one record with validity flags removes most of the
  completions. Neither has been benchmarked. Treat quantization as unproven.
* Fleet rate: 0.216/h is the bench figure; production wall time is 5.2-5.3 s per
  context, giving ~0.19/h. Quote the production figure.
* "C3 uniform to 28 bits": the counts at successive thresholds are nested.
  Conditional on n_k, the transition n_(k+1) is Binomial(n_k, 1/2) under the
  null, but the observed ratios do not thereby become unconditionally
  independent because their denominators are random. On the H100 record, the
  aggregate transition score over bits 20--28 is 2,654 of 5,370 against
  2,685 expected (z = -0.85); the most extreme of the eight inspected steps
  has exact unadjusted one-sided p = 0.00843 and Bonferroni-adjusted
  p = 0.0674. The
  deepest cell with useful occupancy is at 24 bits. P(C3 = 0) = 2^-32 remains
  a model-based extrapolation, not a measured rate or a distribution-free
  factor-three bound.
* Campaign lo-pass lift 3.41x is trajectory-counted (duplicates 0.03%); with per-row
  counts only, identity-level deduplication can move it within [2.42x, 3.44x]; the
  audit's 1.79x was a valid but loose bound.
* Cost units: 2^47.3 swept a0 per 20-round preimage is the ONE-root figure; with
  three root tables it is 2^45.4 swept a0 (about 2^48.5 table lookups), about
  15 h on one A100 at the production rate under the exact-zero residual model.
* The repository's `.gitignore` excluded `*.log`, so the first snapshot commit
  contained no logs although MANIFEST.txt listed them; they are force-added now.

* Final production rates from each Triton pod's status.json (a0/elapsed_s): dev 8.61e8, t1 8.84e8, t2 9.01e8 a0/s (4.8-5.0 s per context); the 8.2-8.8e8 / 5.2-5.3 s figures quoted earlier in this file and in FLEET.txt were early readings.
* The second all-ones pod's final exposure is 12,075 contexts / 5.186e13 a0 / 16.3 h (gpu_run_snapshot/live/t1/status.json), not the 12,016 / 5.161e13 counted at the moment of the hit.

* 2026-09-06: elimination search at C3
  (analysis/elimination_search/): seven tested angles produced no improvement.
  In the tested representation, condition grammar and solve order, the
  "offset law" puts Sigma0(a_k) into W_(k+5), and a1/a2/a3 retain fixed
  C0/C1/C2 roles. The closest tested miss is a5 (deferrable in C0, linear in
  C3, blocked by Sigma0(a5) in e6 -> C1). This is not an impossibility or
  optimality result for other representations.

* 2026-09-06: falsification test of the a5->C1 edge under a widened condition
  menu (analysis/elimination_search/falsify_a5/): about 1.59 million legal
  full-width conditions plus about 105 million reduced-width assignments;
  global legal minimum 10.86 changed bits per flipped bit (tested absorption
  threshold 2), with none below 4. In this representation a5 can be made
  absent from C0 and C2 and linear in C3, leaving a5 -> C1 as the observed
  edge. The result is finite and representation-specific.

* 2026-09-06: both papers were submitted to the Cryptology ePrint Archive:
  first paper 111557 and second paper 111558. These were submission IDs, not
  citable report numbers. Both submissions were later declined under the
  archive's general editorial criteria; no paper-specific technical error was
  identified. The revised manuscripts have no ePrint identifiers.

* 2026-09-07: SAT baseline + algebra/SAT hybrid (analysis/sat_hybrid/): 28 jobs at R=19,20 with family conditions / fixed family context / random-context control, CaDiCaL and kissat, 1800 s cap -> 0 solved, all timed out; structured instances are slower than unstructured. On the identical fixed-context instance the table attack takes ~12 ms and the solvers >1800 s (>10^5 gap). The two methods do not compose. Symmetry angle untested (session limit).

* 2026-09-08: lattice/3D readings (analysis/lattice_3d/) and differential plus
  message-word-domain tests (analysis/differential_wdomain/) found no measured
  improvement in their tested representations. Notable: (a) the table can
  invert Sigma0(u)-u, while the remaining R21 cost comes from two exact value
  equalities whose 2^-32 rates are modelling assumptions; (b) the tested
  chunk-separated meet-in-the-middle formulation has an exact
  2^96/2^128/2^160 ladder at R19/R20/R21; (c) 13,981,848 tested message
  differences on five witnesses gave no second preimage; (d) the tested
  W-domain is a prefix-triangular re-coordinatisation of the a-domain; and
  (e) a counting argument bounds the tested symmetry mechanism at 10.3 bits.
