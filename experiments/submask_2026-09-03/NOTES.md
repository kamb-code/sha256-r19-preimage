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
* "C3 uniform to 28 bits": the per-step conditional-halving tests are independent
  and null (GLRT p = 0.21), but the aggregate has little power at depth; the pooled
  28-bit cell is 10 vs 16.6 expected (one-sided p = 0.06), so the data bound the
  exact-zero rate only to within a factor of about 3. P(C3 = 0) = 2^-32 is an
  extrapolation over the last ~4 bits. State it that way.
* Campaign lo-pass lift 3.41x is trajectory-counted (duplicates 0.03%); with per-row
  counts only, identity-level deduplication can move it within [2.42x, 3.44x]; the
  audit's 1.79x was a valid but loose bound.
* Cost units: 2^47.3 swept a0 per 20-round preimage is the ONE-root figure; with
  three root tables it is 2^45.4 swept a0 (about 2^48.5 table lookups), about 14 h
  on one A100 SXM 80GB in expectation with the Triton kernel.
* The repository's `.gitignore` excluded `*.log`, so the first snapshot commit
  contained no logs although MANIFEST.txt listed them; they are force-added now.

* Final production rates from each Triton pod's status.json (a0/elapsed_s): dev 8.61e8, t1 8.84e8, t2 9.01e8 a0/s (4.8-5.0 s per context); the 8.2-8.8e8 / 5.2-5.3 s figures quoted earlier in this file and in FLEET.txt were early readings.
* The second all-ones pod's final exposure is 12,075 contexts / 5.186e13 a0 / 16.3 h (gpu_run_snapshot/live/t1/status.json), not the 12,016 / 5.161e13 counted at the moment of the hit.

* 2026-09-06: elimination search at C3 (analysis/elimination_search/): seven angles, all excluded, 0 bits. The structural statement is the "offset law": a_k enters W_{k+5} through Sigma0(a_k) (one input, nothing to saturate) and a1/a2/a3 have forced homes C0/C1/C2, so no fifth word can be absorbed at R=20; the (3/4)^32 collapse is the cheapest cut of the resulting cycle. Closest miss: a5 (deferrable in C0, linear in C3, blocked only by Sigma0(a5) in e6 -> C1).

* 2026-09-06: falsification test of the a5->C1 edge under a widened condition menu (analysis/elimination_search/falsify_a5/): ~1.59M legal full-width conditions + ~105M reduced-width assignments; global legal minimum 10.86 bits per flipped bit (threshold for absorption 2); zero below 4. Side result: a5 can be made exactly absent from C0 and C2 and exactly linear in C3, so the whole 20-round obstruction is this single edge.

* 2026-09-06: both papers submitted to the Cryptology ePrint Archive. Submission ids (pending; NOT citable report numbers): first paper (paper_r19_final, corrected resubmission of 110082) = 111557; second paper (paper_submask_r20) = 111558. On acceptance each receives a 2026/NNNN report number; then update the cross-citations (Basra2026R19 / Basra2026Submask bibitems, CITATION.cff, README) and file minor revisions.

* 2026-09-07: SAT baseline + algebra/SAT hybrid (analysis/sat_hybrid/): 28 jobs at R=19,20 with family conditions / fixed family context / random-context control, CaDiCaL and kissat, 1800 s cap -> 0 solved, all timed out; structured instances are slower than unstructured. On the identical fixed-context instance the table attack takes ~12 ms and the solvers >1800 s (>10^5 gap). The two methods do not compose. Symmetry angle untested (session limit).

* 2026-09-08: lattice/3D readings (analysis/lattice_3d/) and differential + message-word domain (analysis/differential_wdomain/), all 0 bits. Notable: (a) the barrier is NOT "cannot invert Sigma0" (the table does that in one lookup) but TWO pure 2^-32 value filters; WHAT_WOULD_BE_NEEDED.md corrected accordingly. (b) chunk-separated MITM has an exact ladder 2^96/2^128/2^160 at R=19/20/21, 83 bits worse than ours at 21. (c) 13,981,848 message differences on our five witnesses gave 0 second preimages. (d) the W-domain is a prefix-triangular re-coordinatisation of the a-domain, not a new frame. (e) symmetry is capped at 10.3 bits by a counting argument before any code runs.

* 2026-09-08: **INDEPENDENT EXTERNAL VERIFICATION.** Oleg Zaikin (Matrosov
  Institute; author of the Constraints 2026 paper whose all-ones digest is the
  target) wrote to say he had verified the 20-round preimage with his own
  implementation: "The preimage is correct! Excellent result!" He is reading
  both papers and asked how the methods arose and whether they will be
  submitted to a peer-reviewed journal. This is the first confirmation of the
  20-round result by a third party using independent code.
