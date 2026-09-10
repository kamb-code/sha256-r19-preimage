# Revision Notes

Revision date: 2026-09-10

Baseline repository commit: `d4ef0965ee86a1685b16ddde359377794a428c81`.
The reviewed baseline PDFs had these SHA-256 values:

- `paper_r19_final.pdf`:
  `a6cecac47f19d8c64f771394b0fd7e2d4aa6b973a4b235fd29c73c284a837193`
- `paper_submask_r20.pdf`:
  `c3b3b3c29f74000420c5041a2376934bf3ee65982906f4fb431623fbfe053e68`

Final rebuilt PDFs:

- `paper_r19_final.pdf`: 19 pages,
  SHA-256 `c186b41786a2560d027540620130c778c392341e6570dddbc0e0fd6e0b1a2ae7`
- `paper_submask_r20.pdf`: 17 pages,
  SHA-256 `dc2d306420872cc3c8faedd64683651eb323d8004ad2b9d6f4cbceff1354c213`

The two manuscripts remain separate. The original R19 paper explains the
random-context fixed-point construction. The newer paper is self-contained,
supersedes the earlier performance figures, and presents the context-shaped
R19 construction and two computed R20 examples. The historical R20 companion
remains in the repository but is not an impossibility result.

## Claim-to-evidence record

| Claim type | Retained claim | Primary repository evidence |
|---|---|---|
| Exact algebra | The global `sigma0(u)-u` table and the R19 schedule identities reduce the original construction to table lookups plus a fixed-point iteration and an exact final check. | Propositions in `paper_r19_final.tex`; `code/verify_sigma0_identities.py`; `code/submask_family.py` |
| Exhaustive table fact | The image of `sigma0(u)-u` has 2,721,603,628 values. | `code/build_sigma0_table.py`; archived table-build records |
| Computed R19 result | Seven supplied blocks forward-evaluate to their stated 19-round targets with standard IV and feed-forward. | `verified_preimages.txt`; `code/verify_r19.py` |
| R19 campaign measurement | Screened arm 3/120, control arm 0/120; pooled 3/240 describes the stratified campaign. | `data_campaign_screening.json` |
| R19 screening measurement | The 3.41 factor is trajectory-counted; row aggregates bound the identity-deduplicated factor within 2.42 to 3.44. Its effect on final yield is unresolved. | `data_campaign_screening.json`; `data_screening_validation.json`; `data_screening_280.txt` |
| Exact context-family algebra | With `a4=a5` and `e8=e9=0xFFFFFFFF`, three constraints are triangular and the final R19 equality is `Maj(a4,a3,a2)=a3`. | Propositions in `paper_submask_r20.tex`; `code/submask_family.py`; reviewer algebra scripts |
| R19 context-family measurement | Three recorded data sets total 1,142,947,840 swept values and 29,398 verified examples, or one per 38,878. | `experiments/submask_2026-09-03/logs/r19_*.log`; `reviewers/reports.json` |
| Derived R19 timing | About 12 ms is a marginal estimate from measured single-core throughput and measured yield after table construction. | `reviewers/reports.json`; R19 logs |
| Exact R20 condition | R20 adds the exact residual equality `c3=0`. | Derivation in `paper_submask_r20.tex`; planted validation records |
| R20 residual model | Existing tests are consistent with an exact-zero rate of `2^-32`, but do not measure it or prove joint uniformity. | `logs/c3_uniformity.log`; `logs/c3_scaling.log`; GPU status records; `reviewers/round2/stats_review.py` |
| Computed R20 result | Two supplied blocks forward-evaluate to their stated 20-round targets. | `R20_PREIMAGE.txt`; `R20_PREIMAGE_ALLONES.txt`; `code/verify_r19.py` |
| R20 observed resources | Successful runs took 2.45 and 16.16 GPU-hours; six recorded production runs total about 71.7 GPU-hours. | `gpu_run_snapshot/**/status.json`; `FLEET.txt` |
| Conditional R20 cost | About `2^45.4` swept values and 15 A100-hours is a model prediction, not a measured mean. | Yield derivation plus recorded production throughput |
| R21 exploration | Finite context-family, dependency, and residual tests did not improve the tested construction. A `2^64` extra factor relative to R19 is conditional on two independent-uniform residual assumptions. | `analysis/elimination_search/`; `analysis/scripts/symdep/`; `analysis/STATUS_AND_STOPPING_POINT.md` |
| Literature statement | No earlier computed witness at R20 or higher was found in the dated, scoped search. This is not a proof of priority. | `experiments/submask_2026-09-03/priority/` |

## Substantive corrections

### Original R19 manuscript

- Rewrote the abstract around the exact setting, demonstrated witnesses, and
  stratified campaign rather than a precise expected runtime.
- Distinguished 25-second one-chain sweeps from 89-second four-chain campaign
  sweeps.
- Replaced the pooled campaign rate as a random-context estimate with the
  screened/control counts and a highly uncertain 3.3-hour population-weighted
  plug-in value.
- Stated that the 3.41 screening factor is trajectory-counted and retained its
  identity-level bounds; no final-preimage speedup is claimed.
- Clarified that the R19 campaign is not exactly replayable because the commit,
  complete target identifiers, and random source messages were not retained.
- Removed the unsupported eightfold extrapolation from Zaikin's one-in-eight
  intermediate acceptance probability.
- Corrected representative-table and sentinel wording.
- Replaced the old general R20 barrier conclusion with a finite,
  representation-specific historical statement and pointed to the newer paper.

### Context-shaped R19/R20 manuscript

- Shortened the title and made the paper independently define the exact attack
  model, SHA-256 functions, state convention, backward chain, lookup equation,
  and sufficiency of the schedule constraints.
- Rewrote the abstract to separate exact identities, measured R19 yield,
  observed R20 runs, and conditional R20/R21 cost models.
- Clarified that the `3^32` count is exact for uniform independent word pairs,
  while lookup yield uses additional distribution assumptions tested only on
  recorded samples.
- Corrected multi-root terminology: 0.8979 and 0.9774 are mean retained roots
  per table index, not image coverage.
- Defined the 12 ms result as a derived marginal CPU time and kept cold-start,
  memory, and preprocessing costs separate.
- Replaced unconditional-independence language for nested residual thresholds
  with conditional-binomial transitions, a multiple-comparison-adjusted
  summary, and the correct 24-bit occupancy limit. Recalculation gives exact
  one-sided `p=0.00843` for the most extreme transition and Bonferroni-adjusted
  `p=0.0674` over the eight examined transitions.
- Removed a distribution-free exact-zero bound. The `2^-32` R20 filter rate
  remains an explicit extrapolation.
- Narrowed Fourier, mutual-information, root-policy, and context-search
  conclusions to the tested distributions, statistics, and detection limits.
- Separated the two successful runtimes, all unsuccessful/concurrent records,
  the 71.7 GPU-hour total, and the model-predicted mean.
- Recast all R21 statements as finite negative evidence and conditional cost
  estimates, not impossibility, independence, or optimality claims.
- Scoped literature novelty and priority statements to the recorded search.
- Reworded automated checking according to what each session actually did.

## Supporting-file changes

- Reorganized `README.md` around the exact setting, the two manuscripts,
  verification commands, evidence map, and current publication status.
- Updated `CITATION.cff` to prefer the newer manuscript while retaining the
  original paper and historical companion as separate references.
- Replaced stale filing text in `SUBMISSION.md` and
  `RESUBMISSION_R19.md` with current manuscript records and qualified
  abstracts.
- Updated the experiment status and correction notes; private correspondence
  is not used as public evidence.
- Corrected the table-builder's sentinel documentation.

## Review findings already resolved or not applicable

- The reviewed PDF hashes matched the repository baseline. The rendered and
  source definitions of `Ch` include the required negation; the suggested
  missing-negation issue was a text-extraction concern, not a manuscript error.
- The current R20 construction does not use the earlier proposed `h`-table or
  an injectivity assumption for an `a11 -> W12` map. Comments about proving or
  collision-hardening that table apply to a superseded draft, not to
  `paper_submask_r20.tex`.
- The current papers already expose all message words for each witness and use
  a standalone forward verifier; no omitted word-recovery step is required to
  check the published examples.
- The earlier eightfold runtime extrapolation from Zaikin's stated `1/8`
  acceptance probability was present in the baseline discussion and has been
  removed from both revised manuscripts.

## Remaining limitations

- No padded-message preimage is demonstrated.
- The exact-zero R20 residual probability is not directly measured.
- Two successful R20 examples do not determine a mean or a target-independent
  distribution.
- The old R19 stratified campaign has only three final events and cannot be
  replayed byte for byte.
- Finite negative R21 experiments do not rule out another representation or
  construction.
- The manuscripts remain unpublished and have no ePrint identifiers.

## Verification performed

The final revision process includes:

- forward evaluation of all seven R19 and both R20 supplied blocks;
- failure checks for the R20 blocks at adjacent round counts;
- recalculation of the stated campaign weighting, production-time total, and
  nested-threshold summary from existing records;
- two-pass LaTeX builds of both manuscripts;
- checks for unresolved references, LaTeX errors, and overfull boxes;
- visual inspection of every final PDF page;
- consistency checks across abstracts, conclusions, README, citation metadata,
  and manuscript records.

No new candidate search or GPU campaign was performed for this revision.
