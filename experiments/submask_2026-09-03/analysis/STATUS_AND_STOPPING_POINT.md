# R21 Analysis Status and Stopping Point

Updated 2026-09-10. This is a scoped record of the finite R21 analyses, not a
claim that another construction is impossible.

## Established results

- The context-shaped R19 construction has exact algebraic identities and
  measured yield. Its approximately 12 ms figure is a derived marginal CPU
  estimate after reusable table construction.
- Two supplied R20 blocks forward-verify in the standard-IV, feed-forward,
  full-output compression-function model.
- The R20 model predicts `2^45.4` swept `a0` values and about 15 A100-hours
  per example with three retained roots. This prediction assumes an exact-zero
  fourth-residual rate of `2^-32`; that rate is not directly measured.

The finite negative experiments below do not alter those demonstrated facts.

## What was tested at R21

The archived directories `elimination_search/`, `falsify_a5/`,
`sat_hybrid/`, `lattice_3d/`, and `differential_wdomain/` examine:

- context conditions expressible in the implemented condition grammars;
- value-table absorption in the recorded state and message-word
  representations;
- the implemented SAT and algebra/SAT formulations;
- tested carry, lattice, symmetry, differential, and meet-in-the-middle
  decompositions;
- joint residual statistics on 214,583 generated candidates.

These experiments include positive controls that recover known properties,
including the R19 bitwise collapse and planted examples. Within the tested
representations, they found no reduction in the two remaining R21 exact
equalities.

## What the evidence supports

The dependency searches repeatedly locate a high-diffusion path through
`Sigma0` of the candidate word. The closest tested context family removes
the other paths but leaves that one. The joint-residual sample shows no
detected first- or second-order departure from the tested reference
distributions.

This supports a practical stopping decision for the current construction. It
does not prove:

- that each exact equality has probability exactly `2^-32`;
- that the two equalities are jointly independent at exact zero;
- that no untested context family, representation, or algebraic construction
  can do better;
- a lower bound or optimality theorem for reduced-round SHA-256.

Under the explicit independent-uniform model, R21 adds `2^64` work relative
to the R19 candidate generator, or `2^32` relative to the R20 model. Those
are conditional estimates.

## Reopening criterion

Further work on R21 is best resumed only with a concrete mechanism that
removes or shares one of the two exact residual equalities, or with a new
representation whose candidate generator can be tested against the existing
positive controls. Repeating the archived finite searches without such a
change would add little evidence.

## Publication status

The September 2026 ePrint submissions 111557 and 111558 were declined under
the archive's general editorial criteria without a paper-specific technical
error being identified. Revised manuscripts are maintained at the repository
root. The submission IDs are not ePrint report numbers.
