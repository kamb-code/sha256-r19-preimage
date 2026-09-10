# R21 Joint-Tail Audit

Date: 2026-09-10

## Scope

This note audits candidate-generation methods for the one-block, 21-round
SHA-256 compression function with the standard IV, feed-forward retained, all
256 output bits fixed, and an unrestricted 512-bit input block. It records no
21-round result.

## Re-established baseline

The current submask-family pipeline determines candidates satisfying C0, C1,
C2 and the bitwise collapse. It then evaluates two exact residuals:

    c3 = sigma0(W4) + W3 - K3p
    c4 = sigma0(W5) + W4 - K4p

The recorded 214,583-candidate sample found no first- or second-order signal
that reduces these residuals. This supports a conditional uniform-residual cost
model; it is not an independence proof or an impossibility result.

## What the existing searches establish

- Repeating the triangular frame at a later index moves unresolved constraints
  to earlier indices; it does not remove their information cost.
- The singleton dependency search tests acyclic, one-constraint/one-variable
  schedules under its context grammar. It cannot test a joint two-variable
  solve or a provisional dependency that is corrected later.
- The committed `L5_50_symdep.py` contains only the control invocation. The
  driver that produced `L5_51_search.log` and its 1,485-family result is absent,
  so that logged enumeration is not currently reproducible from the repository.
- The dedicated a5 study strongly supports that changing a5 alone leaves a
  high-diffusion C1 table-index dependence. It does not directly test a joint
  change of a5 with the C1 output coordinate.
- The old `sha256/oracle_debug/r21_attack.py` imports state and message values
  from a known input. It is a reference-conditioned diagnostic, not a
  target-only R21 procedure.

## Exact joint-tail observation

Because SHA-256's small sigma0 is a full-rank GF(2) map, C3 and C4 can be read
as a sequential tail recovery for fixed preceding data:

    W4 = sigma0_inverse(K3p - W3)
    W5 = sigma0_inverse(K4p - W4)

Thus C3 and C4 are not intrinsically two uninvertible equations. They become
two residual checks in the current construction because a4 and a5, hence W4
and W5, have already been fixed before those equations are reached.

The obstacle moves to state consistency:

    a4 = W4 - (W4 - a4)
    a5 = W5 - (W5 - a5)

Here `W4-a4` is determined by a0..a3, but `W5-a5` depends on the newly recovered
a4. Recovering a4 and a5 is therefore triangular. The remaining question is
whether the context equations used by C0--C2 can accommodate these recovered
values without introducing a new full-width condition.

## Open experiment

Build a reduced-width model with positive controls and compare three systems:

1. Current frame: C0--C2 determine a1--a3; C3/C4 are residual checks.
2. Tail-lift frame: C3/C4 determine W4/W5 and therefore a4/a5.
3. Composite frame: jointly parameterise the C1 output and a5, testing whether
   a coordinate such as `y = a2 - Sigma0(a5)` breaks the feedback cycle.

For each width, enumerate complete fibres and report solution counts, fibre
sizes, collision entropy, and whether the apparent gain survives as width
increases. A useful result must reproduce the known triangular frame as a
positive control and remain exact when converted back to the full compression
equations.

## Decision rule

- If the composite fibre grows by one word with width, the reformulation has
  only moved an unresolved condition and does not improve the modelled cost.
- If it stays bounded or admits a width-independent inverse, derive and verify
  the identity at 32 bits before considering a GPU experiment.
