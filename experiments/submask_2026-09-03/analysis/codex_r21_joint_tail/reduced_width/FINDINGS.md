# Reduced-width test of the tail-lift frame

Date: 2026-09-10. Runs the open experiment specified in `../STATUS.md`.

## What the audit note proposed

Because sigma0 is a GF(2) bijection, the fourth and fifth schedule constraints
can be read as a sequential tail recovery for fixed preceding data:

    W4 = sigma0^{-1}(K3p - W3)      W5 = sigma0^{-1}(K4p - W4)

so `C3` and `C4` are not intrinsically uninvertible.  They degrade into
residual checks in the current construction only because `a4` and `a5`, hence
`W4` and `W5`, are already fixed as context before those equations are reached.
The open question was whether the context equations used by `C0`--`C2` can
accommodate the recovered values without introducing a new full-width
condition.

## The note's algebra is correct

Positive control, on genuine solutions of the w-bit model (`model.py`, which
reproduces real SHA-256 exactly at w = 32, R = 64, checked against `hashlib`):

| width | C_j identities | W4, W5 recovered | a4, a5 recovered |
|---|---|---|---|
| 6 | 200/200 | 200/200 | 200/200 |
| 8 | 200/200 | 200/200 | 200/200 |
| 10 | 200/200 | 200/200 | 200/200 |
| 12 | 200/200 | 200/200 | 200/200 |
| 16 | 200/200 | 200/200 | 200/200 |

The tail recovery is exact and triangular, exactly as the note says.

## The cycle is real, and asymmetric

Single-bit perturbation of `a4` and `a5`, 400 trials per cell:

| | depends on a4 | depends on a5 |
|---|---|---|
| K3p | 400/400 | ~270/400 |
| K4p | **0/400** | 400/400 |

So `K4p` is independent of `a4`, but `K3p` depends on both.  The recovered
values feed back into the constants that produce them.

## The measurement that settles it

Frame 2 replaces two residual checks with a consistency condition: the
tail-derived `(a4, a5)` must equal the pair the context assumed.  Measured
cost of that condition, against the `2w` bits a fresh full-width condition
would cost:

| w | -log2 P(a4) | -log2 P(a5) | -log2 P(both) | 2w | independence ratio |
|---|---|---|---|---|---|
| 5 | 5.01 | 5.00 | 9.98 | 10 | 1.02 |
| 6 | 5.97 | 5.99 | 12.00 | 12 | 0.97 |
| 8 | 7.97 | 8.01 | 15.87 | 16 | 1.07 |
| 9 | 9.11 | 8.99 | — | 18 | — |
| 10 | 9.96 | 10.10 | — | 20 | — |
| 12 | 11.91 | 12.02 | — | 24 | — |

Each recovered word costs exactly `w` bits, the two are independent, and the
joint cost tracks `2w` at every width tested.  (Dashes mark widths where the
joint event did not occur in the sample; the marginals are clean throughout
and independence is established at w = 5, 6, 8.)

## Verdict, against the note's own decision rule

The rule was: if the reformulation only moves an unresolved condition, it does
not improve the modelled cost.  It moves it.  Frame 2 converts two `2^-w`
residual checks into two `2^-w` consistency conditions, with the same total and
no detected structure in either.  At w = 32 that is `2^-64`, identical to the
frame it replaces.

The mathematical observation is nonetheless worth keeping: the fourth and fifth
constraints are invertible in isolation, and the barrier is specifically that
the words they would determine are needed earlier than the constraints that
determine them.  That is the offset law expressed in the tail rather than in
`C0`--`C2`.

## Methodological note

sigma0 is a GF(2) bijection at widths 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17,
18, 19, 23, 24, 25, 26, 27 and 32, and singular at 7, 11, 15, 20, 21, 22, 28,
29, 30 and 31.  The property holds at 32, which is the case that matters, but a
reduced-width model must avoid the singular widths; an experiment that happened
to pick w = 7 would have reported a spurious failure of the whole frame.

## Files

`model.py` (width-scaled model plus sigma0 inverse, with self-test),
`tail_lift.py` (positive control, cycle test, consistency measurement),
`scale.py` and `scale.log` (the scaling table above), `tail_lift.log`.
