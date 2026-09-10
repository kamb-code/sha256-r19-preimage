# Manuscript Record: Context-Shaped R19/R20 Paper

## Current manuscript

- **Title:** Context Shaping for 19- and 20-Round SHA-256 Compression Preimages
- **PDF:** `paper_submask_r20.pdf`
- **Source:** `paper_submask_r20.tex`
- **Author:** Kameldip Singh Basra
- **Status:** Unpublished revised manuscript; no DOI or ePrint report number

The earlier version was submitted to the Cryptology ePrint Archive as
submission 111558 and was declined. The editor cited the archive's general
criteria but did not identify a paper-specific technical error. The current
manuscript is a subsequent revision and has not been resubmitted.

## Scope

The paper concerns one-block SHA-256 compression reduced to 19 or 20 rounds,
with the standard initial value, feed-forward retained, all 256 target output
bits fixed, and an unrestricted 512-bit input block. It does not claim a
preimage of padded SHA-256 and does not affect full 64-round SHA-256.

## Abstract

We study one-block SHA-256 compression reduced to 19 or 20 rounds, with the
standard initial value, feed-forward retained, all 256 output bits fixed, and
an unrestricted 512-bit input block. A global table for
`u -> sigma0(u) - u` underlies an earlier 19-round construction. We show
that choosing internal context words so that `a4 = a5` and
`e8 = e9 = 0xFFFFFFFF` makes its three schedule constraints triangular:
three table lookups replace a fixed-point iteration. The same conditions
reduce the remaining consistency test to `Maj(a4,a3,a2) = a3`, which holds
for exactly `3^32` of the `2^64` pairs `(a2,a3)`. Under an explicit
lookup-distribution model this predicts one 19-round preimage per 39,124
swept values; three recorded runs give one per 38,878. A 64-target run
produced 13,840 forward-verified preimages in 151.7 seconds. The often-quoted
12 ms figure is a marginal estimate obtained from measured single-core
throughput and yield after a reusable 16-GiB table has been built, not a
cold-start latency.

At 20 rounds, candidates must also satisfy an exact fourth schedule
constraint. Existing residual tests are consistent with the working model
that this condition contributes a factor `2^-32`; they do not prove
uniformity at exact zero or rule out all alternative policies. With three
stored roots per table value, the model predicts a mean cost of about
`2^45.4` swept values, or about 15 hours at the measured A100 production
rate. We report two computed, forward-verified examples: one for a
solver-generated target after 2.45 GPU-hours and one for the fixed all-ones
target after 16.16 GPU-hours. The six recorded production runs total about
71.7 GPU-hours. The method uses known choice- and majority-function
degeneracies in a new combination. Negative 21-round observations apply only
to the tested construction and establish no lower bound.

## Evidence qualifications

- The two successful R20 runtimes are observations, not estimates of a mean.
- The approximately 15-hour R20 value is conditional on the residual model.
- The exact-zero R20 rate was not measured directly; useful occupancy reaches
  24 residual bits.
- R21 cost figures are conditional estimates for the tested construction.
- The literature-priority statement is dated and scoped, not a proof.
- Automated verifier passes are described according to what each pass checked;
  they are not human peer review.

## Publication checklist

- Confirm that the PDF hash matches the revised TeX source.
- Forward-check both R20 witnesses with `code/verify_r19.py`.
- Check all references and links against the final PDF.
- Use the title and abstract above for any future filing.
- Do not assign or advertise an ePrint identifier unless the archive issues
  one.
