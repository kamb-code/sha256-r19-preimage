# Manuscript Record: Original R19 Paper

## Current manuscript

- **Title:** A Practical Preimage Attack on the 19-Round SHA-256 Compression Function via a Global sigma0-Difference Table
- **PDF:** `paper_r19_final.pdf`
- **Source:** `paper_r19_final.tex`
- **Author:** Kameldip Singh Basra
- **Status:** Unpublished revised manuscript; no DOI or ePrint report number

The earlier version was submitted to the Cryptology ePrint Archive as
submission 111557 and was declined. The editor cited the archive's general
criteria but did not identify a paper-specific technical error. The current
manuscript is a subsequent revision and has not been resubmitted.

## Relationship to the newer paper

This manuscript remains the full account of the original random-context,
fixed-point construction and its campaign. The separate manuscript
`paper_submask_r20.pdf` gives a later context-shaped construction, supersedes
this paper's performance figures, and reports two computed 20-round examples.
The two papers are intentionally not merged.

## Scope

The paper concerns the one-block 19-round SHA-256 compression function with
the standard initial value, feed-forward retained, all 256 target bits fixed,
and an unrestricted 512-bit input block. It does not claim a padded SHA-256
preimage and does not affect full 64-round SHA-256.

## Abstract

We present an algebraic candidate-generation method for SHA-256 compression
reduced to 19 of 64 rounds. The standard initial value and feed-forward are
retained, all 256 target bits are fixed, and the input is an unrestricted
512-bit block rather than a padded message. From the target, a backward chain
fixes eight state words. Each attempt chooses seven context words, sweeps one
32-bit state word, and enforces three message-schedule constraints with a
reusable 16-GiB table for `u -> sigma0(u) - u`. Two constraints form a
fixed-point iteration; an exact final test checks the provisional treatment
of `W9`. A complete sweep took about 25 seconds on one NVIDIA H100 with one
initial chain and about 89 seconds with the four chains used in the campaign.

We publish seven forward-verified examples: six for digests generated from
random messages and one for the fixed all-ones benchmark. Zaikin's SAT-based
19-round result on that benchmark predates this work. Three dedicated runs
succeeded in their 4th, 21st and 35th contexts. A preregistered stratified
campaign processed 240 full contexts: 3 of 120 selected by a cheap screening
rule produced a preimage, while 0 of 120 controls did. Pooling the arms gives
3/240, but does not estimate a uniformly sampled context. Screening raised
an intermediate low-16-bit counter by a trajectory-counted factor 3.41, with
identity-level bounds 2.42 to 3.44; its effect on final success remains
unresolved. With only three campaign events, the data do not determine a mean
runtime. The newer context-shaped paper supersedes these performance figures
and gives computed 20-round examples.

## Evidence qualifications

- The campaign reports screened and control strata; its pooled rate is not a
  uniformly sampled context rate.
- The population-weighted plug-in value is about 3.3 GPU-hours and is highly
  uncertain because only three final events were observed.
- The screening lift concerns a trajectory-counted intermediate event. The
  archived row aggregates bound its identity-deduplicated value, but no rerun
  measured the exact corrected lift.
- The campaign cannot be replayed byte for byte because its code commit,
  complete target identifiers, and random source messages were not retained.
  Its aggregate rows and all three witness blocks remain available.
- Zaikin has priority for the first computed 19-round full-output witness in
  this model. His one successful runtime and stated one-in-eight acceptance
  probability do not by themselves determine a repeated-run mean.

## Publication checklist

- Confirm that the PDF hash matches the revised TeX source.
- Forward-check all seven witnesses with `code/verify_r19.py`.
- Check all references and links against the final PDF.
- Use the title and abstract above for any future filing.
- Do not assign or advertise an ePrint identifier unless the archive issues
  one.
