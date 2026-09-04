# Pattern mining of 51k 19-round preimages and 69k 20-round candidates

Corpora built by `scripts/corpus.py` (51,268 verified 19-round preimages from
60 targets; 68,680 20-round candidates satisfying every constraint except the
fourth, from 40 targets, each with its fourth-constraint residual). Seven
independent analyses, then a refuter per claimed lever. **Fifteen agents, eight
claimed levers, all eight refuted.** Full reports in `reports.json`.

## The two results that are now proofs, not measurements

**1. The collapsed filter is information-theoretically tight.** The joint
entropy of the two constrained state words is 50.7189 bits against the predicted
32·log2(3) = 50.7188, and the surviving pairs are *uniform* on the 3^32 allowed
set: every bit position carries 1.58496 ± 0.00003 bits, and the 496 pairwise
mutual informations sum to 0.0010 bits. Nothing beyond the known condition is
constrained, so no sharper filter exists in that relation.

**2. No table representative policy can ever beat the fourth constraint.**
The residual is `c3 = sigma0(W4) + W3 - K` with the context entering *only* as
the additive constant K. A sum modulo 2^32 has a factorising characteristic
function, so `phi_c3(b) = phi_W3(b) * phi_sigma0(W4)(b)`. The stored-root policy
puts a large bias in W3 (|phi_W3(1)| = 0.155, a 255-sigma non-uniformity) and it
is annihilated because |phi_sigma0(W4)(1)| = 0.025 sits at its own noise floor.
Hence **even driving |phi_W3| to 1 caps the gain at 9.1e-4 bits**. Independently,
a policy storing roots on a density-2^-k subset raises the residual's structure
by k bits while cutting the lookup hit rate by exactly 2^-k: zero-sum. The same
factorisation shows "are some contexts cheaper" is identically the question "is
`W3 + sigma0(W4)` non-uniform", and it is not, under an 8.4-million-frequency
Fourier scan validated against a positive control.

## Negative results, with their power stated

* The residual is uniform and unpredictable from anything available **before**
  the three table lookups (which are the actual cost): 262,140 exhaustively
  scanned frequencies, 17.3M linear approximations, 15,360 bit-pair mutual
  informations. No predictor removes more than ~6e-4 bits. Aggregate 4-sigma
  upper limits on a spread deficit: 0.006 bits on the low byte, 0.096 on the low
  16 bits.
* Absolute yield audit: observed preimages / predicted = **0.9986 +- 0.0029**,
  so the entire yield is `coverage^3 x (3/4)^32` and any hidden gain lies in
  [-0.010, +0.006] bits.
* No context is better than another; no near-neighbour structure in the swept
  variable; the chain is injective at the birthday rate; no linear or affine
  structure beyond three carry-free least-significant-bit relations per context.
* **21 rounds:** on 214,583 freshly generated 21-round candidates the fourth and
  fifth constraint residuals are uniform and mutually independent (MI < 1e-5
  bits), so 21 rounds costs the full 2^64 above 19. An exhaustive symbolic search
  over 1,485 legal context families confirms the Sigma0 barrier is unremovable.

## Two facts about the table worth recording

* The image of `u -> sigma0(u) - u` is exactly 2,721,603,628 of 2^32 =
  0.633672724, **not** the random-map value 1 - 1/e = 0.632120559, and the
  root-multiplicity distribution is under-dispersed relative to Poisson(1). The
  hit rate also varies with the index residue class (chi2 = 7278 on 255 dof).
  Real non-random-map behaviour; worth under 1% and not steerable.
* The value 0x20000000 has exactly one root, u = 0xFFFFFFFF, which is
  bit-identical to the int32 MISS sentinel, so that one index always reads as a
  miss. Already documented in `code/build_sigma0_table.py`; costs 1 index in
  2^32. It must be mentioned in any exhaustiveness claim.

## Methodological warning for anyone reusing these corpora

`corpus.py` samples the swept variable **with replacement**, so 530 of 68,680
rows (and 205 of 51,268) are exact duplicates. A naive within-context collision
statistic on the residual then reads **+69 sigma, an apparent 6.64-bit entropy
deficit** -- a complete false positive that vanishes (-1.5 sigma) after
deduplicating on (context, swept value). Deduplicate first.
