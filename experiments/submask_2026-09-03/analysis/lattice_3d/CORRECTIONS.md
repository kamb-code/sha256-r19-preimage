# Corrections to the raw explorer reports in this directory

The synthesis agent re-derived four of the load-bearing numbers independently
rather than take them on trust, and found two claims in the raw reports
(`reports.json`) that must not be quoted as written.  Both are recorded here
rather than edited out, so the record shows what was claimed and what survived.

## 1. The round-constant Hamming weight claim (symmetry study)

**As written:** "the round constants `K_r` have mean Hamming weight exactly
16.0 of 32".

**Correct:** that is true only of the **20** constants used at 20 rounds.  Over
all 64 constants the mean is **15.52**.  A referee with a calculator can refute
the sentence as written.

The contrast that sentence was drawing, against Keccak's round constants (mean
3.33 of 64, on one lane of 25), is unaffected, and so is the conclusion: dense
constants make rotational cost `2^-207` over 20 rounds.

## 2. The table-amortisation payoff at 20 rounds (multi-dimensional table study)

**As written:** a headline figure of `2^14.4` for the 20-round cost under
perfect reuse of a signature fibre.

**Correct:** that is the cost per preimage only if all `2^18.6` preimages of one
digest are harvested from a single fibre.  For **one** preimage of **one**
prescribed digest the mandatory sweep is a hard floor:

    cost = 2^32 + 2^13.38 * X

so the payoff at 20 rounds under perfect fibre reuse is **13.4 bits, not 31**.

The 21-round figure in that report (`2^45.4` at `X = 1`) stands, because there
the `2^45.38 * X` term dominates the `2^32` floor.  The practical consequence:
the idea is worth something only at 21 rounds, and it does not reach the ~30
bits needed there either.

## What was independently confirmed

* Carry-free set of `sigma0` at 32 bits: an independent exhaustive `2^32` scan
  gives **exactly 62,129**, matching the lattice study to the unit.
* The ratio 2.30 to 2.38 that appears in all twelve measured cells is
  analytically `sqrt(4*pi*e)/2.5 = 2.3378`, i.e. a property of the
  0/1-with-carry encoding rather than of any width or subsystem.
* Word-level linearity: an independent exhaustive minimum-arc computation over
  all multipliers gives 239/256 and 4040/4096, against the study's 238 and
  4039 (an off-by-one arc convention, identical conclusion: no small root
  exists, so no Coppersmith or bounded-distance-decoding shape exists).
* Constants audit: 0 of 64 constants rotation-invariant, 0 of 8 initial-value
  words, 0 complemented, 0 adjacent-equal, 20 distinct constants at 20 rounds,
  ceiling `log2(32 * 2 * 20) = 10.32` bits.

## Cross-corroboration between studies that did not share code

* Two studies derived the two-fibre amortisation cost independently and agree:
  `2^60.4` (elimination search) against `2^60.88` (this study).
* Three independent measurements of context-word avalanche agree: 10.6 to 16.1,
  10.9 to 16.1, and 11.28 to 16.01 bits per flipped bit.

That agreement is what makes the negative trustworthy; it is the same standard
applied elsewhere in this project through positive controls.
