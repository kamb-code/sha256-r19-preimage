# What a technique would have to do to pass 20 rounds

Six searches have now returned nothing (`elimination_search/`, `falsify_a5/`,
`sat_hybrid/`, and the lattice study).  Taken together they stop being a list
of failures and start being a specification: they say what any successful
technique must have, because they say precisely what every failed one lacked.
This note states that specification.  It is scoped to the attack family of
`paper_submask_r20.tex` and to the conditions tested; it is not a claim about
SHA-256 in general.

## The five structural facts

1. **The offset law.**  Every state word `a_k` re-enters the recovered message
   word `W_{k+5}` through `Sigma0(a_k)`, and again through `Sigma1(a_k + c)`.
   Both are *one-input* functions of the unknown itself.  The whole technique of
   this attack is saturating a three-input function (`Ch` with an all-ones
   selector, `Maj` with two equal inputs) so that it ignores an unknown; a
   one-input function has no second argument to fix, so the only way to make it
   harmless is to know the word already.

2. **Forced homes.**  In every one of 11,086 legal context configurations,
   `a1` can be recovered only by `C0`, `a2` only by `C1`, `a3` only by `C2`.
   A fourth absorbed word must therefore be light in all three and absorbable
   by `C3`, and fact 1 puts `Sigma0` of it heavily into at least one of them
   for every candidate `k = 4..11`.

3. **The obstruction is one edge.**  With `a6 = ~a4`, `a7 = a6`, `e8 = -1` the
   word `a5` is *exactly* absent from `C0` and `C2`, and with `e11 = 0` exactly
   linear in `C3`.  One path survives: `a5 -> C1` through `-Sigma0(a5)` inside
   `e6` in `W10`.  Across about 1.6 million legal context conditions at full
   width and 105 million reduced-width assignments, that edge never fell below
   **10.86 bits** per flipped bit, against the **2 bits** absorption needs.

4. **What made the one collapse work.**  The consistency residual became a
   bitwise condition of probability `(3/4)^32` because it was built only from
   bitwise atoms: `eps = Maj(v,a3,a2) - a3`, identity and `Maj`, and the
   difference is borrow-free.  A residual's zero set factors into independent
   per-bit conditions only if it is built from bitwise atoms.  The fourth
   residual carries five rotate-and-XOR atoms of the unknowns, so it cannot
   factor.  An exhaustive scan confirms this with a working positive control:
   the scan rediscovers the known collapse at exactly `(3/2)^w`, and finds
   nothing anywhere near `C3`.

5. **Ceilings on the alternatives.**  Tables: the entire per-context
   computation depends on the digest and all eight context words through only
   **five** 32-bit constants, so a universal table needs at least five axes; a
   three-axis table at full resolution answers "yes" in every cell.  Symmetry:
   a symmetry saves bits only by quotienting a stabilised target, and the
   available groups (bit rotation 32, complementation 2, round translation 20)
   cap the entire direction at **10.3 bits** against the ~30 that 21 rounds
   needs.  Search: on the identical instance our attack solves in ~12 ms,
   CaDiCaL and kissat find nothing in 1800 s.

## The specification

A technique that beats 20 rounds in this framework must satisfy **all** of:

* **(R1)** It must not require inverting `Sigma0` or `sigma0` *of an unknown*
  jointly with modular addition.  Every blocked path in every search reduces to
  exactly that operation.
* **(R2)** If it beats a 32-bit filter by factorisation, its residual must be
  built from bitwise atoms alone (fact 4).  Otherwise it needs a mechanism
  that is not per-bit factorisation at all.
* **(R3)** It cannot be a precomputed table on the fourth constraint: that
  constraint is a *membership* question, not an inversion, and needs five axes.
* **(R4)** It cannot draw its advantage from symmetry (10.3-bit ceiling).
* **(R5)** It cannot draw it from general search: a clause learner is five
  orders of magnitude worse on the identical instance, and imposing our
  structure as clauses makes it worse still.

## Correction: where the barrier actually sits

An earlier revision of this note said the attack is blocked by having to invert
`Sigma0`/`sigma0` of an unknown.  That is wrong as stated, and the differential
study (`differential_wdomain/`) found the error.  The 16 GB table inverts
`sigma0(u) - u` in **one lookup**; inversion is not the cost.  The cost is two
pure **value filters**: the collapsed consistency condition (`(3/4)^32`) and the
fourth schedule constraint (`2^-32`), together `2^45.4` at 20 rounds.  The offset
law matters because it is what stops a fourth unknown being absorbed, which is
the only way to remove the `2^-32`.  The distinction is not pedantic: it is
exactly why the direction this note previously recommended does not work.

## The two gaps of the previous revision, now closed

**Difference-based techniques buy nothing, for a reason the earlier note missed.**
The complementary-difficulty observation below is arithmetically true and
strategically irrelevant: a differential relates a *pair* and imposes no
condition on an output *value*, while a preimage fixes the value and has no
pair.  Three independent measurements agree.

* Chunk-separated meet-in-the-middle has an exact cost ladder here, not an
  extrapolation: `2^96` at 19 rounds, `2^128` at 20, `2^160` at 21, reaching
  `2^256` at 24 where chunk separation dies (which is where Isobe-Shibutani had
  to switch technique, the model's own control).  At 21 rounds that is **83 bits
  worse** than this attack's `2^77`.  The cap is structural: the message
  schedule couples 13 of 16 words, leaving neutral sets `{6,7,8}` at 21 rounds
  (confirmed: perturbing those three changed `W16..W20` in 0 of 3,000 trials),
  and beating `2^77` would need at least 5.6 disjoint neutral words per side.
  Published biclique dimension on SHA-256 is 1 to 3 bits, so amplifiers cannot
  close it.
* The best characteristic reachable in the frame where the linear maps are free
  costs `2^-2889` at 21 rounds against a `2^-77` bar, because the charged
  modular additions outnumber the free linear maps 3 to 1 (162 against 52 per
  21 rounds).  Modular addition costs exactly the Hamming weight of the
  difference below the top bit; `Ch` and `Maj` cost a bit at six of eight input
  patterns.
* Applied directly to the five witnesses this project owns, **13,981,848**
  nonzero message differences produced **0** second preimages (95% bound
  `p <= 2^-22.2`).  With a fixed initial value, any preimage-to-preimage
  difference is exactly a fixed-IV collision.
* Rebound attacks give 0 by construction, for the same reason: a characteristic
  constrains a pair's difference and says nothing about the output value.

**The message-word domain is not a second frame at all.**  The map from the 16
message words to the state words is a prefix-triangular bijection (verified
300/300 in both directions), so fixing a message-word prefix *is* fixing a
state-word prefix, and the counting is identical rather than merely comparable.
Where the two domains differ, the message domain is strictly worse: the
absorber exists because `a_k` enters `W_k` with `+1` and `W_{k+8}` with `-1`,
forming the `sigma0(u) - u` atom the table inverts, whereas `W_i` enters
`W_{i+7}` and `W_{i+16}` both with `+1`, so no such atom can form.  Measured: 0
of 128 exact `+/-1` cells at 20 and 21 rounds, minimum edge weight 7.69 bits,
and 0 absorbable cells at four reduced widths with the state-domain diagonal
correctly reproduced as a positive control.  The only message-domain absorbers
are the last few message words, which the backward chain already peels free,
and they run out at 20 rounds.

## The observation itself, kept for the record

`Sigma0` and `sigma0` are GF(2)-linear, so an exclusive-or difference passes
through them exactly and for free (0 violations in 1,048,576 trials each),
while modular addition destroys such differences.  The profiles really are
complementary.  What the measurements above establish is that this buys nothing
for a *preimage*, because the technique whose difficulty is arranged the other
way round is a technique about pairs, and a preimage is about a value.

## How to use this

For a referee or a reader, this is the honest statement of scope: the barrier
is sharp, single-edged and measured, and every named alternative now has a
number against it rather than an absence of evidence.  For anyone continuing
the work, R1--R5 are the filter to apply to a new idea *before* spending
compute: a proposal must say which of the two `2^-32` value filters it removes
and how, given that no fourth unknown can be absorbed while the offset law
holds.  Six searches have now terminated at that same point.
