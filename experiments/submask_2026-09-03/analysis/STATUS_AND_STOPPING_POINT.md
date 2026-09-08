# Status and stopping point, 2026-09-08

A summary of where the 21-round question stands, why the search was stopped,
and what remains open.  Written so that anyone returning to this project (the
author, a referee, or a later reader) does not have to reconstruct it from
eight separate findings files.

## The three things that must be kept separate

**1. The results are finished, not stuck.**  19 rounds costs about 12 ms of one
CPU core after a reusable table precomputation; 20 rounds costs `2^45.4` swept
`a0`, and two computed and verified 20-round preimages exist, one of them of
the all-ones digest that Zaikin inverted at 19 rounds.  Nothing in the negative
results below touches any of that.

**2. The barrier itself became a result.**  A referee will ask why the work
stops at 20 rounds.  Before this week the answer was a dependency argument.  It
is now a single named edge, measured across about 1.6 million legal context
conditions at full width and 105 million reduced-width assignments, with every
alternative technique priced.  That is the difference between a paper that
stops and a paper that concludes, and it is recorded in
`WHAT_WOULD_BE_NEEDED.md` and in the barrier paragraph of
`paper_submask_r20.tex` §7.

**3. Only the push to 21 rounds is dead**, and it is dead in a form precise
enough for someone else to attack.

## Why the search was stopped

Eight independent searches have now returned zero bits:
`elimination_search/` (seven angles), `falsify_a5/` (five condition classes),
`sat_hybrid/` (SAT baseline, algebra/SAT hybrid, a/e duality),
`lattice_3d/` (multi-dimensional tables, symmetry, carry structure), and
`differential_wdomain/` (MITM ladder, differential characteristics, second
preimages from our own witnesses, the message-word domain).

The reason to trust this set more than the June 2026 verdict (which claimed the
20-round barrier was proven, and which the author correctly overruled) is that
these searches carry **positive controls**:

* the carry-structure scan rediscovers the known `(3/4)^32` collapse at exactly
  `(3/2)^w` at three reduced widths, and finds nothing near `C3`;
* the meet-in-the-middle model reproduces the literature's known failure point
  (chunk separation dies at 24 rounds, exactly where Isobe-Shibutani had to
  switch technique) before pricing 21 rounds at `2^160`, 83 bits too high;
* the frame searches reproduce the a-domain absorber diagonal as a control
  before reporting zero absorbable cells elsewhere;
* the planted-preimage tests recover every plant whose table roots are stored.

A search that can find what is already known, and still finds nothing new, is
much stronger evidence than one that merely returns empty.  In June there was
no such control, which is precisely why that verdict was wrong.

## What "dead end" does and does not mean

It means: no cheap idea remains inside the family of *value-based table
absorption* attacks, and the requirements in `WHAT_WOULD_BE_NEEDED.md` (R1--R5)
are the filter any new idea must pass first.

It does not mean: a proof about SHA-256, or about techniques nobody has
invented.  Every negative here is scoped to this construction and the
conditions tested, and every paper says so.

## Where the remaining value is

Not in more searching.  In order:

1. **The two ePrint submissions** (111557 and 111558, pending as of
   2026-09-08) and, after them, a peer-reviewed venue for the second paper.
   That is what turns the result into a credential.
2. **A reply from any of the four researchers emailed on 2026-09-06** (Zaikin,
   Davydov, Bright, the Guo group).  A confirmation from Zaikin, whose target
   was inverted and whose stated open problem was closed, is the single most
   valuable outcome available.
3. **The author's inference-systems work** (vLLM, Dynamo), where one pull
   request is a rebase from landing and three operators have independently
   validated it on their own hardware.

## If work on 21 rounds resumes

Leave it dormant rather than pursue it.  The trigger for reopening is an idea
that can say **which of the two value filters it removes and how**, given that
no fourth unknown can be absorbed while the offset law holds.  An idea that
cannot answer that will terminate where the previous eight searches did, and an
afternoon is enough to check the answer before spending anything.
