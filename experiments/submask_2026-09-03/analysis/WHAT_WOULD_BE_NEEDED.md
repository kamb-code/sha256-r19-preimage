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

## What that leaves, and the one observation worth acting on

`Sigma0` and `sigma0` are **GF(2)-linear** (each is an exclusive-or of
rotations and shifts).  Our attack is blocked by them because it must invert
them *on values*, simultaneously with modular addition.  But an exclusive-or
difference passes through a GF(2)-linear map exactly and for free:
`Sigma0(x XOR d) = Sigma0(x) XOR Sigma0(d)`.

So the precise operation that blocks this attack is **transparent to a
difference-based technique**, and conversely what our table handles for free,
the modular additions, is what carries make hard for differences.  The two
approaches have complementary difficulty profiles.  That does not mean a
differential or rebound attack reaches 21 rounds; it means the barrier proved
here is a barrier to *value-based table absorption* specifically, and says
nothing against the one family of techniques whose difficulty is arranged the
other way round.  Nothing in this project has tested that family.

One further structural gap: every frame tried so far parameterises the attack
by state words (`a`-words, and the `e`-word duals, which fail because the
branch coupling is one-directional).  An attack parameterised in the
**message-word domain** has a different dependency graph and has not been
examined.

## How to use this

For a referee or a reader, this is the honest statement of scope: the barrier
is sharp, single-edged and measured, and it is a barrier to a named family of
techniques rather than to the problem.  For anyone continuing the work, R1--R5
are the filter to apply to a new idea *before* spending compute on it: if a
proposal does not say how it avoids inverting a one-input function of an
unknown, it will terminate where the previous six searches did.
