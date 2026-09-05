# Filing package — Cryptology ePrint Archive

Everything the ePrint submit form asks for, ready to paste. The PDF to upload is
`paper_submask_r20.pdf` (A4, 14 pages, author e-mail on page 1).

**Title**

Context Shaping for Reduced-Round SHA-256 Compression-Function Preimages:
Nineteen Rounds in Milliseconds, and Computed Preimages at Twenty Rounds

**Authors (paper order)**

Kameldip Singh Basra — Independent Researcher — kameldipbasra@gmail.com —
ORCID 0009-0001-0977-4614

**Category**

Attacks and cryptanalysis

**Keywords** (comma-separated, no LaTeX, each under 40 characters)

SHA-256, preimage attack, reduced-round, compression function, hash function
cryptanalysis, table-based attack, GPU cryptanalysis, practical attack

**Publication status** (public field)

Published nowhere else; no DOI.

**Message to the editors** (private field, not displayed)

This is a new paper, not a revision of my earlier submission (submission
110082, "Oracle-Free Preimage Attack on 19-Round Reduced SHA-256", not
accepted). It cites that manuscript as unpublished, supersedes its cost
figures, and reports new results (computed 20-round preimages). Every citation
was checked against its source and every number is traceable to a committed
log in the public repository.

**Licence**

CC BY 4.0

**Abstract** (plain text, `$…$` math only, no citations)

The table-based preimage attack of our earlier manuscript on the 19-round
SHA-256 compression function chooses seven internal state words, the context,
at random, and then pays twice: a fixed-point iteration that converges for
about one $C_0$ survivor in $2^{16}$, and an exact 32-bit consistency check on
a provisional value of the message word $W_9$. We show that both costs vanish
on a large subfamily of contexts. Choosing the context so that $a_4=a_5$ and
two derived words satisfy $e_8=e_9=$ 0xFFFFFFFF makes the three schedule
constraints solve triangularly in three table lookups per attempt, with no
iteration, and collapses the consistency residual to $\mathrm{Maj}(a_4,a_3,a_2)-a_3$
modulo $2^{32}$, whose vanishing is a bitwise condition of probability exactly
$(3/4)^{32}$ rather than $2^{-32}$. The measured cost of a 19-round preimage
falls from about $2^{38.3}$ swept values of $a_0$ to about $2^{15.25}$, within
one per cent of the derived $39{,}124$: about 12 milliseconds of one CPU core.
We exhibit 13,840 verified preimages produced in 152 seconds, and 108
preimages of the all-ones digest that Zaikin inverted at 19 rounds in 18 hours
33 minutes on 192 cores.

At 20 rounds the fourth schedule constraint remains an exact 32-bit filter. We
measure its residual to be uniform to 24 bits and find, by a
characteristic-function analysis with a validated positive control, no bias
that a table representative policy or a context choice could exploit: the first-harmonic contribution is below $10^{-3}$ bits at our
sample size. The cost is therefore $2^{45.4}$ swept $a_0$ with three-root
tables, about 15 hours on one NVIDIA A100. We report two computed and verified 20-round preimages, one for the
all-ones digest, found after 16 hours on one of two A100s searching it (about 26 US dollars on
that GPU, 52 for the pair, against an expected 15 hours and 24 dollars per
preimage). To
our knowledge, and after a search of the IACR ePrint archive, dblp, OpenAlex,
arXiv and the SAT, CP and AI venues as of September 2026, no computed preimage
of the SHA-256 compression function beyond 19 rounds has been published. We
also show that, for this construction, 21 rounds costs the full additional
$2^{64}$: the fourth and fifth constraints are independent, and the dependency
that blocks a further collapse passes through $\Sigma_0$ of the unknown itself,
which no context word can saturate. Neither lever is new on its own; the
contribution is their combination and the bitwise collapse. Nothing here
affects full SHA-256.

---

## Before you press submit

* The two witnesses can be checked by a referee in under a second with the
  commands in Appendix A of the paper.
* All code, logs, reviewer artefacts and the manifest are public at
  https://github.com/kamb-code/sha256-r19-preimage
* The earlier manuscript is cited as a manuscript, not as an ePrint report,
  because it is not on the archive.
* Decide whether to keep the acknowledgement of AI assistance. It is accurate
  as written; several venues now require such a statement, and ePrint does not
  forbid one.
