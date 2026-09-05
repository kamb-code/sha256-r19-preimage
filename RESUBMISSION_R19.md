# Filing package — Cryptology ePrint Archive (corrected resubmission of the first paper)

PDF to upload: `paper_r19_final.pdf` (A4, 19 pages, author e-mail on page 1).

**Title**

A Practical Preimage Attack on the 19-Round SHA-256 Compression Function via a Global sigma0-Difference Table

(the title field is plain text: write "sigma0" or "σ0", not LaTeX)

**Authors (paper order)**

Kameldip Singh Basra — Independent Researcher — kameldipbasra@gmail.com — ORCID 0009-0001-0977-4614

**Category**

Attacks and cryptanalysis

**Keywords** (comma-separated, no LaTeX, each under 40 characters)

SHA-256, preimage attack, step-reduced, compression function, message schedule, table lookup, GPU cryptanalysis, hash function cryptanalysis

**Publication status** (public field)

Preprint. Published nowhere else; no DOI. A companion note and a second paper in the same repository are cited as manuscripts.

**Licence**

CC BY 4.0

**Message to the editors** (private field, not displayed)

This is a corrected resubmission of submission 110082 ("Oracle-Free Preimage Attack on 19-Round Reduced SHA-256", not accepted). What changed: the false lemma is replaced by a proposition in its correct provisional form; the near-collision "structure" claim is withdrawn to an appendix as chance; Zaikin's 19-round priority (Constraints, January 2026) is stated in the abstract and introduction and his target is inverted as a seventh preimage; every citation was resolved against the publisher and unverifiable ones removed; the "oracle-free" framing is gone; the screening lift is labelled as trajectory-counted with its deduplication bound; the cost figure is scoped to the pooled campaign rate with the control arm disclosed; and every campaign, screening and preimage number is traceable to a data file in the public repository named in the Artefacts paragraph (https://github.com/kamb-code/sha256-r19-preimage, tag r19-resubmission); the E1/E2 CPU pilots and the 165-context uniformity run are reported from notes and are labelled as pilots in the paper. A second paper on the same repository supersedes this paper's cost figures and is filed separately.

**Abstract** (plain text; `$…$` math only, no citations)

We present a practical preimage attack on the compression function of
SHA-256 reduced to 19 of its 64 rounds, in the standard setting: the
chaining value is the fixed initial value, the target is an arbitrary
256-bit digest, and the attacker chooses the 512-bit message block. The
attack combines a backward chain that fixes eight state words from the
target, a precomputed $16$ GB table inverting $u\mapsto\sigma_0(u)-u$, and
a two-stage fixed-point iteration over the three schedule-consistency
constraints. Each attempt sweeps one $32$-bit variable, costs $O(2^{32})$
table lookups, and runs in about $25$ s on one NVIDIA H100 with a single seed chain ($89$ s with the four chains used in the campaign). Seven preimages were found and independently verified. Six are of digests of random messages: three from dedicated runs that succeeded in their 4th, 21st and 35th contexts, and three from a preregistered campaign of 240 attempts that gives a direct estimate of the per-attempt success rate, $\hat\mu = 3/240 =
1.25\times10^{-2}$ (95% Poisson interval $[2.6\times10^{-3}, 
3.7\times10^{-2}]$; all three events fell in the screened arm and the
control arm gave $0/120$). Attempts are far from interchangeable: in a
paired campaign with matched exposure, a screening pass costing $1/4096$ of
an attempt raised the yield of the intermediate 16-bit filter by $3.41\times$ (95% CI $[2.13, 5.82]$, target-blocked bootstrap; counted per converged trajectory, with identity-level deduplication bounding it within $[2.42\times, 3.44\times]$), helping on 31 of 40 targets and hurting on 9; its effect on the final success rate is not established.

SAT-based cryptanalysis inverts 17 and 18 rounds in seconds on one core,
and Zaikin has recently inverted the 19-round compression function for a
fixed all-ones digest with a parameterized Cube-and-Conquer solver, in
$18.5$ hours on 192 CPU cores. Priority at 19 rounds is his. The
contribution here is a different route to the same round count, algebraic
rather than search-based, that reaches an arbitrary digest in about two hours of one GPU at the pooled campaign rate (one attempt in eighty; the unscreened control arm gave $0/120$, so the figure for a uniformly random context is nearer three hours), did so for his target in 84 attempts ($123$ minutes; the seventh preimage, and a second preimage of that digest) after a first run of 14 attempts had produced a match that a since-fixed bug discarded, and comes with complete artefacts. A companion note explains why the fourth schedule constraint costs this construction a further factor of $2^{32}$ at 20 rounds; a note added at the end points to a later chosen-context variant that pays that factor and reaches 20 rounds. Full SHA-256 is not affected by any of
these results.
