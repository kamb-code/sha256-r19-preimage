# A computed padded 20-round SHA-256 preimage

Kameldip Singh Basra · 17 October 2026

This release accompanies **A Computed Preimage of 20-Round SHA-256 with Standard Padding for a Prescribed Structured Digest**.

The result is a concrete 55-byte message matching all 256 bits of a digest fixed before the recorded search, with the standard SHA-256 IV, message schedule, constants, feed-forward, and length padding. Only the round count is reduced, to the first 20 rounds (0–19). The target was deliberately structured; this is not an arbitrary-target performance result. Full 64-round SHA-256 is unaffected.

- [Paper (PDF)](paper_padded_r20.pdf)
- [Editable LaTeX source](paper_padded_r20.tex)
- [Exact message, target and padded block](evidence/witness.json)
- [Replayed verification results](evidence/verification.json)

## Verify

Python 3.9 or later, standard library only; no GPU, tables, network access, or search software is needed.

```sh
python3 verification/verify_release.py --output verification_local.json
```

The command exits with an error if any check fails. Its output must report `PASS`. To compare it to the published deterministic result:

```sh
python3 -c "from pathlib import Path; import json; assert json.loads(Path('verification_local.json').read_text()) == json.loads(Path('evidence/verification.json').read_text())"
```

Checks include:

- Exact message length and standard padding, constructed from the message bytes.
- Two forward implementations agreeing at 19, 20, 21 and 64 rounds.
- Target matched at 20 rounds, and not at 19 or 21 rounds.
- Seventeen 64-round comparisons with `hashlib`, including this message and padding-boundary/multi-block controls.
- The complete reconstructed schedule, all four schedule residuals, and the disclosed state relations.
- Recorded counters and agreement of the final logged context with the witness's forward trace.
- Eleven supplied Wain 19-round padded vectors, Wain's 20-round unrestricted compression vector, and Basra's earlier unrestricted all-ones vector. These comparisons are witness checks, not new searches or performance benchmarks.

The unmodified `verification/verify_r19.py` is the earlier repository's standalone raw-block verifier. Its upstream source and license are recorded in `evidence/provenance.json` and `verification/UPSTREAM_LICENSE.txt`. It does not itself check padding; use `verify_release.py` for the complete message-hash check.

## Witness

Message (hexadecimal; concatenate lines):

```text
63823741194daa9bb857c160fe7a02c0
b400ea9d2f4c7f94a33eb2234bd3144e
1c4e9b5d350df3ebf5001d7211a1b83d
0886e96afa6b6b
```

20-round target:

```text
c799d8515f3907ec43324031d31ea13e9d2287d94cf0f4f741989d70463e45b0
```

Padded block:

```text
63823741 194daa9b b857c160 fe7a02c0
b400ea9d 2f4c7f94 a33eb223 4bd3144e
1c4e9b5d 350df3eb f5001d72 11a1b83d
0886e96a fa6b6b80 00000000 000001b8
```

For a 55-byte message the padding is one `80` byte followed immediately by the eight-byte length field `00000000000001b8`. There are no intervening zero-fill bytes.

## Recorded run

One NVIDIA A100-SXM4-80GB run recorded 6,947 complete context sweeps (6,944 distinct contexts), 29,837,137,805,312 swept early-word values and one reconstructed match. The driver's elapsed search time was 33,474.646 seconds, about 9.30 hours, excluding prior table construction/copy. Approximately US$16.79 is a cost estimate from the reported rental duration and rate, not an audited invoice or an expected cost for another target.

`evidence/context_counters.csv` contains all recorded context counters, including repeats. `run_summary.json` and `pilot_summary.json` preserve numerical measurements and explain their scope. A nominal residual model is labelled as an assumption; neither independent trials nor uniform residuals are proved.

## All-ones status

**A padded all-ones run has not been executed in this release. No padded all-ones message is claimed.** Section 6 of the paper explains the target-preserving formulation and its remaining conditions. The suggested `2^40` padding-context estimate is conditional on the joint residual distribution; it is not a measured total preimage cost. No week-long or dollar-cost forecast is asserted.

The earlier **unpadded** all-ones compression witness is included only as a comparison vector and is clearly identified as such.

## Reproduction scope

This package reproduces the witness verification and checks consistency of the published numerical records. It is not a complete GPU search-software release: the search kernel and all its dependencies are not bundled. No new candidate search was performed to prepare these verification results.

Files in `evidence/provenance.json` identify the original records by hash and cite the exact upstream verifier commit. Those hashes are integrity/provenance identifiers, not external timestamps. No private correspondence, cloud credentials, deployment scripts, or internal research handoffs are included.

## Build the paper

With a LaTeX installation providing `pdflatex`, `geometry`, `lmodern`, `microtype`, AMS packages, `booktabs`, `array`, `xurl`, `hyperref`, and `fancyvrb`:

```sh
pdflatex -interaction=nonstopmode -halt-on-error paper_padded_r20.tex
pdflatex -interaction=nonstopmode -halt-on-error paper_padded_r20.tex
```

Check the distributed files before rebuilding or editing them:

```sh
sha256sum -c SHA256SUMS
```

The checksum file excludes itself. Rebuilding a PDF can change its byte-level checksum even when the typeset contents agree.

## Citation and related work

See `CITATION.cff` and the paper's bibliography. This manuscript is a GitHub research release; no journal acceptance, DOI, or ePrint identifier is asserted.

- [Basra's context-shaping work](https://github.com/kamb-code/sha256-r19-preimage)
- [Wain's padded 19-round work](https://github.com/srw/sha256-19step-preimages-with-padding)
- [Wain's unpadded 20-round work](https://github.com/srw/sha256-20step-preimages-with-no-padding)
- [Zaikin's SAT-based compression preimages](https://doi.org/10.1007/s10601-025-09383-0)

The reference verifier retains its upstream MIT license. Numerical comparison vectors are attributed individually in `evidence/comparison_vectors.json`. The manuscript is copyright 2026 Kameldip Singh Basra; no change to the parent repository's licensing is made by this verification bundle.
