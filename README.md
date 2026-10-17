# Reduced-round SHA-256 preimages

Kameldip Singh Basra (ORCID 0009-0001-0977-4614), 2026.

Papers, explicit witnesses and reproducible verification for 19- and 20-round
SHA-256 compression preimages, including a 20-round preimage with standard
padding for a prescribed structured digest. Full 64-round SHA-256 is unaffected.

## Padded 20-round result — 17 October 2026

The new paper reports a verified 55-byte message matching a prescribed structured
digest under 20-round SHA-256, with the standard IV, schedule, constants,
feed-forward and length padding.

- [Read the paper (PDF)](padded_r20/paper_padded_r20.pdf)
- [Editable LaTeX](padded_r20/paper_padded_r20.tex)
- [Verification package and evidence](padded_r20/README.md)
- [Citation](padded_r20/CITATION.cff)

Run the complete verification from the repository root:

```bash
python3 padded_r20/verification/verify_release.py
```

The padded all-ones route is formulated but **has not been run**. Its conditional
`2^40` padding-context estimate is not a measured total preimage cost.
The earlier all-ones witness below is an **unpadded compression block**.
The new package reproduces witness verification, not the full GPU search.

## Earlier compression-function manuscripts

Three manuscripts on preimages of the SHA-256 compression function reduced to
19 and 20 of its 64 rounds, in the standard setting: fixed initial value,
feed-forward retained, all 256 output bits fixed, one unrestricted 512-bit
block, no padding constraint. Nothing here affects full SHA-256.

| file | what it is |
|---|---|
| `paper_submask_r20.pdf` / `.tex` | Context shaping: 19 rounds in milliseconds after a one-time table precomputation, and two computed 20-round preimages, one of the all-ones digest. Supersedes the cost figures of the other two. |
| `paper_r19_final.pdf` / `.tex` | The original 19-round table-based attack with random contexts and seven verified preimages. |
| `paper_r20_barrier.pdf` / `.tex` | Companion note on the 20-round barrier for the random-context construction. Historical; not an impossibility result. |

## Verify the earlier compression witnesses

`code/verify_r19.py` is a standalone reference implementation (Python 3,
standard library only, no shared code with any solver). The 20-round preimage
of the all-ones digest:

```bash
python3 code/verify_r19.py --rounds 20 \
  --hash ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff \
  --words "0b0187bf b9aea692 b66effa5 087dca3a 0caea827 1d2f9916 739a224e a87c0eae 7f9ef4a7 b318a7de a8848c61 a7a141a4 11ac114b 952299be 97aaa67f c6acfe57"
```

prints `result: OK`; with `--rounds 19` or `--rounds 21` it prints `FAIL`.
The second 20-round witness is in Appendix A of the first paper, and the
seven 19-round witnesses are in `verified_preimages.txt`.

## Citation

Cite the manuscript relevant to the result you use:

- **Padded 20-round result:** [paper-specific citation](padded_r20/CITATION.cff).
- **Context shaping and earlier compression results:** [repository citation](CITATION.cff).

The main citation file lists all four manuscripts. No ePrint report number or DOI
is assigned to these manuscripts.
