# Reduced-round SHA-256 compression-function preimages

Kameldip Singh Basra (ORCID 0009-0001-0977-4614), 2026.

Three manuscripts on preimages of the SHA-256 compression function reduced to
19 and 20 of its 64 rounds, in the standard setting: fixed initial value,
feed-forward retained, all 256 output bits fixed, one unrestricted 512-bit
block, no padding constraint. Nothing here affects full SHA-256.

| file | what it is |
|---|---|
| `paper_submask_r20.pdf` / `.tex` | Context shaping: 19 rounds in milliseconds after a one-time table precomputation, and two computed 20-round preimages, one of the all-ones digest. Supersedes the cost figures of the other two. |
| `paper_r19_final.pdf` / `.tex` | The original 19-round table-based attack with random contexts and seven verified preimages. |
| `paper_r20_barrier.pdf` / `.tex` | Companion note on the 20-round barrier for the random-context construction. Historical; not an impossibility result. |

## Verify the witnesses in one second

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

See `CITATION.cff`. The manuscripts have no ePrint report number or DOI.
