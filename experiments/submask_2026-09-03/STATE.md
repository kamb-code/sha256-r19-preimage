# Evidence snapshot and publication status

## Results
* Two verified 20-round preimages: `R20_PREIMAGE_ALLONES.txt` (all-ones digest,
  Zaikin's target, pod oxgcuwlleea1fd, seed 3971886063) and `R20_PREIMAGE.txt`
  (solver-generated digest). Both verify with
  `code/verify_r19.py --rounds 20` and fail at 19 and 21.
* All GPU pods deleted; nothing billing. Total campaign about $180.
* `analysis/` corpus mining (no exploitable structure found), `priority/`
  literature search (no computed preimage beyond 19 rounds found, scoped as
  stated in `priority/ADJUDICATION.md`), `reviewers/` two verification rounds.
* All-ones obstruction test finished: `logs/ones_obstruction.log` with
  `scripts/ones_obstruction.py`. No deficit that deepens with depth in either
  arm; the all-ones arm runs high at 18–20 bits and returns to expectation at
  22. The target was never obstructed, which the preimage independently settles.

## Manuscripts
`paper_submask_r20.tex` / `.pdf` is committed at the repository root with
`SUBMISSION.md`; `paper_r19_final.tex` / `.pdf` is accompanied by
`RESUBMISSION_R19.md`. Automated checking passes preceded the original
filing; their archived reports record the checks each pass actually performed.
Both papers were submitted to ePrint on 2026-09-06 (submission IDs 111557 and
111558) and were later declined under the archive's general editorial
criteria. Those IDs are not citable report numbers. The repository PDFs were
subsequently revised and have no ePrint identifiers.
