# Record state, 2026-09-05

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

## The second paper
`paper_submask_r20.tex` / `.pdf` (14 pp) is committed at the repository root
with `SUBMISSION.md`. Two independent fact-check passes were applied before
filing; their working files lived in the session scratchpad and are not part of
the record. The paper's evidence map is its Appendix B. Nothing is in progress.
