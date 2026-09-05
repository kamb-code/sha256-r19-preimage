# Working state, 2026-09-05 — resume point if the session is lost

## Done and committed
* Two verified 20-round preimages: `R20_PREIMAGE_ALLONES.txt` (all-ones digest,
  Zaikin's target, pod oxgcuwlleea1fd, seed 3971886063) and `R20_PREIMAGE.txt`
  (solver-generated digest). Both verify with
  `code/verify_r19.py --rounds 20`, fail at 19 and 21.
* All GPU pods deleted; nothing billing. Total campaign about $180.
* `analysis/` corpus mining (no lever exists), `priority/` literature search
  (no computed preimage beyond 19 rounds), `reviewers/` two verification rounds.
* All-ones obstruction test FINISHED and archived as
  `logs/ones_obstruction.log` + `scripts/ones_obstruction.py`. Final: all-ones
  3,948,132 candidates, control 3,954,274. No deficit at any depth; a surplus
  at 18–20 bits (k=20: 11 observed vs 3.77) that does not persist to 22+
  (1 vs 0.94) and is a 6-look expanding-window statistic, so noise. The target
  was never obstructed, which the preimage independently settles.

## In progress: the second paper
`/home/administrator/sha/publish/paper_submask_r20.tex` — drafted, compiles
clean (13 pp, 0 overfull after fixes). NOT yet committed.

A 9-agent fact-check (6 lenses + 3 refuters) produced 8 blocking and 46 major
findings, saved in full at
`/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/p2_findings.txt`
and `/home/administrator/.claude/projects/.../tool-results/bk8sgy7q2.txt`.

Refuter verdicts: the preimage/priority claim STANDS; the 19-round cost claim
STANDS; the Fourier claim was REFUTED as stated and must be weakened from
"prove ... no policy can bias it" to "find, with a validated positive control,
no bias, bounded below 1e-3 bits at our sample size".

### Fixes still to apply (grouped)
1. Citations: Guo et al. initials are J. Guo, H. Li, M. Liu, S. Wang, T. Zhang
   (LNCS 16546 pp. 121–151); Bouam first author is M. not S., published in
   Parallel Computing 106:102804; Bouam et al. is a 3XOR collision-type result
   on 128 bits, NOT a partial preimage; the 16–30-round SAT instances come from
   GraphBench (Stoll et al., arXiv:2512.04475), not from Nejati/Alamgir; the
   "two literatures do not cite each other" is false one way (Zaikin cites KRS).
2. Numbers: second all-ones pod is 12,075 contexts / 5.19e13 / 1.133; campaign
   total 3.48 and P(<=2)=0.32; predicted candidates per context 402,897 (not
   402,970), observed per-pod mean 402,880–402,910; Triton 9.1–9.3e8 kernel-timed
   and 8.6–9.0e8 production, ~3.6x not 3.59; 21 rounds is 2^77 with three roots
   (2^79 with one); unify the CPU per-preimage time; the base attack converges at
   ~2^-16 per survivor not 2^-18; "3.5e11 over 46 targets" matches no file.
3. Evidence gaps: Table 1's all-ones/all-zero 19-round rows and Table 4's frames
   have no committed log — RERUN and archive, or cite the reviewers' figures.
   The GPU halving statistic must be recomputed from the committed
   `gpu_run_snapshot/h100_status.json` (2,654 of 5,370, z=-0.85), not the
   uncommitted intermediate look. The all-ones bullet must use the now-final
   obstruction log.
4. Claims: no bare "first" anywhere (title included); every priority claim needs
   "computed and verified", "compression function", "to our knowledge", a date
   and the indices; 21-round result is a property of THIS family, not a bound;
   scope paragraph must not imply anything about full SHA-224; the "next
   experiment" quotation attributed to the frozen note DOES NOT EXIST there and
   must be removed (this is the fabricated-citation class that caused the
   earlier rejection); Guo et al. never say "absorption rule".
5. Format: `\hypersetup` for PDF metadata; `\path` for the overfull verifier
   path; ragged-right p-columns in Table 2; delete author TODO comments;
   create `SUBMISSION.md` with the ePrint metadata block.
