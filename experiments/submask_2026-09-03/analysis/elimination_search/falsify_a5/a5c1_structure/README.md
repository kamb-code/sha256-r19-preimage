# a5 -> C1 edge at R = 20: extended condition menu + structure of the residual

Files (all CPU, no table needed):
- `a5edge.py`        vectorised extension of elim/a10/numcheck.py: same `defs` format plus
                     ('rot', j, n), ('rotneq', j, n), ('xor', j, c), ('add', j, c), ('hoff', t)
                     (e_i - a_{i-4} = t, e.g. e9 = a5 + t), arbitrary constants in sat_r/sat_h;
                     numerical legality (context invariant under the unknown); edge weights
                     for the four targets T_j and the exact C1 / C0 table indices.
                     `python3 a5edge.py` reproduces the prior numbers (a5: C0 0.99 / C1 12.04,
                     a4 -> C0 control 12.28).
- `stage1_sweep.py`  96,228 configurations of the W10-reaching knobs (a6 vs a4: eq/neq/31 rot/
                     31 rotneq/17 xor/17 add; a7 vs a6; e8 = 17 constants; e9 = a5 + 17 offsets);
                     35,640 legal; 200 states x 32 flips each.  -> stage1.log, stage1_results.json
- `stage2_sweep.py`  2,680 configurations of the non-W10 knobs (e10/e11 constants, sat_h,
                     a7 rotational/xor/add ties, a4 tied to digest words) on top of the best
                     bases; controls.  -> stage2.log, stage2_results.json
- `stage3_structure.py <defs> [--scan N] [--hist]`
                     closed form of the edge (verified exact), 32x32 sensitivity matrix,
                     GF(2) ranks, distinct additive differences, GF(2)-linearity, exhaustive
                     2^32 residual scans at fixed (state, a5p), and the full histogram of the
                     C1 index over all 2^32 a5 (distinct values, max multiplicity, sum n_v^2).
                     -> stage3.log (driver: run_stage3.sh)
