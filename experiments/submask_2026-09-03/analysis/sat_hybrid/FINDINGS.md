# SAT baseline and the algebra/SAT hybrid, 2026-09-06/07

Tests the author's hypothesis that going beyond 20 rounds "needs a hybrid or a
different mathematical technique".  CPU only (28 cores), CaDiCaL 1.9.5 and
kissat 4.0.4, single-thread jobs, 1800 s cap.  A Tseitin CNF generator for the
R-round compression function (standard IV, feed-forward, no padding, 16 free
message words) was written and validated: every model is decoded to
W0..W15 and checked with `code/verify_r19.py`.  Encoding size at R=19 is
11,355 vars / 70,487 clauses, comparable to Zaikin's Transalg CNF
(12,644 / 74,662).

## Baseline ladder (median solve seconds, 5 targets/seeds per cell)

| rounds | kissat, random digest | CaDiCaL, random | kissat, all-ones | CaDiCaL, all-ones |
|---|---|---|---|---|
| 14 | 0.08 | 0.07 | 0.08 | 0.08 |
| 15 | 0.09 | 0.08 | 0.09 | 0.08 |
| 16 | 0.09 | 0.08 | 0.09 | 0.08 |
| 17 | 0.92 | 0.39 | 0.90 | 1.03 |
| 18 | 102.23 | 154.57 | 48.06 | 888.82 |

This reproduces Zaikin's published ladder (17 rounds ~1 s, 18 rounds ~25 s on
one core) to within instance variance, so the encoding is a fair basis for the
hybrid test.  19 rounds is out of reach on one core, as his 3,550 core-hours
imply.

## The hybrid: negative

28 jobs at R=19 and R=20, three structure levels, two solvers, two seeds,
plus the all-ones digest.  **0 solved.**  Every job timed out at 1800 s.

| level | what is imposed | R=19 | R=20 |
|---|---|---|---|
| `family` | a4 = a5, e8 = e9 = 0xFFFFFFFF as clauses | 6/6 timeout | 4/4 timeout |
| `ctx` | family + a full legal context a4..a_(R-9) fixed | 6/6 timeout | 4/4 timeout |
| `ctxrand` | a random context fixed (control) | 4/4 timeout | 4/4 timeout |

An earlier partial run at R=17 and R=18 found the structured instances
*slower* than the unstructured ones (at R=18, seed 1: unstructured kissat 10 s,
family 591 s), and the collapsed condition (a2 XOR a3) AND (a3 XOR v) = 0 was
the worst of all (73 s against 2 s at R=17).

## The comparison worth quoting

In the `ctx` jobs the solver is handed **exactly the instance our attack
solves**: same digest, same seven context words pinned, only a0..a3 free.

| method, identical instance | time to a 19-round preimage |
|---|---|
| submask-family table attack | about 12 ms of one core |
| CaDiCaL / kissat | no solution in 1800 s |

A gap of more than 10^5, on the same instance, with an encoding the size of
Zaikin's.  The reason is structural: the 16 GB table answers
sigma0(u) - u = c in one lookup, and a clause learner has to rediscover that
inversion by search on every branch.  `ctxrand` matching `ctx` shows there is
no partial credit for handing a solver the algebraic structure.

## Conclusion

The two methods do not compose.  What makes the algebraic attack fast is
precisely what a CDCL solver cannot represent cheaply, and imposing our
conditions as clauses costs the solver rather than helping it.  A SAT/algebra
hybrid is not a route to 21 rounds.

## Also tested: the a/e duality (`duality/`)

SHA-256's two branches are coupled one way only (the e-words are determined by
the a-words, not the reverse), so the a-word parametrisation is the unique
local one.  Over 513 mixed frames at R=20 exactly four single-table absorbers
exist, all already known, and none is dual; the lightest version of the
blocking edge over all frames is 7.96 bits per flipped bit, against the 2-bit
threshold for absorption.  Symmetric contexts at R=21 gain nothing.  0 bits.

## Not tested

Rotational and complementation symmetry of the design, and Cube-and-Conquer on
the structured instance: both explorers were terminated by a session limit
before running.  The symmetry question is therefore still open.
