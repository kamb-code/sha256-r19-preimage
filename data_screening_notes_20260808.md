# Context Screening: a ~12x Speedup (2026-08-08)

Scripts: `rw_context_screening.py`, and the predictiveness/economics drivers.

**This is a positive result: an algorithmic improvement to the R=19 attack as
published, not a correction to a claim about it.**

## The observation

The attack samples a random context and then sweeps `a0` over `2^32`, giving
every context the same budget. That is optimal only if contexts are
interchangeable. They are not.

Over 280 sampled contexts, deduplicated C1/C2 fixed points per context:

```text
mean 63.4     variance 97,082     variance/mean = 1,530     (Poisson = 1.0)
min 0         median 12           max 4,924
top 10% of contexts hold 68.6% of all yield
top 25% hold 85.3%
```

A 400x spread between the median context and the best one, and 1,530x
overdispersion against the Poisson that would hold if contexts were equivalent.

## It is predictive, which is what makes it usable

Overdispersion alone proves nothing: it could be per-sample noise. The test that
matters is whether a score measured on one `a0` sample predicts yield on an
independent one. Scoring the SAME context twice on disjoint `a0` samples:

```text
392 contexts, 2^20 a0 per sample
  Spearman rank correlation between samples = 0.910
  top decile chosen on sample 1 -> 5.96x average yield on sample 2
```

A rank correlation of 0.91 means productivity is a stable property of the
context, recoverable from a cheap sample.

## Economics

Screening costs `2^20` a0 against a full sweep of `2^32`, i.e. **1/4096**. So
even aggressive screening is nearly free. Out-of-sample lift and net speedup:

```text
keep    lift   screen overhead   NET speedup
 50%   1.84x        0.049%          1.84x
 25%   2.97x        0.098%          2.97x
 10%   4.98x        0.244%          4.97x
  5%   7.09x        0.488%          7.06x
  2%  10.17x        1.221%         10.04x
  1%  12.99x        2.441%         12.68x
```

**Best measured: keep the top 1%, ~12.7x net speedup.**

The algorithm is simply: sample many contexts, run a short `2^20` a0 pass on
each, keep the best 1%, and spend the full `2^32` sweep only on those.

## Caveats

- The score counts C1/C2 fixed points, which is a PROXY. Whether the lift
  carries through to lo-pass, hi-pass and full matches is not yet measured --
  the screening sample produced too few of those events (lo-pass mean 0.0 per
  context at this budget). This must be confirmed before the figure is quoted
  as a speedup on preimages rather than on fixed points.
- The top-1% row rests on ~4 contexts of 392 and is correspondingly noisy. The
  trend suggests deeper screening pays more, but the tail is not well measured.
- The two-dimensional optimum (screening depth x keep fraction) is unexplored;
  a longer screening pass would raise the correlation at higher overhead.

## Why it matters beyond R=19

Nothing in this depends on the round count. It is a property of how contexts are
sampled, so it applies unchanged at R=20 and to any variant of the attack that
samples a free context and then sweeps.

## Related results from the same batch

**Representative policy does not matter.** MAX vs MIN representative tables on
byte-identical contexts:

```text
            lo enhancement    hi enhancement
MAX table        28.5x             6.9x
MIN table        30.0x            11.1x
```

Agreeing within Poisson error on 8 and 13 hi events. This closes the review
objection that the measured enhancements might be artefacts of which
representative the table stores: two very different policies give the same
rates, so the effect is a property of `sigma0(W)-W`.

**lo and hi are independent at measurable widths**, contrary to the concern that
lo-consistent seeding would correlate them:

```text
width    D = P(both)/[P(lo)P(hi)]
  4b        1.00 +/- 0.03
  6b        1.06 +/- 0.10
  8b        1.39 +/- 0.27  (MAX) / 0.76 +/- 0.21 (MIN)
```

So the independence-derived projection for mu is better justified than the
earlier caution allowed -- though the 12-bit row still has zero joint events and
cannot test the exact 16-bit condition.
