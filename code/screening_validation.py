#!/usr/bin/env python3
"""Reproduce the screening predictiveness and keep-fraction economics.

`context_screening.py` measures the per-context yield DISTRIBUTION and shows it
is heavily overdispersed. Overdispersion alone proves nothing usable: it could
be per-sample noise rather than a stable property of the context. This script
runs the two follow-on measurements that decide whether screening is usable,
which were previously done with uncommitted ad-hoc drivers:

  1. PREDICTIVENESS. Score the SAME context twice on two independent a0
     samples and correlate the scores (Spearman, on ranks). A high correlation
     means productivity is a recoverable property of the context.

  2. ECONOMICS. Rank contexts by their sample-1 score, keep the top f, and
     measure the realised yield on sample 2 -- i.e. strictly OUT OF SAMPLE.
     Charge the screening pass its true cost and report the net speedup.

Both are reported for fixed points and, when the budget produces any, for
lo-pass events.

    IMPORTANT SCOPE LIMIT
    ---------------------
    The score counts deduplicated C1/C2 FIXED POINTS. Any lift reported here is
    a lift in fixed-point yield per attacked context. It is NOT a measured
    preimage speedup: whether productive contexts also convert fixed points
    into full 32-bit W9 matches and verified preimages at the same rate is not
    established by this script; the preregistered paired experiment that
    measures it is campaign_screening.py (result: 3.41x on lo-passes,
    conversion to preimages unmeasured). Do not quote these numbers as an
    attack speedup.

Usage (the exact command behind the published figures):

    SIGMA0_TABLE=/path/to/sigma0_u_table.npy \\
    python3 screening_validation.py --jobs 14 --contexts-per-job 28 \\
        --batch 1048576 --nbatch 1 --seed 99001 --json screening_validation.json

Every context and every a0 sample is drawn from a seeded generator, so the run
is deterministic given --seed, --jobs, --contexts-per-job, --batch and --nbatch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
np.seterr(over="ignore")

from context_screening import _score, _G  # noqa: E402
from measure_w9_filters import make_context  # noqa: E402


def spearman(x, y):
    """Rank correlation, with average ranks for ties (no SciPy dependency)."""
    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=float)
        r[order] = np.arange(v.size, dtype=float)
        # average ranks within tied groups
        vs = v[order]
        i = 0
        while i < vs.size:
            j = i
            while j + 1 < vs.size and vs[j + 1] == vs[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = ranks(np.asarray(x, float)), ranks(np.asarray(y, float))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def worker(job):
    """Score each context TWICE on independent a0 samples."""
    wid, nctx, nbatch, batch, iters, kseeds, seed = job
    sig = _G["sig"]
    out = []
    for i in range(nctx):
        # the context is built from its own stream, so it is identical across
        # the two scoring passes; only the a0 sample differs
        ctx_rng = np.random.default_rng(seed + 1013 * i)
        C = make_context(ctx_rng)
        r1 = np.random.default_rng(seed + 1013 * i + 7_000_003)
        r2 = np.random.default_rng(seed + 1013 * i + 11_000_003)
        conv1, lo1 = _score(sig, C, r1, nbatch, batch, iters, kseeds)
        conv2, lo2 = _score(sig, C, r2, nbatch, batch, iters, kseeds)
        out.append((conv1, lo1, conv2, lo2))
    return out


def economics(s1, s2, a0_each, keeps=(0.50, 0.25, 0.10, 0.05, 0.02, 0.01)):
    """Out-of-sample lift and net speedup for each keep-fraction.

    s1 selects (the screening pass), s2 measures (the independent sample).
    Screening costs `a0_each` a0 per context against a full sweep of 2^32, and
    is paid on EVERY context, while the full sweep is paid only on the kept
    fraction -- so the overhead of screening to keep f is (a0_each/2^32)/f.
    """
    n = s1.size
    base = s2.mean()
    rows = []
    order = np.argsort(s1)[::-1]
    for f in keeps:
        k = max(1, int(round(f * n)))
        sel = order[:k]
        lift = float(s2[sel].mean() / base) if base > 0 else float("nan")
        overhead = (a0_each / 2.0 ** 32) / f
        rows.append({
            "keep": f, "contexts_kept": k,
            "lift_out_of_sample": lift,
            "screen_overhead": overhead,
            "net_speedup": lift / (1.0 + overhead),
        })
    return rows, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table",
                    default=os.environ.get("SIGMA0_TABLE", "sigma0_u_table.npy"),
                    help="sigma0(u)-u table; defaults to $SIGMA0_TABLE")
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--contexts-per-job", type=int, default=28)
    ap.add_argument("--nbatch", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1 << 20)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--kseeds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=99001)
    ap.add_argument("--json", default=None, help="write full results here")
    a = ap.parse_args()

    print(f"loading table {a.table} ...", flush=True)
    _G["sig"] = np.load(a.table).view(np.uint32)

    import multiprocessing as mp
    jobs = [(w, a.contexts_per_job, a.nbatch, a.batch, a.iters, a.kseeds,
             a.seed + 100003 * w) for w in range(a.jobs)]
    t0 = time.time()
    with mp.get_context("fork").Pool(a.jobs) as pool:
        res = pool.map(worker, jobs)
    flat = [r for chunk in res for r in chunk]
    c1 = np.array([r[0] for r in flat], float)
    l1 = np.array([r[1] for r in flat], float)
    c2 = np.array([r[2] for r in flat], float)
    l2 = np.array([r[3] for r in flat], float)
    a0_each = a.nbatch * a.batch
    elapsed = time.time() - t0

    print(f"\ncontexts = {c1.size}, a0 per sample = {a0_each:,}, "
          f"two samples each  ({elapsed:.0f}s)")

    out = {
        "command": " ".join(sys.argv),
        "seed": a.seed, "jobs": a.jobs,
        "contexts_per_job": a.contexts_per_job,
        "contexts_total": int(c1.size),
        "a0_per_sample": a0_each,
        "nbatch": a.nbatch, "batch": a.batch,
        "iters": a.iters, "kseeds": a.kseeds,
        "elapsed_s": elapsed,
        "scope_limit": ("score counts C1/C2 fixed points; lift is fixed-point "
                        "yield, NOT a measured preimage speedup"),
        "raw": {"conv_sample1": c1.tolist(), "conv_sample2": c2.tolist(),
                "lo_sample1": l1.tolist(), "lo_sample2": l2.tolist()},
        "metrics": {},
    }

    for name, x1, x2 in (("fixed_points", c1, c2), ("lo_pass", l1, l2)):
        tot = x1.sum() + x2.sum()
        print(f"\n=== {name} ===  total events {tot:.0f}")
        if tot == 0:
            print("  no events at this budget; nothing to correlate.")
            out["metrics"][name] = {"total_events": 0}
            continue
        rho = spearman(x1, x2)
        rows, base = economics(x1, x2, a0_each)
        print(f"  Spearman(sample1, sample2) = {rho:.3f}")
        print(f"  mean per context (sample2) = {base:.2f}")
        print(f"  {'keep':>6} {'lift':>8} {'overhead':>10} {'net':>8}   "
              f"{'n kept':>7}")
        for r in rows:
            print(f"  {r['keep']*100:5.0f}% {r['lift_out_of_sample']:7.2f}x "
                  f"{r['screen_overhead']*100:9.3f}% "
                  f"{r['net_speedup']:7.2f}x   {r['contexts_kept']:7d}")
        out["metrics"][name] = {
            "total_events": float(tot),
            "spearman": rho,
            "mean_sample2": float(base),
            "economics": rows,
        }

    if out["metrics"].get("lo_pass", {}).get("total_events", 0) == 0:
        print("\nNOTE: zero lo-pass events at this budget. The lift above is a "
              "FIXED-POINT lift only.\n      It does not establish that "
              "screening improves the preimage success rate.")

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
