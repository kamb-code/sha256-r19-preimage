#!/usr/bin/env python3
"""Preregistered paired campaign: does context screening speed up the ATTACK?

Screening is known to raise C1/C2 fixed-point yield per context by ~12.7x, and
that lift is known to carry through to W9 lo-passes (p = 1.6e-3). What is NOT
established is whether it carries through the hi filter to a full 32-bit W9
match, and thence to verified preimages -- the events that actually determine
attack cost. This campaign measures that directly.

DESIGN (fixed before the run; do not amend after seeing results)
----------------------------------------------------------------
* N_TARGETS independent targets.
* For each target, generate a pool of contexts.
* Score EVERY context with a fixed, cheap 2^20 a0 sample.
* SCREENED arm: keep the top --keep fraction by score.
* CONTROL arm : an equal number of contexts drawn at random from the same pool,
  disjoint from the screened set where possible.
* Both arms receive EXACTLY the same full exposure per context (--exposure a0).
* No early stop on a hit. A found preimage is recorded and the context runs to
  completion, so the two arms are never given unequal exposure.
* Global dedup of fixed points within each (target, context).
* Every counter recorded: fixed points, lo-passes, hi-passes, full W9 matches,
  verified preimages, wall time.

HEADLINE ESTIMATOR
------------------
    lift = (N_LH_screened / C_screened) / (N_LH_random / C_random)

with N_LH = full 32-bit W9 matches and C = contexts fully attacked. Reported
with a target-blocked bootstrap interval, since contexts within a target are
not independent.

The screening score is deliberately the SAME quantity used in
context_screening.py (deduplicated C1/C2 fixed points on a short a0 sample), so
a null result here falsifies the operational claim rather than some proxy.

Usage:
    python3 campaign_screening.py --targets 50 --pool 20 --keep 0.2 \
        --exposure 4294967296 --seed 20260810 --out campaign.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from h100_extended import FpSeen, build_table, run_ctx
from extended_solver import backward_chain
from sha256_core import sha256_full_trace

R = 19


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def manifest(args, extra=None):
    m = {
        "commit": git_sha(),
        "command": " ".join(sys.argv),
        "args": vars(args),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": (torch.cuda.get_device_properties(0).name
                   if torch.cuda.is_available() else None),
        "platform": platform.platform(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        m.update(extra)
    return m


def make_target(rng):
    msg = rng.integers(0, 256, size=55, dtype=np.uint8).tobytes()
    return sha256_full_trace(msg, num_rounds=R).final_hash


def make_context(hb, rng):
    ka, _ = backward_chain(hb, R)
    ka = dict(ka)
    for r in range(4, R - 8):
        if r not in ka:
            ka[r] = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    return ka


def attack(tbl, hb, ka, exposure, max_iter, k_seeds, fp, gen):
    """Full attack on one context at a FIXED a0 exposure. Never stops early."""
    batch = min(1 << 24, exposure)
    stats, resid, found = run_ctx(tbl, hb, ka, R, max_iter, batch=batch,
                                  K_seeds=k_seeds, fp_seen=fp, gen=gen,
                                  n_a0=exposure, stop_on_found=False)
    return stats, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, default=50)
    ap.add_argument("--pool", type=int, default=20,
                    help="contexts generated per target")
    ap.add_argument("--keep", type=float, default=0.2,
                    help="top fraction kept in the screened arm")
    ap.add_argument("--screen-a0", type=int, default=1 << 20,
                    help="a0 budget for the cheap screening pass")
    ap.add_argument("--exposure", type=int, default=1 << 32,
                    help="a0 budget for each fully attacked context (both arms)")
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--k-seeds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--out", default="campaign.json")
    args = ap.parse_args()

    print(json.dumps(manifest(args), indent=2), flush=True)
    tbl = build_table()
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device="cuda")

    rows = []
    t_start = time.time()
    n_keep = max(1, int(round(args.keep * args.pool)))

    for ti in range(args.targets):
        hb = make_target(rng)
        tid = hb.hex()[:16]
        ctxs = [make_context(hb, rng) for _ in range(args.pool)]

        # --- screening pass: identical cheap budget on every context ---
        scores = []
        for ci, ka in enumerate(ctxs):
            gen.manual_seed(args.seed + 1013 * ti + ci)
            fp = FpSeen(tid, ci)
            st, _ = attack(tbl, hb, ka, args.screen_a0, args.max_iter,
                           args.k_seeds, fp, gen)
            scores.append(st["conv_unique"])
        order = np.argsort(scores)[::-1]

        screened = list(order[:n_keep])
        rest = [i for i in order[n_keep:]]
        rng.shuffle(rest)
        control = rest[:n_keep]          # disjoint from screened by construction

        # --- full exposure, identical for both arms, no early stop ---
        for arm, idxs in (("screened", screened), ("control", control)):
            for ci in idxs:
                gen.manual_seed(args.seed + 7_000_003 + 1013 * ti + int(ci))
                fp = FpSeen(tid, int(ci))
                t0 = time.time()
                st, found = attack(tbl, hb, ctxs[int(ci)], args.exposure,
                                   args.max_iter, args.k_seeds, fp, gen)
                rows.append({
                    "target": ti, "target_id": tid, "context": int(ci),
                    "arm": arm, "screen_score": int(scores[int(ci)]),
                    "fixed_points": st["conv_unique"],
                    "fp_trajectories": st["conv"],
                    "fp_replay": st["conv_replay"],
                    "lo": st["lo"], "hi": st["hi"],
                    "full_w9": st["cons"], "verified": st["verified"],
                    "seconds": time.time() - t0,
                })
                print(f"  t{ti:03d} c{int(ci):02d} {arm:8s} "
                      f"fp={st['conv_unique']:>7,} lo={st['lo']:>5} "
                      f"hi={st['hi']:>3} full={st['cons']:>3} "
                      f"ver={st['verified']} {rows[-1]['seconds']:.0f}s",
                      flush=True)
                json.dump({"manifest": manifest(args), "rows": rows},
                          open(args.out, "w"), indent=1)

    # --- headline estimator ---
    def agg(arm, field):
        v = [r[field] for r in rows if r["arm"] == arm]
        return sum(v), len(v)

    print(f"\n{'='*64}\ncampaign complete in {(time.time()-t_start)/3600:.2f} h")
    out = {"manifest": manifest(args), "rows": rows, "summary": {}}
    for field in ("fixed_points", "lo", "hi", "full_w9", "verified"):
        s_tot, s_n = agg("screened", field)
        c_tot, c_n = agg("control", field)
        s_rate = s_tot / max(s_n, 1)
        c_rate = c_tot / max(c_n, 1)
        lift = (s_rate / c_rate) if c_rate > 0 else float("nan")
        out["summary"][field] = {
            "screened_total": s_tot, "screened_contexts": s_n, "screened_rate": s_rate,
            "control_total": c_tot, "control_contexts": c_n, "control_rate": c_rate,
            "lift": lift,
        }
        print(f"  {field:<14} screened {s_tot:>8,}/{s_n:<4} = {s_rate:>10.4f}   "
              f"control {c_tot:>8,}/{c_n:<4} = {c_rate:>10.4f}   lift {lift:.2f}x")

    # target-blocked bootstrap on the decisive counter
    tg = sorted({r["target"] for r in rows})
    if len(tg) > 1:
        bs = []
        brng = np.random.default_rng(args.seed ^ 0x5EED)
        for _ in range(10000):
            pick = brng.choice(tg, size=len(tg), replace=True)
            sn = sd = cn = cd = 0
            for t in pick:
                for r in rows:
                    if r["target"] != t:
                        continue
                    if r["arm"] == "screened":
                        sn += r["full_w9"]; sd += 1
                    else:
                        cn += r["full_w9"]; cd += 1
            if sd and cd and cn > 0:
                bs.append((sn / sd) / (cn / cd))
        if bs:
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out["summary"]["full_w9_bootstrap_95"] = [float(lo), float(hi)]
            print(f"\n  full_w9 lift, target-blocked bootstrap 95% CI: "
                  f"[{lo:.2f}x, {hi:.2f}x]  ({len(bs)} resamples)")
        else:
            print("\n  bootstrap undefined: no full W9 matches in the control arm.")
            print("  Report as a one-sided bound, not a lift.")

    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
