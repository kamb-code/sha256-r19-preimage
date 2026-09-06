#!/usr/bin/env python3
"""Build one instance, solve it with one solver, verify the model, print JSON.

usage: solve_one.py R structure seed solver timeout [--tight k] [--ones] [--fix-a0]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/administrator/sha/publish/code")
from cnf_sha256 import build, decode, random_target, M  # noqa: E402
from submask_family import make_context, forward  # noqa: E402
import verify_r19  # noqa: E402

KISSAT = "/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/kissat/build/kissat"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("R", type=int)
    ap.add_argument("structure")
    ap.add_argument("seed", type=int)
    ap.add_argument("solver")
    ap.add_argument("timeout", type=float)
    ap.add_argument("--tight", type=int, default=0)
    ap.add_argument("--ones", action="store_true", help="Zaikin's all-ones digest as target")
    ap.add_argument("--fix-a0", action="store_true", help="also fix a0 (context variants only)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-cnf", action="store_true")
    args = ap.parse_args()

    R, S, seed = args.R, args.structure, args.seed
    target = "ff" * 32 if args.ones else random_target(seed, R)
    rng = np.random.default_rng(10_000 + seed)
    ctx = make_context(rng, R) if S in ("context", "ctxcoll") else None
    if ctx is not None:
        ctx = {k: v for k, v in ctx.items() if k <= R - 9}
    extra = None
    if args.fix_a0:
        a0 = int(rng.integers(0, 1 << 32, dtype=np.uint64))

        def extra(cnf, a, e, W):
            cnf.fix_word(a[0], a0)
    t0 = time.time()
    cnf, info = build(R, target, S, ctx=ctx, tight=args.tight, extra=extra)
    t_build = time.time() - t0
    tag = f"R{R}_{S}_t{args.tight}_s{seed}_{'ones' if args.ones else 'rnd'}{'_a0' if args.fix_a0 else ''}"
    res = dict(R=R, structure=S, tight=args.tight, seed=seed, solver=args.solver, target=target,
               ones=args.ones, fix_a0=args.fix_a0, nvars=cnf.nv, nclauses=len(cnf.clauses),
               build_s=round(t_build, 2), timeout=args.timeout)
    model = None
    t0 = time.time()
    if args.solver == "kissat":
        path = os.path.join(HERE, "cnf", tag + ".cnf")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cnf.write(path)
        cmd = [KISSAT, f"--seed={seed}", f"--time={int(args.timeout)}", "-q", path]
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = p.stdout
        status = "UNKNOWN"
        lits = []
        for line in out.splitlines():
            if line.startswith("s "):
                status = line[2:].strip()
            elif line.startswith("v "):
                lits.extend(int(x) for x in line[2:].split())
        if status == "SATISFIABLE":
            model = [l for l in lits if l != 0]
        res["status"] = status
        if not args.keep_cnf:
            os.remove(path)
    else:
        from pysat.solvers import Solver
        s = Solver(name=args.solver, bootstrap_with=cnf.clauses)
        timer = threading.Timer(args.timeout, s.interrupt)
        timer.start()
        r = s.solve_limited(expect_interrupt=True)
        timer.cancel()
        res["status"] = {True: "SATISFIABLE", False: "UNSATISFIABLE", None: "UNKNOWN"}[r]
        if r:
            model = s.get_model()
        s.delete()
    res["solve_s"] = round(time.time() - t0, 3)
    if model is not None:
        Wm, st = decode(model, info)
        comp = verify_r19.sha256_reduced_raw_block(Wm, R)
        tgt = verify_r19.parse_hash(target)
        res["verified"] = comp == tgt
        res["preimage"] = " ".join(f"{w:08x}" for w in Wm)
        a_true, e_true, _ = forward(Wm, R)
        res["family_ok"] = (a_true[4] == a_true[5] and e_true[8] == M and (e_true[9] == M or R < 18))
        res["collapse_ok"] = ((a_true[2] ^ a_true[3]) & (a_true[3] ^ a_true[4])) == 0
        if ctx is not None:
            res["context_ok"] = all(a_true[k] == v for k, v in ctx.items())
    line = json.dumps(res)
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
