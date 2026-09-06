#!/usr/bin/env python3
"""One benchmark job: build CNF for (R, target), solve with one solver/seed,
verify the decoded message with the project's code/verify_r19.py, write JSON.

usage: run_job.py R KIND TSEED SOLVER SSEED OUT.json [--extra NAME] [--timeout SEC]
  KIND   rand  : target = R-round digest of a random 55-byte message (seed TSEED)
         ones  : target = 0xFF..FF (Zaikin's 1-hash); TSEED ignored
  SOLVER cadical195 | kissat
  --extra family      : add the submask-family conditions a4=a5, e8=e9=-1 as clauses
  --extra ctx         : fix a4..a10 (and a11 if R>=20) to a random family context (seed TSEED)
  --extra ctx0        : ctx plus a0 fixed (a 3-lookup instance; sanity)
  --extra fixw NBITS  : fix the low NBITS bits of W0 (Zaikin-style weakening control)
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import random

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = "/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad"
KISSAT = f"{SCR}/kissat/build/kissat"
PUBLISH = "/home/administrator/sha/publish"

sys.path.insert(0, HERE)
import cnf_sha256 as C  # noqa: E402


def load_verifier():
    spec = importlib.util.spec_from_file_location("verify_r19", f"{PUBLISH}/code/verify_r19.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def family_context(R, seed):
    """Random context in the submask family (same construction as make_context)."""
    rng = random.Random(seed)
    M = C.MASK32
    S0 = lambda x: C.rotr(x, 2) ^ C.rotr(x, 13) ^ C.rotr(x, 22)
    Maj = lambda a, b, c: (a & b) ^ (a & c) ^ (b & c)
    v = rng.getrandbits(32)
    ctx = {4: v, 5: v, 6: rng.getrandbits(32), 7: rng.getrandbits(32)}
    ctx[8] = (M - ctx[4] + S0(ctx[7]) + Maj(ctx[7], ctx[6], ctx[5])) & M
    ctx[9] = (M - ctx[5] + S0(ctx[8]) + Maj(ctx[8], ctx[7], ctx[6])) & M
    ctx[10] = rng.getrandbits(32)
    if R >= 20:
        ctx[11] = rng.getrandbits(32)
    return ctx


def main():
    args = sys.argv[1:]
    R = int(args[0]); kind = args[1]; tseed = int(args[2]); solver = args[3]; sseed = int(args[4]); out = args[5]
    extra = None; extra_arg = None; timeout = None
    i = 6
    while i < len(args):
        if args[i] == "--extra":
            extra = args[i + 1]; i += 2
            if extra == "fixw":
                extra_arg = int(args[i]); i += 1
        elif args[i] == "--timeout":
            timeout = float(args[i + 1]); i += 2
        else:
            raise SystemExit(f"bad arg {args[i]}")

    if kind == "rand":
        rng = random.Random(1000 + tseed)
        msg = bytes(rng.getrandbits(8) for _ in range(55))
        target = C.digest_of_message(msg, R)
    elif kind == "ones":
        target = b"\xff" * 32
    else:
        raise SystemExit("kind must be rand|ones")

    cons = []
    if extra == "family":
        cons = [("eq", "a", 4, "a", 5), ("fix", "e", 8, 0xFFFFFFFF), ("fix", "e", 9, 0xFFFFFFFF)]
    elif extra in ("ctx", "ctx0", "ctx11", "ctxrand", "ctxrand11"):
        # ctx*: submask-family context a4..a10 (a11 too for ctx11 at R>=20)
        # ctxrand*: uniformly random context words, same slots (the control)
        if extra.startswith("ctxrand"):
            rng = random.Random(6000 + tseed)
            ctx = {r: rng.getrandbits(32) for r in range(4, 12 if R >= 20 else 11)}
        else:
            ctx = family_context(R, 5000 + tseed)
        if not extra.endswith("11"):
            ctx.pop(11, None)
        cons = [("fix", "a", r, val) for r, val in ctx.items()]
        if extra == "ctx0":
            cons.append(("fix", "a", 0, random.Random(7000 + tseed).getrandbits(32)))
    elif extra == "fixw":
        nb = extra_arg
        rng = random.Random(9000 + tseed)
        cons = [("bits", "W", 0, {i: rng.getrandbits(1) for i in range(nb)})]

    t0 = time.perf_counter()
    clauses, vm = C.build(R, target, cons)
    t_build = time.perf_counter() - t0
    rec = {"R": R, "kind": kind, "tseed": tseed, "solver": solver, "sseed": sseed, "extra": extra,
           "extra_arg": extra_arg, "target": target.hex(), "nvars": vm.nvars, "nclauses": len(clauses),
           "t_build": t_build, "status": "unknown", "t_solve": None, "verified": None, "words": None}

    t0 = time.perf_counter()
    if solver == "kissat":
        path = out.replace(".json", ".cnf")
        C.write_dimacs(clauses, vm.nvars, path, [f"sha256 R={R} {kind} tseed={tseed} extra={extra}"])
        status, model, tail = C.solve_kissat(path, KISSAT, seed=sseed, timeout=timeout)
        rec["status"] = status
        try:
            os.remove(path)
        except OSError:
            pass
    else:
        model = C.solve_pysat(clauses, vm.nvars, solver, seed=sseed)
        status = "sat" if model is not None else "unsat"
        rec["status"] = status
    rec["t_solve"] = time.perf_counter() - t0

    if model is not None:
        words = vm.decode_message(model)
        ver = load_verifier()
        got = ver.sha256_reduced_raw_block(words, R)
        tgt = ver.parse_hash(target.hex())
        rec["verified"] = (got == tgt)
        rec["words"] = " ".join(f"{w:08x}" for w in words)
        if cons:
            # also record the state words the constraints touched, for the hybrid runs
            rec["a"] = {r: f"{vm.decode_word('a', r, model):08x}" for r in range(-4, R)}
            rec["e"] = {r: f"{vm.decode_word('e', r, model):08x}" for r in range(-4, R)}
    with open(out, "w") as f:
        json.dump(rec, f)
    print(json.dumps({k: rec[k] for k in ("R", "kind", "tseed", "solver", "sseed", "extra", "status", "t_solve", "verified")}))


if __name__ == "__main__":
    main()
