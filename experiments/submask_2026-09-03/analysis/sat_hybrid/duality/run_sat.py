#!/usr/bin/env python3
"""Run the hybrid SAT experiment: build CNFs for a list of jobs and solve with kissat,
8 jobs in parallel, one thread each, hard cap per run.  Results -> results.jsonl."""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sha_cnf import build, model_words, plant
from submask_family import digest, make_context, backward_chain, M

HERE = os.path.dirname(os.path.abspath(__file__))
KISSAT = "/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/kissat/build/kissat"
ONES = b"\xff" * 32
CAP_DEFAULT = 1800


def make_job(name, R, variant, target, seed, cap=CAP_DEFAULT):
    return dict(name=name, R=R, variant=variant, target=target, seed=seed, cap=cap)


def instance(job):
    """Return (digest, ctx, a0, planted_W) for a job."""
    R, variant, target, seed = job["R"], job["variant"], job["target"], job["seed"]
    rng = np.random.default_rng(1000 * R + seed)
    ctx = a0 = Wp = None
    if target == "ones":
        h = ONES
        if variant.startswith("ctx"):
            fam = not variant.startswith("ctxrand")
            ctx = make_context(rng, R) if fam else {i: int(rng.integers(0, 1 << 32, dtype=np.uint64)) for i in range(4, R - 8)}
            for i in range(4, R - 8):
                if i not in ctx:
                    ctx[i] = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    elif target == "planted":
        fam = not variant.startswith("ctxrand")
        Wp, h, ctx, a = plant(R, rng, family=fam)
        a0 = a[0]
    else:
        raise ValueError(target)
    return h, ctx, a0, Wp


def run(job):
    h, ctx, a0, Wp = instance(job)
    cnf, Wv = build(job["R"], h, job["variant"], ctx=ctx, a0=a0)
    path = os.path.join(HERE, "cnf", job["name"] + ".cnf")
    cnf.write(path)
    log = os.path.join(HERE, "logs", job["name"] + ".log")
    t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.run(["timeout", "-s", "KILL", str(job["cap"] + 30), KISSAT, "-q",
                            f"--time={job['cap']}", path], stdout=f, stderr=subprocess.STDOUT)
    el = time.time() - t0
    status, verified, Wm = "unknown", None, None
    txt = open(log).read()
    if "s SATISFIABLE" in txt:
        status = "SAT"
        model = []
        for line in txt.splitlines():
            if line.startswith("v "):
                model += [int(x) for x in line[2:].split()]
        Wm = model_words(model, Wv)
        verified = digest(Wm, job["R"]) == h
    elif "s UNSATISFIABLE" in txt:
        status = "UNSAT"
    elif p.returncode in (124, 137) or "UNKNOWN" in txt:
        status = "TIMEOUT"
    res = dict(job, nvars=cnf.nv, nclauses=len(cnf.clauses), status=status, time=round(el, 1),
               verified=verified, digest=h.hex(), ctx={k: f"{v:08x}" for k, v in (ctx or {}).items()},
               W=[f"{w:08x}" for w in Wm] if Wm else None, rc=p.returncode)
    with open(os.path.join(HERE, "results.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")
    print(f"{job['name']:<28} {status:<8} {el:8.1f}s  verified={verified}", flush=True)
    return res


def job_list(which):
    J = []
    if which in ("main", "all"):
        # group 1: W-frame calibration against Zaikin (all-ones digest = his 1-hash)
        for R in (17, 18):
            for s, tg in ((0, "ones"), (1, "planted"), (2, "planted")):
                J.append(make_job(f"W_R{R}_{tg}{s}", R, "W", tg, s, 1200))
        J.append(make_job("W_R19_ones0", 19, "W", "ones", 0))
        J.append(make_job("W_R19_planted1", 19, "W", "planted", 1))
        # group 2: a-frame (backward chain as units)
        J.append(make_job("A_R18_ones0", 18, "A", "ones", 0, 1200))
        J.append(make_job("A_R19_ones0", 19, "A", "ones", 0))
        J.append(make_job("A_R20_ones0", 20, "A", "ones", 0))
        # group 3: fixed contexts
        for s in (1, 2):
            J.append(make_job(f"ctxfam_R19_ones{s}", 19, "ctxfam", "ones", s))
            J.append(make_job(f"ctxrand_R19_ones{s}", 19, "ctxrand", "ones", s, 1200))
        J.append(make_job("ctxfam_R19_planted3", 19, "ctxfam", "planted", 3))
        J.append(make_job("ctxrand_R19_planted3", 19, "ctxrand", "planted", 3, 1200))
        for s in (4, 5):
            J.append(make_job(f"ctxfam_R20_planted{s}", 20, "ctxfam", "planted", s))
        J.append(make_job("ctxfam_R21_planted4", 21, "ctxfam", "planted", 4))
        # group 4: collapse clauses / a0 fixed
        for s in (1, 2):
            J.append(make_job(f"ctxfamcol_R19_ones{s}", 19, "ctxfam+col", "ones", s))
        J.append(make_job("ctxfamcol_R19_planted3", 19, "ctxfam+col", "planted", 3))
        for s in (6, 7):
            J.append(make_job(f"ctxfama0_R19_planted{s}", 19, "ctxfam+a0", "planted", s))
        # group 5: family relations only, context otherwise free
        for R in (19, 20, 21):
            J.append(make_job(f"famrel_R{R}_ones0", R, "famrel", "ones", 0))
        for R in (19, 20):
            J.append(make_job(f"famrelcol_R{R}_ones0", R, "famrel+col", "ones", 0))
    return J


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "main"
    jobs = job_list(which)
    done = set()
    if os.path.exists(os.path.join(HERE, "results.jsonl")):
        for line in open(os.path.join(HERE, "results.jsonl")):
            done.add(json.loads(line)["name"])
    jobs = [j for j in jobs if j["name"] not in done]
    print(f"{len(jobs)} jobs, kissat at {KISSAT}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(run, jobs))
    print(f"all done in {time.time()-t0:.0f}s", flush=True)
