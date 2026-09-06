"""How much of the instance does unit propagation alone settle, per structure level?"""
import sys, numpy as np
from pysat.solvers import Glucose4
from cnf_sha256 import build, random_target
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import make_context
R = int(sys.argv[1])
for S in ("none", "family", "collapse", "context"):
    ctx = {k: v for k, v in make_context(np.random.default_rng(10001), R).items() if k <= R - 9} if S == "context" else None
    cnf, info = build(R, random_target(1, R), S, ctx=ctx)
    units = [c[0] for c in cnf.clauses if len(c) == 1]
    s = Glucose4(bootstrap_with=[c for c in cnf.clauses if len(c) > 1])
    ok, lits = s.propagate(assumptions=units)
    fixed = set(abs(l) for l in lits)
    def nfixed(w): return sum(1 for b in w if isinstance(b, bool) or abs(b) in fixed)
    aw = " ".join(f"{nfixed(info['a'][r]):2d}" for r in range(R))
    ew = " ".join(f"{nfixed(info['e'][r]):2d}" for r in range(R))
    Ww = " ".join(f"{nfixed(info['W'][t]):2d}" for t in range(R))
    print(f"R={R} {S:8s} vars={cnf.nv} fixed-by-UP={len(fixed)} ({100*len(fixed)/cnf.nv:.1f}%)")
    print(f"   a_r bits fixed: {aw}\n   e_r bits fixed: {ew}\n   W_t bits fixed: {Ww}")
