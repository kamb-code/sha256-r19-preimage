#!/usr/bin/env python3
"""(a) Does the LP relaxation help?  CP-SAT with linearization_level 0 (pure
CP/CDCL, no LP), 1 (default) and 2 (full MILP-style linear relaxation solved by
the internal simplex at every node).  If the LP carried information, level 2
would win.  Same planted instances as a_milp.py.
"""
import sys
import time
import numpy as np
from ortools.sat.python import cp_model
from alg import Alg
import a_milp


def build_lvl(A, inst, timeout, workers, level):
    # monkey-patch: rebuild the model then set the parameter
    import a_milp as am
    orig = cp_model.CpSolver

    class S(orig):
        def __init__(self):
            super().__init__()
            self.parameters.linearization_level = level
    cp_model.CpSolver = S
    try:
        r = am.build(A, inst, timeout, workers)
    finally:
        cp_model.CpSolver = orig
    return r


if __name__ == "__main__":
    ws = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "6,7").split(',')]
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 180.0
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    ntr = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    print("CP-SAT linearization_level sweep (0 = no LP, 1 = default, "
          "2 = full LP at every node)")
    print(" w  trial  level   status      seconds   branches   conflicts")
    for w in ws:
        A = Alg(w)
        for t in range(ntr):
            rng = np.random.default_rng(5000 + 97 * w + t)
            inst, sol, v = a_milp.planted_family_instance(A, rng)
            for level in (0, 1, 2):
                st, el, got, nb, nc = build_lvl(A, inst, timeout, workers, level)
                print(f"{w:3d} {t:5d} {level:6d}   {st:10s} {el:9.2f} {nb:10d} {nc:10d}",
                      flush=True)
