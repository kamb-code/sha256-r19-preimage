#!/usr/bin/env python3
"""(a) INTEGER PROGRAMMING / CP on the R=20 constraint system.

The instance is exactly the one the table attack solves: submask-family context
a4..a11 and digest chain a12..a19 fixed, unknowns a0..a3, four schedule
constraints.  A solution is planted, so every instance is satisfiable.

Two models:
  * CP-SAT (OR-Tools): integer word variables channelled to bits, modular sums
    as linear equations with an explicit carry multiplier (so the LP relaxation
    sees the arithmetic), Maj as a two-sided linear inequality, Ch by
    reification, rotations/XOR at the bit level.  This is a genuine
    MILP/CP hybrid: lazy clause generation + LP + CDCL.
  * CBC through PuLP: the same model as a pure 0-1 MILP.

The ladder over word width w gives the scaling exponent, which is what decides
whether the method can ever beat the attack's 2^w sweep.
"""
import sys
import time
import numpy as np
from alg import Alg, Instance, R

from ortools.sat.python import cp_model


def planted_family_instance(A, rng, v=None):
    """Family context + planted a0..a3 + a consistent digest chain a12..a19."""
    M = A.M
    rv = lambda: int(rng.integers(0, 1 << A.w))
    if v is None:
        v = rv()
    ctx = A.family(v, rv(), rv(), rv(), rv())
    a = {-1: A.IV[0], -2: A.IV[1], -3: A.IV[2], -4: A.IV[3]}
    e = {-1: A.IV[4], -2: A.IV[5], -3: A.IV[6], -4: A.IV[7]}
    sol = [rv() for _ in range(4)]
    for i in range(4):
        a[i] = sol[i]
    a.update(ctx)
    for r in range(0, 12):
        e[r] = (a[r - 4] + a[r] - A.T2(a[r - 1], a[r - 2], a[r - 3])) & M
    W = {}
    for r in range(0, 12):
        W[r] = (a[r] - A.T2(a[r - 1], a[r - 2], a[r - 3]) - e[r - 4] - A.S1(e[r - 1])
                - A.Ch(e[r - 1], e[r - 2], e[r - 3]) - A.K[r]) & M
    for r in range(12, 16):
        W[r] = rv()
    for r in range(16, R):
        W[r] = (A.s1(W[r - 2]) + W[r - 7] + A.s0(W[r - 15]) + W[r - 16]) & M
    for r in range(12, R):
        T1 = (e[r - 4] + A.S1(e[r - 1]) + A.Ch(e[r - 1], e[r - 2], e[r - 3])
              + A.K[r] + W[r]) & M
        a[r] = (T1 + A.T2(a[r - 1], a[r - 2], a[r - 3])) & M
        e[r] = (a[r - 4] + T1) & M
    chain = {i: a[i] for i in range(12, R)}
    inst = Instance(A, chain, ctx)
    res = inst.residuals(*sol)
    assert all(x == 0 for x in res), f"planted instance is inconsistent: {res}"
    return inst, sol, v


# --------------------------------------------------------------------------
class Word:
    def __init__(self, m, w, name, const=None):
        self.w = w
        self.m = m
        if const is not None:
            self.const = const & ((1 << w) - 1)
            self.bits = [(self.const >> i) & 1 for i in range(w)]
            self.x = self.const
            self.is_const = True
        else:
            self.is_const = False
            self.bits = [m.NewBoolVar(f"{name}_{i}") for i in range(w)]
            self.x = m.NewIntVar(0, (1 << w) - 1, name)
            m.Add(self.x == sum((1 << i) * self.bits[i] for i in range(w)))


def build(A, inst, timeout, workers, hint=None):
    w = A.w
    M = A.M
    m = cp_model.CpModel()
    words = {}

    def W_(name, const=None):
        return Word(m, w, name, const)

    def bitv(b):
        return b if not isinstance(b, int) else b

    def lit(b, neg=False):
        if isinstance(b, int):
            return (1 - b) if neg else b
        return b.Not() if neg else b

    def xor3(name, a, b, c):
        """t = a^b^c on bit literals (ints or BoolVars)."""
        consts = [x for x in (a, b, c) if isinstance(x, int)]
        vs = [x for x in (a, b, c) if not isinstance(x, int)]
        p = sum(consts) & 1
        if not vs:
            return p
        t = m.NewBoolVar(name)
        # t xor v1 xor v2 ... xor p = 0
        lits = list(vs) + [t]
        if p == 0:
            # xor of lits must be 0 -> negate one literal to use AddBoolXOr(=1)
            m.AddBoolXOr(lits[:-1] + [lits[-1].Not()])
        else:
            m.AddBoolXOr(lits)
        return t

    def rot_bits(word, r):
        return [word.bits[(i + r) % w] for i in range(w)]

    def shr_bits(word, r):
        return [word.bits[i + r] if i + r < w else 0 for i in range(w)]

    def SIG(name, word, amts, shift=False):
        r0, r1, r2 = amts
        b0 = rot_bits(word, r0)
        b1 = rot_bits(word, r1)
        b2 = shr_bits(word, r2) if shift else rot_bits(word, r2)
        out = W_(name) if not word.is_const else None
        if word.is_const:
            v = 0
            for i in range(w):
                v |= ((b0[i] ^ b1[i] ^ b2[i]) & 1) << i
            return W_(name, const=v)
        for i in range(w):
            t = xor3(f"{name}x{i}", b0[i], b1[i], b2[i])
            if isinstance(t, int):
                m.Add(out.bits[i] == t)
            else:
                m.Add(out.bits[i] == t)
        return out

    def Maj_(name, a, b, c):
        if a.is_const and b.is_const and c.is_const:
            return W_(name, const=A.Maj(a.const, b.const, c.const))
        out = W_(name)
        for i in range(w):
            s = a.bits[i] + b.bits[i] + c.bits[i]
            m.Add(s - 2 * out.bits[i] >= 0)
            m.Add(s - 2 * out.bits[i] <= 1)
        return out

    def Ch_(name, e, f, g):
        if e.is_const and f.is_const and g.is_const:
            return W_(name, const=A.Ch(e.const, f.const, g.const))
        out = W_(name)
        for i in range(w):
            ei, fi, gi = e.bits[i], f.bits[i], g.bits[i]
            if isinstance(ei, int):
                src = fi if ei else gi
                if isinstance(src, int):
                    m.Add(out.bits[i] == src)
                else:
                    m.Add(out.bits[i] == src)
            else:
                if isinstance(fi, int):
                    m.Add(out.bits[i] == fi).OnlyEnforceIf(ei)
                else:
                    m.Add(out.bits[i] == fi).OnlyEnforceIf(ei)
                if isinstance(gi, int):
                    m.Add(out.bits[i] == gi).OnlyEnforceIf(ei.Not())
                else:
                    m.Add(out.bits[i] == gi).OnlyEnforceIf(ei.Not())
        return out

    def modsum(name, terms, const=0):
        """terms: list of (sign, Word or int).  Returns a Word equal to the sum
        mod 2^w, with an explicit integer carry multiplier."""
        expr = []
        npos = nneg = 0
        cst = const
        for s, t in terms:
            if isinstance(t, int):
                cst += s * t
                continue
            if t.is_const:
                cst += s * t.const
                continue
            expr.append(s * t.x)
            if s > 0:
                npos += 1
            else:
                nneg += 1
        out = W_(name)
        lo = -(nneg * (M) + max(0, -cst))
        hi = npos * M + max(0, cst)
        kmin = (lo - M) // (1 << w) - 1
        kmax = hi // (1 << w) + 1
        k = m.NewIntVar(int(kmin), int(kmax), name + "_k")
        m.Add(sum(expr) + cst == out.x + (1 << w) * k)
        return out

    # ---- constants -------------------------------------------------------
    a = {}
    for r in (-1, -2, -3, -4):
        a[r] = W_(f"aIV{r}", const=A.IV[{-1: 0, -2: 1, -3: 2, -4: 3}[r]])
    e = {}
    for r in (-1, -2, -3, -4):
        e[r] = W_(f"eIV{r}", const=A.IV[{-1: 4, -2: 5, -3: 6, -4: 7}[r]])
    for r in range(4, R):
        a[r] = W_(f"actx{r}", const=inst.a[r])
    for r in range(8, R):
        e[r] = W_(f"ehi{r}", const=inst.e_hi[r])
    # ---- unknowns --------------------------------------------------------
    for r in range(4):
        a[r] = W_(f"a{r}")
    # ---- derived ---------------------------------------------------------
    T2w = {}
    for r in range(0, 12):
        s0a = SIG(f"S0a{r-1}", a[r - 1], A.S0r)
        mj = Maj_(f"Maj{r}", a[r - 1], a[r - 2], a[r - 3])
        T2w[r] = modsum(f"T2_{r}", [(1, s0a), (1, mj)])
    for r in range(0, 8):
        e[r] = modsum(f"e{r}", [(1, a[r - 4]), (1, a[r]), (-1, T2w[r])])
    Wv = {}
    for r in range(0, 12):
        s1e = SIG(f"S1e{r-1}", e[r - 1], A.S1r)
        ch = Ch_(f"Ch{r}", e[r - 1], e[r - 2], e[r - 3])
        Wv[r] = modsum(f"W{r}", [(1, a[r]), (-1, T2w[r]), (-1, e[r - 4]),
                                 (-1, s1e), (-1, ch)], const=-A.K[r])
    for r in range(12, R):
        Wv[r] = W_(f"Wc{r}", const=inst.Wc[r])
    # W12..W19 constants; the s1 terms of the constraints are then constants
    for j in range(4):
        s1t = SIG(f"s1W{14+j}", Wv[14 + j], A.s1r, shift=True) if (14 + j) < 12 \
            else W_(f"s1W{14+j}", const=A.s1(inst.Wc[14 + j]))
        s0t = SIG(f"s0W{1+j}", Wv[1 + j], A.s0r, shift=True)
        z = modsum(f"c{j}", [(1, Wv[16 + j]), (-1, s1t), (-1, Wv[9 + j]),
                             (-1, s0t), (-1, Wv[j])])
        m.Add(z.x == 0)
    if hint is not None:
        for r in range(4):
            m.AddHint(a[r].x, hint[r])
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_search_workers = workers
    solver.parameters.log_search_progress = False
    t0 = time.time()
    st = solver.Solve(m)
    el = time.time() - t0
    got = None
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        got = [solver.Value(a[r].x) for r in range(4)]
    return solver.StatusName(st), el, got, solver.NumBranches(), solver.NumConflicts()


if __name__ == "__main__":
    ws = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [6, 8, 10, 12, 14, 16]
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 28
    ntr = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    print(f"CP-SAT on the context-fixed R=20 family instance, planted solution, "
          f"{timeout}s cap, {workers} workers")
    print(" w   trial   status        seconds    branches    conflicts   solved?")
    for w in ws:
        A = Alg(w)
        rng = np.random.default_rng(1000 + w)
        for t in range(ntr):
            inst, sol, v = planted_family_instance(A, rng)
            st, el, got, nb, nc = build(A, inst, timeout, workers)
            ok = "-"
            if got is not None:
                res = inst.residuals(*got)
                ok = "YES" if all(x == 0 for x in res) else "BAD"
            print(f"{w:3d}  {t:5d}   {st:12s} {el:9.2f}  {nb:10d}  {nc:10d}   {ok}",
                  flush=True)
