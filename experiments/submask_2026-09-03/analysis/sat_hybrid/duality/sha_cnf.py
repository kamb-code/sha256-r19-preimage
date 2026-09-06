#!/usr/bin/env python3
"""CNF encodings of the R-round SHA-256 preimage problem (raw block, standard IV,
feed-forward, no padding: exactly the convention of code/verify_r19.py).

Encodings ("variants"):
  W          standard: W0..W15 free, R rounds, digest fixed (Nossum-like structure,
             ripple-carry adders).
  A          W + unit clauses for the backward chain a_{R-8}..a_{R-1} (the a-frame).
  ctxfam     A + a_4..a_{R-9} fixed to a submask-family context.
  ctxrand    A + a_4..a_{R-9} fixed to a random context (control).
  ctxfam+col ctxfam + the collapse condition (a2^a3)&(a3^v)=0 (32 binary clauses).
  ctxfam+a0  ctxfam + a0 fixed (96 free bits).
  famrel     A + a4=a5 (equivalences) + e8=e9=-1 (units); a6,a7,a10.. free.
  famrel+col famrel + collapse clauses with v = a4 (variable v: ternary clauses).

Bits are LSB-first lists; each bit is either a bool constant or an int literal.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, digest,
                            backward_chain, forward, recover_W, make_context)


class CNF:
    def __init__(self):
        self.nv = 0
        self.clauses = []

    def var(self):
        self.nv += 1
        return self.nv

    def add(self, *lits):
        self.clauses.append(tuple(lits))

    # ---- bit gates with constant folding ---------------------------------
    @staticmethod
    def neg(x):
        return (not x) if isinstance(x, bool) else -x

    def xor2(self, a, b):
        if isinstance(a, bool) and isinstance(b, bool):
            return a != b
        if isinstance(a, bool):
            return self.neg(b) if a else b
        if isinstance(b, bool):
            return self.neg(a) if b else a
        x = self.var()
        self.add(-a, -b, -x); self.add(a, b, -x); self.add(a, -b, x); self.add(-a, b, x)
        return x

    def and2(self, a, b):
        if isinstance(a, bool):
            return b if a else False
        if isinstance(b, bool):
            return a if b else False
        x = self.var()
        self.add(-a, -b, x); self.add(a, -x); self.add(b, -x)
        return x

    def or2(self, a, b):
        return self.neg(self.and2(self.neg(a), self.neg(b)))

    def xor3(self, a, b, c):
        if any(isinstance(t, bool) for t in (a, b, c)):
            return self.xor2(self.xor2(a, b), c)
        x = self.var()
        for sa in (1, -1):
            for sb in (1, -1):
                for sc in (1, -1):
                    par = (sa < 0) ^ (sb < 0) ^ (sc < 0)
                    # clause excludes the assignment a=(sa<0),b=..,c=.. with x != parity
                    self.add(sa * a, sb * b, sc * c, x if par else -x)
        return x

    def maj(self, a, b, c):
        if any(isinstance(t, bool) for t in (a, b, c)):
            return self.or2(self.and2(a, b), self.or2(self.and2(a, c), self.and2(b, c)))
        x = self.var()
        self.add(-a, -b, x); self.add(-a, -c, x); self.add(-b, -c, x)
        self.add(a, b, -x); self.add(a, c, -x); self.add(b, c, -x)
        return x

    def ch(self, e, f, g):
        if any(isinstance(t, bool) for t in (e, f, g)):
            return self.xor2(self.and2(e, f), self.and2(self.neg(e), g))
        x = self.var()
        self.add(-e, -f, x); self.add(-e, f, -x); self.add(e, -g, x); self.add(e, g, -x)
        return x

    # ---- words ------------------------------------------------------------
    def const(self, v):
        return [bool((v >> i) & 1) for i in range(32)]

    def free(self):
        return [self.var() for _ in range(32)]

    @staticmethod
    def rotr(w, n):
        return [w[(i + n) % 32] for i in range(32)]

    @staticmethod
    def shr(w, n):
        return [w[i + n] if i + n < 32 else False for i in range(32)]

    def xorw(self, *ws):
        out = ws[0]
        for w in ws[1:]:
            out = [self.xor2(x, y) for x, y in zip(out, w)]
        return out

    def add32(self, x, y):
        out, c = [], False
        for i in range(32):
            out.append(self.xor3(x[i], y[i], c))
            if i < 31:
                c = self.maj(x[i], y[i], c)
        return out

    def sum32(self, *ws):
        out = ws[0]
        for w in ws[1:]:
            out = self.add32(out, w)
        return out

    def S0(self, w): return self.xorw(self.rotr(w, 2), self.rotr(w, 13), self.rotr(w, 22))
    def S1(self, w): return self.xorw(self.rotr(w, 6), self.rotr(w, 11), self.rotr(w, 25))
    def s0(self, w): return self.xorw(self.rotr(w, 7), self.rotr(w, 18), self.shr(w, 3))
    def s1(self, w): return self.xorw(self.rotr(w, 17), self.rotr(w, 19), self.shr(w, 10))
    def Ch(self, e, f, g): return [self.ch(a, b, c) for a, b, c in zip(e, f, g)]
    def Maj(self, a, b, c): return [self.maj(x, y, z) for x, y, z in zip(a, b, c)]

    def assert_eq_const(self, w, v):
        for i in range(32):
            bit = bool((v >> i) & 1)
            if isinstance(w[i], bool):
                if w[i] != bit:
                    self.add()          # empty clause: UNSAT
            else:
                self.add(w[i] if bit else -w[i])

    def assert_eq(self, w, u):
        for a, b in zip(w, u):
            if isinstance(a, bool) and isinstance(b, bool):
                if a != b:
                    self.add()
            elif isinstance(a, bool):
                self.add(b if a else -b)
            elif isinstance(b, bool):
                self.add(a if b else -a)
            else:
                self.add(-a, b); self.add(a, -b)

    def write(self, path):
        with open(path, "w") as f:
            f.write(f"p cnf {self.nv} {len(self.clauses)}\n")
            f.write("".join(" ".join(map(str, c)) + " 0\n" for c in self.clauses))


def build(R, target, variant, ctx=None, a0=None, seed=0):
    """Return (cnf, Wvars) for an R-round instance with the given target digest."""
    cnf = CNF()
    W = [cnf.free() for _ in range(16)]
    for t in range(16, R):
        W.append(cnf.sum32(cnf.s1(W[t - 2]), W[t - 7], cnf.s0(W[t - 15]), W[t - 16]))
    a = {i: cnf.const(IV[-i - 1 + 0]) for i in range(0)}  # placeholder
    a = {-1: cnf.const(IV[0]), -2: cnf.const(IV[1]), -3: cnf.const(IV[2]), -4: cnf.const(IV[3])}
    e = {-1: cnf.const(IV[4]), -2: cnf.const(IV[5]), -3: cnf.const(IV[6]), -4: cnf.const(IV[7])}
    for r in range(R):
        T1 = cnf.sum32(e[r - 4], cnf.S1(e[r - 1]), cnf.Ch(e[r - 1], e[r - 2], e[r - 3]),
                       cnf.const(K[r]), W[r])
        T2w = cnf.add32(cnf.S0(a[r - 1]), cnf.Maj(a[r - 1], a[r - 2], a[r - 3]))
        a[r] = cnf.add32(T1, T2w)
        e[r] = cnf.add32(a[r - 4], T1)
    # digest: state + IV == target  ->  state == target - IV
    h = [int.from_bytes(target[4 * i:4 * i + 4], "big") for i in range(8)]
    fin = [a[R - 1], a[R - 2], a[R - 3], a[R - 4], e[R - 1], e[R - 2], e[R - 3], e[R - 4]]
    for i in range(8):
        cnf.assert_eq_const(fin[i], (h[i] - IV[i]) & M)
    if variant == "W":
        return cnf, W
    ab, eb = backward_chain(target, R)
    for r in range(R - 8, R):
        cnf.assert_eq_const(a[r], ab[r])
    if variant == "A":
        return cnf, W
    if variant.startswith("ctx"):
        for r in range(4, R - 8):
            cnf.assert_eq_const(a[r], ctx[r])
        if variant.endswith("+col"):
            v = ctx[4]
            for i in range(32):
                # (a2^a3)&(a3^v)=0 : v_i=0 -> (a3_i -> a2_i); v_i=1 -> (a2_i -> a3_i)
                if (v >> i) & 1:
                    cnf.add(-a[2][i], a[3][i])
                else:
                    cnf.add(-a[3][i], a[2][i])
        if variant.endswith("+a0"):
            cnf.assert_eq_const(a[0], a0)
        return cnf, W
    if variant.startswith("famrel"):
        cnf.assert_eq(a[4], a[5])
        cnf.assert_eq_const(e[8], M)
        cnf.assert_eq_const(e[9], M)
        if variant.endswith("+col"):
            for i in range(32):
                v, x2, x3 = a[4][i], a[2][i], a[3][i]
                # forbid (x2 != x3) and (x3 != v):  x3=1,x2=0,v=0 and x3=0,x2=1,v=1
                cnf.add(-x3, x2, v)
                cnf.add(x3, -x2, -v)
        return cnf, W
    raise ValueError(variant)


def model_words(model, Wvars):
    """Extract W0..W15 from a kissat/pysat model (list of ints)."""
    pos = set(l for l in model if l > 0)
    out = []
    for w in Wvars[:16]:
        v = 0
        for i, b in enumerate(w):
            bit = b if isinstance(b, bool) else (b in pos)
            v |= int(bit) << i
        out.append(v)
    return out


def plant(R, rng, family=True, v=None):
    """Plant a message with a family (or random) context a4..a_{R-9}; return (W, digest, ctx, a)."""
    import numpy as np
    ctx = make_context(rng, R, v) if family else {i: int(rng.integers(0, 1 << 32, dtype=np.uint64)) for i in range(4, R - 8)}
    if family:
        # make_context builds 4..10 (or 11); extend to R-9 for R=21
        for i in range(4, R - 8):
            if i not in ctx:
                ctx[i] = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    a = {-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]}
    for i in range(0, 16):
        a[i] = ctx[i] if i in ctx else int(rng.integers(0, 1 << 32, dtype=np.uint64))
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(16):
        e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M
    W = [recover_W(a, e, r) for r in range(16)]
    h = digest(W, R)
    ctx = {i: ctx[i] for i in range(4, R - 8)}
    return W, h, ctx, a


if __name__ == "__main__":
    import numpy as np, hashlib, struct, time
    from pysat.solvers import Cadical195
    # self-test: 8 rounds, planted message, W-frame, CaDiCaL
    rng = np.random.default_rng(1)
    W, h, ctx, a = plant(12, rng)
    t0 = time.time()
    cnf, Wv = build(12, h, "W")
    print(f"R=12 W-frame: {cnf.nv} vars {len(cnf.clauses)} clauses, built in {time.time()-t0:.1f}s")
    with Cadical195(bootstrap_with=cnf.clauses) as s:
        ok = s.solve(); m = s.get_model()
    Wm = model_words(m, Wv)
    print("SAT" if ok else "UNSAT", "verified" if digest(Wm, 12) == h else "WRONG")
    # a-frame with family context, R=19, check the planted solution satisfies it
    W, h, ctx, a = plant(19, rng)
    cnf, Wv = build(19, h, "ctxfam+col", ctx=ctx)
    # force the planted a0 to test consistency of the context/units (collapse may fail: skip col)
    cnf2, Wv2 = build(19, h, "ctxfam", ctx=ctx)
    for i in range(16):
        cnf2.assert_eq_const(Wv2[i], W[i])
    with Cadical195(bootstrap_with=cnf2.clauses) as s:
        print("planted R=19 ctxfam instance with planted W forced:", "SAT (consistent)" if s.solve() else "UNSAT (BUG)")
    print(f"R=19 ctxfam+col: {cnf.nv} vars {len(cnf.clauses)} clauses")
