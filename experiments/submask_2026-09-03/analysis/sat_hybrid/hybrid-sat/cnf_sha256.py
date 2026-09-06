#!/usr/bin/env python3
"""CNF encoding of the R-round SHA-256 compression function (standard IV,
feed-forward folded into the target, no padding: 16 free 32-bit words), with
optional structural constraints from the submask-family attack.

Structure levels
  none     : plain preimage instance (target digest only)
  family   : + a4 == a5, e8 == e9 == 0xFFFFFFFF          (96 bit-conditions)
  collapse : family + (a2 ^ a3) & (a3 ^ a4) == 0          (bitwise, 2 clauses/bit)
  context  : family + a4..a_{R-9} fixed to a legal family context (make_context)
  ctxcoll  : context + collapse

Bit i of a word is the i-th least significant bit.  Literals are ints; the
constants True/False are folded at generation time.
"""
from __future__ import annotations

import struct
import sys

sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import K, IV, M, forward, digest, make_context, backward_chain, T2 as T2_int  # noqa: E402

TRUE, FALSE = True, False


class CNF:
    def __init__(self):
        self.nv = 0
        self.clauses = []

    def var(self):
        self.nv += 1
        return self.nv

    def word(self):
        return [self.var() for _ in range(32)]

    def const_word(self, x):
        return [bool((x >> i) & 1) for i in range(32)]

    def add(self, cl):
        # constant folding
        out = []
        for l in cl:
            if l is True:
                return
            if l is False:
                continue
            out.append(l)
        self.clauses.append(out)

    # ---- gates -----------------------------------------------------------
    @staticmethod
    def neg(l):
        if l is True:
            return False
        if l is False:
            return True
        return -l

    def xor2(self, a, b):
        if a is True:
            return self.neg(b)
        if a is False:
            return b
        if b is True:
            return self.neg(a)
        if b is False:
            return a
        if a == b:
            return False
        if a == -b:
            return True
        z = self.var()
        self.add([-a, -b, -z]); self.add([a, b, -z]); self.add([a, -b, z]); self.add([-a, b, z])
        return z

    def xor3(self, a, b, c):
        cs = [x for x in (a, b, c) if isinstance(x, bool)]
        if cs:
            vs = [x for x in (a, b, c) if not isinstance(x, bool)]
            par = sum(cs) & 1
            if len(vs) == 0:
                return bool(par)
            if len(vs) == 1:
                return self.neg(vs[0]) if par else vs[0]
            r = self.xor2(vs[0], vs[1])
            return self.neg(r) if par else r
        z = self.var()
        for va in (0, 1):
            for vb in (0, 1):
                for vc in (0, 1):
                    vz_bad = 1 - (va ^ vb ^ vc)
                    # clause falsified exactly by (a,b,c,z) = (va,vb,vc,vz_bad)
                    self.add([a if va == 0 else -a, b if vb == 0 else -b,
                              c if vc == 0 else -c, z if vz_bad == 0 else -z])
        return z

    def and2(self, a, b):
        if a is False or b is False:
            return False
        if a is True:
            return b
        if b is True:
            return a
        if a == b:
            return a
        if a == -b:
            return False
        z = self.var()
        self.add([-a, -b, z]); self.add([a, -z]); self.add([b, -z])
        return z

    def or2(self, a, b):
        return self.neg(self.and2(self.neg(a), self.neg(b)))

    def maj(self, a, b, c):
        if isinstance(a, bool) or isinstance(b, bool) or isinstance(c, bool):
            # reduce: maj with a constant
            vals = [a, b, c]
            cs = [x for x in vals if isinstance(x, bool)]
            vs = [x for x in vals if not isinstance(x, bool)]
            if len(cs) == 3:
                return sum(cs) >= 2
            if len(cs) == 2:
                if cs[0] == cs[1]:
                    return cs[0]
                return vs[0]
            # one constant
            if cs[0]:
                return self.or2(vs[0], vs[1])
            return self.and2(vs[0], vs[1])
        if a == b or a == c:
            return a
        if b == c:
            return b
        if a == -b:
            return c
        if a == -c:
            return b
        if b == -c:
            return a
        z = self.var()
        self.add([-a, -b, z]); self.add([-a, -c, z]); self.add([-b, -c, z])
        self.add([a, b, -z]); self.add([a, c, -z]); self.add([b, c, -z])
        return z

    def ch(self, e, f, g):
        # z = e ? f : g
        if e is True:
            return f
        if e is False:
            return g
        if isinstance(f, bool) and isinstance(g, bool):
            if f == g:
                return f
            return e if f else self.neg(e)
        if isinstance(f, bool):
            return self.or2(e, g) if f else self.and2(self.neg(e), g)
        if isinstance(g, bool):
            return self.or2(self.neg(e), f) if g else self.and2(e, f)
        if f == g:
            return f
        z = self.var()
        self.add([-e, -f, z]); self.add([-e, f, -z]); self.add([e, -g, z]); self.add([e, g, -z])
        self.add([-f, -g, z]); self.add([f, g, -z])
        return z

    # ---- words -----------------------------------------------------------
    @staticmethod
    def rotr(w, n):
        return [w[(i + n) % 32] for i in range(32)]

    @staticmethod
    def shr(w, n):
        return [w[i + n] if i + n < 32 else False for i in range(32)]

    def xor3w(self, x, y, z):
        return [self.xor3(x[i], y[i], z[i]) for i in range(32)]

    def S0(self, w):
        return self.xor3w(self.rotr(w, 2), self.rotr(w, 13), self.rotr(w, 22))

    def S1(self, w):
        return self.xor3w(self.rotr(w, 6), self.rotr(w, 11), self.rotr(w, 25))

    def s0(self, w):
        return self.xor3w(self.rotr(w, 7), self.rotr(w, 18), self.shr(w, 3))

    def s1(self, w):
        return self.xor3w(self.rotr(w, 17), self.rotr(w, 19), self.shr(w, 10))

    def chw(self, e, f, g):
        return [self.ch(e[i], f[i], g[i]) for i in range(32)]

    def majw(self, a, b, c):
        return [self.maj(a[i], b[i], c[i]) for i in range(32)]

    def addw(self, x, y):
        """ripple-carry 32-bit modular addition"""
        out = []
        c = False
        for i in range(32):
            out.append(self.xor3(x[i], y[i], c))
            if i < 31:
                c = self.maj(x[i], y[i], c)
        return out

    def add_many(self, ws):
        acc = ws[0]
        for w in ws[1:]:
            acc = self.addw(acc, w)
        return acc

    def eq_words(self, x, y):
        for i in range(32):
            a, b = x[i], y[i]
            self.add([self.neg(a), b]); self.add([a, self.neg(b)])

    def fix_word(self, w, val):
        for i in range(32):
            self.add([w[i] if (val >> i) & 1 else self.neg(w[i])])

    def write(self, path):
        with open(path, "w") as f:
            f.write(f"p cnf {self.nv} {len(self.clauses)}\n")
            f.write("\n".join(" ".join(map(str, cl)) + " 0" for cl in self.clauses))
            f.write("\n")


def build(R, target_hex, structure="none", ctx=None, tight=0, extra=None):
    """Return (cnf, info).  info holds variable words for decoding.

    tight: number of extra saturated words (a4=..=a_{4+tight}, e8..e_{8+tight} = -1) beyond
    the family (tight=0 is the family itself)."""
    h = bytes.fromhex(target_hex)
    tgt = [(struct.unpack(">I", h[4 * i:4 * i + 4])[0] - IV[i]) & M for i in range(8)]
    cnf = CNF()
    W = [cnf.word() for _ in range(16)]
    # message schedule
    for t in range(16, R):
        W.append(cnf.add_many([cnf.s1(W[t - 2]), W[t - 7], cnf.s0(W[t - 15]), W[t - 16]]))
    a = {-1: cnf.const_word(IV[0]), -2: cnf.const_word(IV[1]), -3: cnf.const_word(IV[2]), -4: cnf.const_word(IV[3])}
    e = {-1: cnf.const_word(IV[4]), -2: cnf.const_word(IV[5]), -3: cnf.const_word(IV[6]), -4: cnf.const_word(IV[7])}
    for r in range(R):
        T1 = cnf.add_many([e[r - 4], cnf.S1(e[r - 1]), cnf.chw(e[r - 1], e[r - 2], e[r - 3]),
                           cnf.const_word(K[r]), W[r]])
        T2 = cnf.addw(cnf.S0(a[r - 1]), cnf.majw(a[r - 1], a[r - 2], a[r - 3]))
        a[r] = cnf.addw(T1, T2)
        e[r] = cnf.addw(a[r - 4], T1)
    # the digest (feed-forward folded into constants)
    for i, r in enumerate((R - 1, R - 2, R - 3, R - 4)):
        cnf.fix_word(a[r], tgt[i])
        cnf.fix_word(e[r], tgt[4 + i])
    # --- structure ---------------------------------------------------------
    if structure != "none":
        # e_r = a_{r-4} + a_r - T2 is a condition on context words only while
        # a_r is a context word (r <= R-9); at R = 17 the e9 condition would
        # pin a digest word, so it is dropped there (the family degenerates).
        for j in range(tight + 1):
            cnf.eq_words(a[4 + j], a[5 + j])
            if 8 + j <= R - 9:
                cnf.fix_word(e[8 + j], M)
            if 9 + j <= R - 9:
                cnf.fix_word(e[9 + j], M)
    if structure in ("collapse", "ctxcoll"):
        for i in range(32):
            a2, a3, v = a[2][i], a[3][i], a[4][i]
            # forbid (a2,a3,v) = (0,1,0) and (1,0,1)
            cnf.add([a2, -a3, v]); cnf.add([-a2, a3, -v])
    if structure in ("context", "ctxcoll"):
        assert ctx is not None
        for k, val in ctx.items():
            cnf.fix_word(a[k], val)
    if extra:
        extra(cnf, a, e, W)
    info = dict(W=W, a=a, e=e, R=R, target=target_hex)
    return cnf, info


def decode(model, info):
    """model: set/list of true literals (pysat style list of ints).  Returns W0..W15."""
    ms = set(l for l in model if l > 0)

    def val(w):
        x = 0
        for i in range(32):
            b = w[i]
            if b is True:
                bit = 1
            elif b is False:
                bit = 0
            elif b > 0:
                bit = 1 if b in ms else 0
            else:
                bit = 0 if -b in ms else 1
            x |= bit << i
        return x
    Wm = [val(info["W"][i]) for i in range(16)]
    st = {("a", r): val(w) for r, w in info["a"].items() if r >= 0}
    st.update({("e", r): val(w) for r, w in info["e"].items() if r >= 0})
    return Wm, st


def random_target(seed, R):
    import numpy as np
    rng = np.random.default_rng(seed)
    msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
    pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
    Wt = [struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)]
    return digest(Wt, R).hex()


if __name__ == "__main__":
    # self-test: encode R rounds, fix the message, solve, compare the decoded digest.
    import numpy as np
    from pysat.solvers import Cadical195
    for R in (17, 19, 20):
        rng = np.random.default_rng(R)
        Wt = [int(x) for x in rng.integers(0, 1 << 32, 16, dtype=np.uint64)]
        h = digest(Wt, R).hex()
        cnf, info = build(R, h, "none")
        for i in range(16):
            cnf.fix_word(info["W"][i], Wt[i])
        s = Cadical195(bootstrap_with=cnf.clauses)
        ok = s.solve()
        Wm, st = decode(s.get_model(), info)
        a_true, e_true, _ = forward(Wt, R)
        okst = all(st[("a", r)] == a_true[r] and st[("e", r)] == e_true[r] for r in range(R))
        print(f"R={R}: vars {cnf.nv} clauses {len(cnf.clauses)} sat={ok} W ok={Wm == Wt} state ok={okst}")
