#!/usr/bin/env python3
"""cnf_sha256.py -- Tseitin CNF generator for the R-round SHA-256 compression
function (raw one-block convention: standard IV, feed-forward, NO padding,
sixteen free 32-bit message words), matching code/verify_r19.py exactly.

Public API
----------
    build(R, target, extra_constraints=None, adder="ripple")
        -> (clauses, varmap)

    R                  number of rounds, 1 <= R <= 64
    target             the 256-bit digest as (i) 32-byte bytes, (ii) 64 hex
                       chars, or (iii) a list of eight 32-bit ints.  May be
                       None, in which case the output is left free (useful for
                       counting / planted experiments).
    extra_constraints  optional list of extra constraints; each item is one of
                         ("fix", name, r, value)        word (name,r) == value
                         ("eq",  name1, r1, name2, r2)  word == word
                         ("bits", name, r, {bit: 0/1})  fix individual bits
                         ("clauses", [[lit,...], ...])  raw clauses on varmap literals
                       where name in {"W", "a", "e"} and r is the index
                       (W: 0..R-1, a/e: -4..R-1 with -1..-4 the IV words).
    adder              "ripple" (2-input ripple-carry chains of full adders;
                       constant operands fold into half adders) -- the only
                       encoding shipped; the hook is there for others.

    clauses            list of clauses (lists of nonzero ints)
    varmap             VarMap object with
                          .nvars                  number of variables
                          .W[t][i], .a[r][i], .e[r][i]   bit i (LSB = 0) of the
                                                  word as a literal (int) or a
                                                  Python bool if constant
                          .word(name, r)          list of 32 bits
                          .decode_word(name, r, model)  -> int
                          .decode_message(model)  -> [W0..W15]
                          .stats                  dict of gate counts

    write_dimacs(clauses, nvars, path, comments=())
    solve_pysat(clauses, nvars, solver="cadical195", seed=0, ...) -> model|None
    parse_kissat_model(text) -> model (list of ints) or None
    digest_of_message(msg_bytes, R) -> 32 bytes   (pads like SHA-256, one block,
                       then applies the R-round raw compression: satisfiable by
                       construction for messages of <= 55 bytes)

State convention (as in the papers): a_r, e_r are the a- and e-registers after
round r (r = 0..R-1); a_{-1..-4} = IV[0..3], e_{-1..-4} = IV[4..7].  The digest
is IV + (a_{R-1}, a_{R-2}, a_{R-3}, a_{R-4}, e_{R-1}, ..., e_{R-4}).

Encoding: every 32-bit word is a list of 32 "bits", each either a DIMACS
literal or a constant bool.  Sigma/sigma are 3-input XORs of rotated/shifted
bits (8 clauses each), Ch is an if-then-else (4 clauses + 2 redundant), Maj is
6 clauses, a full adder is XOR3 (sum) + MAJ (carry).  Constants fold, so round
0 (all-constant state) costs almost nothing.  The target is applied as unit
clauses on the (digest - IV) state words, since the feed-forward with a
constant IV is a bijection per word.
"""
from __future__ import annotations

import hashlib
import struct
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

MASK32 = 0xFFFFFFFF

IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
      0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]

K = [
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
]

Bit = Union[int, bool]      # DIMACS literal (nonzero int) or constant bool
Word = List[Bit]            # 32 bits, index 0 = LSB


# ----------------------------------------------------------------------------
# reference (python ints) -- identical to code/verify_r19.py
# ----------------------------------------------------------------------------
def rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & MASK32


def sha256_reduced_raw_block(words: Sequence[int], rounds: int) -> List[int]:
    w = [int(x) & MASK32 for x in words]
    assert len(w) == 16
    for t in range(16, rounds):
        s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3)
        s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10)
        w.append((s1 + w[t - 7] + s0 + w[t - 16]) & MASK32)
    a, b, c, d, e, f, g, h = IV
    for t in range(rounds):
        S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
        ch = (e & f) ^ (~e & g)
        t1 = (h + S1 + ch + K[t] + w[t]) & MASK32
        S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
        mj = (a & b) ^ (a & c) ^ (b & c)
        t2 = (S0 + mj) & MASK32
        h, g, f, e = g, f, e, (d + t1) & MASK32
        d, c, b, a = c, b, a, (t1 + t2) & MASK32
    return [(IV[i] + v) & MASK32 for i, v in enumerate([a, b, c, d, e, f, g, h])]


def pad_one_block(msg: bytes) -> List[int]:
    """Standard SHA-256 padding of a message of <= 55 bytes into 16 words."""
    assert len(msg) <= 55
    blk = msg + b"\x80" + b"\x00" * (55 - len(msg)) + struct.pack(">Q", 8 * len(msg))
    return list(struct.unpack(">16I", blk))


def digest_of_message(msg: bytes, R: int) -> bytes:
    """R-round raw digest of the padded one-block message (satisfiable target)."""
    words = pad_one_block(msg)
    out = sha256_reduced_raw_block(words, R)
    d = b"".join(struct.pack(">I", x) for x in out)
    if R == 64:
        assert d == hashlib.sha256(msg).digest()
    return d


def parse_target(target) -> List[int]:
    if target is None:
        return None
    if isinstance(target, (bytes, bytearray)):
        assert len(target) == 32
        return list(struct.unpack(">8I", bytes(target)))
    if isinstance(target, str):
        t = target.strip().lower().replace("0x", "")
        assert len(t) == 64, "target hex must be 64 chars"
        return [int(t[i:i + 8], 16) for i in range(0, 64, 8)]
    target = [int(x) & MASK32 for x in target]
    assert len(target) == 8
    return target


# ----------------------------------------------------------------------------
# the CNF builder
# ----------------------------------------------------------------------------
class VarMap:
    def __init__(self):
        self.nvars = 0
        self.clauses: List[List[int]] = []
        self.W: Dict[int, Word] = {}
        self.a: Dict[int, Word] = {}
        self.e: Dict[int, Word] = {}
        self.stats = {"xor2": 0, "xor3": 0, "and2": 0, "maj3": 0, "ch3": 0,
                      "full_adders": 0, "half_adders": 0, "units": 0}

    # ---- variables / clauses
    def new_var(self) -> int:
        self.nvars += 1
        return self.nvars

    def add(self, clause: Sequence[Bit]) -> None:
        """Add a clause over bits; constant-true literal drops the clause,
        constant-false literals are removed."""
        out = []
        for lit in clause:
            if lit is True:
                return
            if lit is False:
                continue
            out.append(lit)
        if not out:
            raise ValueError("empty clause: constraints are contradictory")
        self.clauses.append(out)

    # ---- gates with constant folding
    @staticmethod
    def neg(x: Bit) -> Bit:
        if isinstance(x, bool):
            return not x
        return -x

    def xor2(self, x: Bit, y: Bit) -> Bit:
        if isinstance(x, bool):
            return self.neg(y) if x else y
        if isinstance(y, bool):
            return self.neg(x) if y else x
        if x == y:
            return False
        if x == -y:
            return True
        o = self.new_var()
        self.stats["xor2"] += 1
        self.clauses += [[-x, -y, -o], [x, y, -o], [x, -y, o], [-x, y, o]]
        return o

    def xor3(self, x: Bit, y: Bit, z: Bit) -> Bit:
        lits = [x, y, z]
        consts = [v for v in lits if isinstance(v, bool)]
        if consts:
            rest = [v for v in lits if not isinstance(v, bool)]
            par = sum(consts) & 1
            if len(rest) == 0:
                return bool(par)
            if len(rest) == 1:
                return self.neg(rest[0]) if par else rest[0]
            r = self.xor2(rest[0], rest[1])
            return self.neg(r) if par else r
        o = self.new_var()
        self.stats["xor3"] += 1
        # o = x ^ y ^ z : forbid the 8 assignments of odd/even parity mismatch
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    # parity of (x,y,z) true-count: literal signs positive mean "x is false" in clause
                    ntrue = (sx < 0) + (sy < 0) + (sz < 0)   # assignment: x=(sx<0), ...
                    val = ntrue & 1
                    # clause forbids assignment (x=sx<0,...) with o != val
                    self.clauses.append([sx * x, sy * y, sz * z, o if val else -o])
        return o

    def and2(self, x: Bit, y: Bit) -> Bit:
        if isinstance(x, bool):
            return y if x else False
        if isinstance(y, bool):
            return x if y else False
        if x == y:
            return x
        if x == -y:
            return False
        o = self.new_var()
        self.stats["and2"] += 1
        self.clauses += [[-x, -y, o], [x, -o], [y, -o]]
        return o

    def or2(self, x: Bit, y: Bit) -> Bit:
        return self.neg(self.and2(self.neg(x), self.neg(y)))

    def maj3(self, x: Bit, y: Bit, z: Bit) -> Bit:
        lits = [x, y, z]
        consts = [v for v in lits if isinstance(v, bool)]
        if consts:
            rest = [v for v in lits if not isinstance(v, bool)]
            if len(rest) == 0:
                return sum(consts) >= 2
            if len(rest) == 1:
                if all(consts):
                    return True
                if not any(consts):
                    return False
                return rest[0]
            # one constant
            return self.or2(rest[0], rest[1]) if consts[0] else self.and2(rest[0], rest[1])
        if x == y:
            return x
        if x == z:
            return x
        if y == z:
            return y
        if x == -y:
            return z
        if x == -z:
            return y
        if y == -z:
            return x
        o = self.new_var()
        self.stats["maj3"] += 1
        self.clauses += [[-x, -y, o], [-x, -z, o], [-y, -z, o],
                         [x, y, -o], [x, z, -o], [y, z, -o]]
        return o

    def ch3(self, x: Bit, y: Bit, z: Bit) -> Bit:
        """Ch(x,y,z) = x ? y : z."""
        if isinstance(x, bool):
            return y if x else z
        if isinstance(y, bool) and isinstance(z, bool):
            if y == z:
                return y
            return x if y else self.neg(x)
        if isinstance(y, bool):
            # x ? const : z
            return self.or2(x, z) if y else self.and2(self.neg(x), z)
        if isinstance(z, bool):
            return self.or2(self.neg(x), y) if z else self.and2(x, y)
        if y == z:
            return y
        o = self.new_var()
        self.stats["ch3"] += 1
        self.clauses += [[-x, -y, o], [-x, y, -o], [x, -z, o], [x, z, -o],
                         [-y, -z, o], [y, z, -o]]
        return o

    # ---- words
    def const_word(self, v: int) -> Word:
        return [bool((v >> i) & 1) for i in range(32)]

    def free_word(self) -> Word:
        return [self.new_var() for _ in range(32)]

    def rotr_w(self, w: Word, n: int) -> Word:
        return [w[(i + n) % 32] for i in range(32)]

    def shr_w(self, w: Word, n: int) -> Word:
        return [w[i + n] if i + n < 32 else False for i in range(32)]

    def xor3_w(self, p: Word, q: Word, r: Word) -> Word:
        return [self.xor3(p[i], q[i], r[i]) for i in range(32)]

    def Sig0(self, w: Word) -> Word:
        return self.xor3_w(self.rotr_w(w, 2), self.rotr_w(w, 13), self.rotr_w(w, 22))

    def Sig1(self, w: Word) -> Word:
        return self.xor3_w(self.rotr_w(w, 6), self.rotr_w(w, 11), self.rotr_w(w, 25))

    def sig0(self, w: Word) -> Word:
        return self.xor3_w(self.rotr_w(w, 7), self.rotr_w(w, 18), self.shr_w(w, 3))

    def sig1(self, w: Word) -> Word:
        return self.xor3_w(self.rotr_w(w, 17), self.rotr_w(w, 19), self.shr_w(w, 10))

    def Ch_w(self, x: Word, y: Word, z: Word) -> Word:
        return [self.ch3(x[i], y[i], z[i]) for i in range(32)]

    def Maj_w(self, x: Word, y: Word, z: Word) -> Word:
        return [self.maj3(x[i], y[i], z[i]) for i in range(32)]

    def add2(self, x: Word, y: Word) -> Word:
        """Ripple-carry modular addition of two words."""
        out = []
        carry: Bit = False
        for i in range(32):
            xi, yi = x[i], y[i]
            s = self.xor3(xi, yi, carry)
            if i < 31:
                c = self.maj3(xi, yi, carry)
                nconst = sum(isinstance(v, bool) for v in (xi, yi, carry))
                if nconst == 0:
                    self.stats["full_adders"] += 1
                elif nconst == 1:
                    self.stats["half_adders"] += 1
                carry = c
            out.append(s)
        return out

    def add_many(self, words: Sequence[Word]) -> Word:
        """Sum of several words; constants are merged first and added last."""
        consts = [w for w in words if all(isinstance(b, bool) for b in w)]
        rest = [w for w in words if not all(isinstance(b, bool) for b in w)]
        cval = 0
        for w in consts:
            cval = (cval + word_to_int(w)) & MASK32
        if not rest:
            return self.const_word(cval)
        acc = rest[0]
        for w in rest[1:]:
            acc = self.add2(acc, w)
        if cval:
            acc = self.add2(acc, self.const_word(cval))
        return acc

    def fix_word(self, w: Word, value: int) -> None:
        for i in range(32):
            want = bool((value >> i) & 1)
            b = w[i]
            if isinstance(b, bool):
                if b != want:
                    raise ValueError("constant word contradicts fixed value")
                continue
            self.clauses.append([b if want else -b])
            self.stats["units"] += 1

    def eq_words(self, x: Word, y: Word) -> None:
        for i in range(32):
            p, q = x[i], y[i]
            if isinstance(p, bool) and isinstance(q, bool):
                if p != q:
                    raise ValueError("constant words differ")
                continue
            if isinstance(p, bool):
                self.clauses.append([q if p else -q])
            elif isinstance(q, bool):
                self.clauses.append([p if q else -p])
            else:
                self.clauses += [[-p, q], [p, -q]]

    # ---- helpers
    def word(self, name: str, r: int) -> Word:
        return {"W": self.W, "a": self.a, "e": self.e}[name][r]

    def decode_word(self, name: str, r: int, model: Sequence[int]) -> int:
        return decode_word_bits(self.word(name, r), model)

    def decode_message(self, model: Sequence[int]) -> List[int]:
        return [self.decode_word("W", t, model) for t in range(16)]


def word_to_int(w: Word) -> int:
    v = 0
    for i in range(32):
        if w[i] is True:
            v |= 1 << i
    return v


def decode_word_bits(w: Word, model: Sequence[int]) -> int:
    """model: list of literals (pysat style) or a set/dict; a variable v is true
    iff +v is in the model."""
    if isinstance(model, dict):
        truth = model
    else:
        truth = {abs(l): (l > 0) for l in model}
    v = 0
    for i in range(32):
        b = w[i]
        if isinstance(b, bool):
            t = b
        else:
            t = truth.get(abs(b), False)
            if b < 0:
                t = not t
        if t:
            v |= 1 << i
    return v


def build(R: int, target, extra_constraints=None, adder: str = "ripple") -> Tuple[List[List[int]], VarMap]:
    """Build the CNF for: exists W0..W15 with SHA256_compress_R(IV, W) + IV == target."""
    assert 1 <= R <= 64
    assert adder == "ripple", "only the ripple-carry adder is implemented"
    vm = VarMap()

    # message words
    for t in range(16):
        vm.W[t] = vm.free_word()
    for t in range(16, R):
        vm.W[t] = vm.add_many([vm.sig1(vm.W[t - 2]), vm.W[t - 7], vm.sig0(vm.W[t - 15]), vm.W[t - 16]])

    # initial state
    for i in range(4):
        vm.a[-1 - i] = vm.const_word(IV[i])
        vm.e[-1 - i] = vm.const_word(IV[4 + i])

    # rounds
    for r in range(R):
        a1, a2, a3, a4 = vm.a[r - 1], vm.a[r - 2], vm.a[r - 3], vm.a[r - 4]
        e1, e2, e3, e4 = vm.e[r - 1], vm.e[r - 2], vm.e[r - 3], vm.e[r - 4]
        T1 = vm.add_many([e4, vm.Sig1(e1), vm.Ch_w(e1, e2, e3), vm.const_word(K[r]), vm.W[r]])
        T2 = vm.add_many([vm.Sig0(a1), vm.Maj_w(a1, a2, a3)])
        vm.a[r] = vm.add2(T1, T2)
        vm.e[r] = vm.add2(a4, T1)

    # target: digest = IV + state  ->  state = digest - IV
    tw = parse_target(target)
    if tw is not None:
        state_words = [vm.a[R - 1], vm.a[R - 2], vm.a[R - 3], vm.a[R - 4],
                       vm.e[R - 1], vm.e[R - 2], vm.e[R - 3], vm.e[R - 4]]
        for i in range(8):
            vm.fix_word(state_words[i], (tw[i] - IV[i]) & MASK32)

    # extra constraints
    for c in (extra_constraints or []):
        kind = c[0]
        if kind == "fix":
            _, name, r, value = c
            vm.fix_word(vm.word(name, r), int(value) & MASK32)
        elif kind == "eq":
            _, n1, r1, n2, r2 = c
            vm.eq_words(vm.word(n1, r1), vm.word(n2, r2))
        elif kind == "bits":
            _, name, r, bits = c
            w = vm.word(name, r)
            for i, val in bits.items():
                b = w[i]
                if isinstance(b, bool):
                    if b != bool(val):
                        raise ValueError("bit constraint contradicts constant")
                    continue
                vm.clauses.append([b if val else -b])
        elif kind == "clauses":
            for cl in c[1]:
                vm.add(cl)
        else:
            raise ValueError(f"unknown constraint kind {kind}")

    return vm.clauses, vm


# ----------------------------------------------------------------------------
# I/O and solving helpers
# ----------------------------------------------------------------------------
def write_dimacs(clauses: List[List[int]], nvars: int, path: str, comments: Sequence[str] = ()) -> None:
    with open(path, "w") as f:
        for c in comments:
            f.write(f"c {c}\n")
        f.write(f"p cnf {nvars} {len(clauses)}\n")
        f.write("\n".join(" ".join(map(str, c)) + " 0" for c in clauses))
        f.write("\n")


def solve_pysat(clauses, nvars, solver="cadical195", seed=0, options=None):
    """Solve with python-sat.  Returns the model (list of ints) or None.
    CaDiCaL is seeded through its 'seed' option; others get a shuffled
    variable numbering derived from the seed."""
    from pysat.solvers import Solver
    opts = dict(options or {})
    if solver.startswith("cadical"):
        opts.setdefault("seed", int(seed))
        s = Solver(name=solver)
        if opts:
            s.configure(opts)
        s.append_formula(clauses)
        ok = s.solve()
        model = s.get_model() if ok else None
        s.delete()
        return model
    # generic: permute variables for seeding
    import random
    rng = random.Random(seed)
    perm = list(range(1, nvars + 1))
    rng.shuffle(perm)
    fwd = {i + 1: perm[i] for i in range(nvars)}
    inv = {v: k for k, v in fwd.items()}
    cl = [[(fwd[abs(l)] if l > 0 else -fwd[abs(l)]) for l in c] for c in clauses]
    rng.shuffle(cl)
    s = Solver(name=solver)
    s.append_formula(cl)
    ok = s.solve()
    model = None
    if ok:
        m = s.get_model()
        model = [(inv[abs(l)] if l > 0 else -inv[abs(l)]) for l in m]
    s.delete()
    return model


def parse_kissat_model(text: str) -> Optional[List[int]]:
    """Parse 's SATISFIABLE' / 'v ...' lines from kissat's stdout."""
    if "s SATISFIABLE" not in text:
        return None
    model = []
    for line in text.splitlines():
        if line.startswith("v "):
            for tok in line[2:].split():
                v = int(tok)
                if v == 0:
                    break
                model.append(v)
    return model


def solve_kissat(dimacs_path: str, kissat_bin: str, seed: int = 0, timeout: Optional[float] = None,
                 extra_args: Sequence[str] = ()) -> Tuple[str, Optional[List[int]], str]:
    """Run kissat on a DIMACS file.  Returns (status, model, stdout_tail)."""
    cmd = [kissat_bin, f"--seed={seed}", "-q", *extra_args, dimacs_path]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", None, ""
    out = p.stdout
    if "s SATISFIABLE" in out:
        return "sat", parse_kissat_model(out), out[-200:]
    if "s UNSATISFIABLE" in out:
        return "unsat", None, out[-200:]
    return "unknown", None, out[-500:] + p.stderr[-500:]


def verify_model(vm: VarMap, model, R: int, target) -> Tuple[bool, List[int], List[int]]:
    words = vm.decode_message(model)
    got = sha256_reduced_raw_block(words, R)
    tw = parse_target(target)
    return got == tw, words, got


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------
def _selftest():
    import random
    import time
    rng = random.Random(1)
    # 1. planted message, R small: solve and verify
    for R in (4, 8, 12):
        msg = bytes(rng.getrandbits(8) for _ in range(55))
        tgt = digest_of_message(msg, R)
        t0 = time.time()
        clauses, vm = build(R, tgt)
        tb = time.time() - t0
        t0 = time.time()
        model = solve_pysat(clauses, vm.nvars, "cadical195", seed=1)
        ts = time.time() - t0
        assert model is not None, "planted instance must be SAT"
        ok, words, got = verify_model(vm, model, R, tgt)
        assert ok, "decoded preimage does not verify"
        print(f"selftest R={R}: vars={vm.nvars} clauses={len(clauses)} build={tb:.2f}s solve={ts:.2f}s OK")
    # 2. consistency: fix all 16 W words to the planted message, unit-propagate the digest
    R = 19
    msg = bytes(rng.getrandbits(8) for _ in range(55))
    words = pad_one_block(msg)
    tgt = digest_of_message(msg, R)
    clauses, vm = build(R, tgt, [("fix", "W", t, words[t]) for t in range(16)])
    model = solve_pysat(clauses, vm.nvars, "cadical195", seed=1)
    assert model is not None
    ok, w2, _ = verify_model(vm, model, R, tgt)
    assert ok and w2 == words
    # 3. wrong target with all W fixed must be UNSAT
    bad = bytearray(tgt); bad[0] ^= 1
    clauses, vm = build(R, bytes(bad), [("fix", "W", t, words[t]) for t in range(16)])
    assert solve_pysat(clauses, vm.nvars, "cadical195", seed=1) is None
    # 4. state words decode correctly against submask_family.forward convention
    from pysat.solvers import Solver
    clauses, vm = build(R, None, [("fix", "W", t, words[t]) for t in range(16)])
    s = Solver(name="cadical195"); s.append_formula(clauses); assert s.solve(); m = s.get_model(); s.delete()
    # recompute a_r, e_r by reference
    a = {-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]}
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    Wf = list(words)
    for t in range(16, R):
        s0 = rotr(Wf[t - 15], 7) ^ rotr(Wf[t - 15], 18) ^ (Wf[t - 15] >> 3)
        s1 = rotr(Wf[t - 2], 17) ^ rotr(Wf[t - 2], 19) ^ (Wf[t - 2] >> 10)
        Wf.append((s1 + Wf[t - 7] + s0 + Wf[t - 16]) & MASK32)
    for r in range(R):
        S1 = rotr(e[r - 1], 6) ^ rotr(e[r - 1], 11) ^ rotr(e[r - 1], 25)
        ch = (e[r - 1] & e[r - 2]) ^ (~e[r - 1] & e[r - 3]) & MASK32
        T1 = (e[r - 4] + S1 + ch + K[r] + Wf[r]) & MASK32
        S0 = rotr(a[r - 1], 2) ^ rotr(a[r - 1], 13) ^ rotr(a[r - 1], 22)
        mj = (a[r - 1] & a[r - 2]) ^ (a[r - 1] & a[r - 3]) ^ (a[r - 2] & a[r - 3])
        a[r] = (T1 + S0 + mj) & MASK32
        e[r] = (a[r - 4] + T1) & MASK32
    for r in range(R):
        assert vm.decode_word("a", r, m) == a[r], f"a[{r}] mismatch"
        assert vm.decode_word("e", r, m) == e[r], f"e[{r}] mismatch"
        assert vm.decode_word("W", r, m) == Wf[r], f"W[{r}] mismatch"
    print("selftest: state-word decoding matches reference for all r; UNSAT check OK")
    print("stats R=19:", vm.stats, "vars", vm.nvars, "clauses", len(clauses))


if __name__ == "__main__":
    _selftest()
