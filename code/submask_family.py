#!/usr/bin/env python3
"""The submask context family for the reduced-round SHA-256 preimage attack.

The attack of the main paper chooses the context words a4..a10 at random, and
then pays for two things: a fixed-point iteration over C1 and C2 that converges
only rarely, and an exact 32-bit consistency check on the provisional value of
W9.  Both costs disappear on a large subfamily of contexts.

THE FAMILY.  Choose the context so that

    a4 = a5 = v   (any v),      e8 = 0xFFFFFFFF,      e9 = 0xFFFFFFFF

which is achieved by picking a6, a7 freely and then solving the two round
equations for a8 and a9:

    a8 = -1 - a4 + Sigma0(a7) + Maj(a7, a6, a5)
    a9 = -1 - a5 + Sigma0(a8) + Maj(a8, a7, a6)

leaving a6, a7, a10 (and a11 at R = 20) free: 2^128 contexts at R = 19.

CONSEQUENCE 1 -- the iteration disappears.  The only a3-dependence of the C1
target is

    D(a3) = (a6 - Sigma0(a5) - Maj(a5, a4, a3)) + Ch(e9, e8, a3 + T1_7).

a5 = a4 collapses Maj(a5, a4, a3) to a4, and e9 = 0xFFFFFFFF collapses
Ch(e9, e8, .) to e8, so D is a constant.  C1 then yields a2 without reference to
a3, and C2 yields a3 from a2: C0 -> a1, C1 -> a2, C2 -> a3 is a triangular
solve, three table lookups per swept a0 and no iteration.  Both lookups hit
together for 0.634^2 = 0.402 of the C0 survivors.

CONSEQUENCE 2 -- the 32-bit check becomes a bitwise one.  This one needs only
a4 = v and e8 = 0xFFFFFFFF; it does not need a5 = a4.  With e8 = 0xFFFFFFFF the
choice function in W9 collapses, Ch(e8, e7, e6) = e7 = a3 + T1_7, and
T1_5 = a5 - Sigma0(a4) - Maj(a4, a3, a2).  Writing W9hat for the provisional
value at a2 = a3 = 0, everything except the Maj cancels and

    eps = W9conv - W9hat = Maj(v, a3, a2) - a3        exactly, mod 2^32.

The subtraction is borrow-free: Maj(v, a3, a2) = a3 - P + Q with
P = a3 & ~a2 & ~v and Q = a2 & ~a3 & v, where P is a submask of a3 and Q is
disjoint from it.  So eps = 0 is the per-bit condition Maj(v, a3, a2) = a3,
equivalently

    (a2 ^ a3) & (a3 ^ v) == 0,

which for uniform independent a2, a3 holds for exactly 3^32 of the 2^64 pairs,
i.e. with probability (3/4)^32 = 1.004524e-4, against 2^-32 = 2.33e-10 for an
unstructured 32-bit match.  At v = 0 it reads a3 & ~a2 = 0, so a3 is a submask
of a2; at v = 0xFFFFFFFF it reads a2 & ~a3 = 0.

WHICH LEVER IS WORTH WHAT.  The two effects are separable and unequal.  The
triangular solve is the larger one: it enumerates the C1/C2 solutions directly,
0.254 per swept a0, where the published fixed-point iteration finds about
1.0e-5 per swept a0, a factor of about 2^15.  The bitwise collapse is worth a
further factor of about 429, i.e. 2^8.7.  Together they account for the
measured 2^23 or so.

Measured cost at R = 19: about 39,000 swept a0 per preimage (2^15.26, pooled
over 270 million swept a0, 24 contexts, 24 targets and 12 values of v), against
about 2^38.3 for the random-context attack of the main paper.  Note two things
about that ratio.  It is a MARGINAL cost: both attacks presuppose the same
one-time 16 GB table, which takes about 215 core-seconds to build, so a
cold-start figure for a single preimage is dominated by the table.  And the
2^38.3 baseline inherits the main paper's own uncertainty on its success rate
(95% interval 2.6e-3 to 3.7e-2), so the honest speedup is about 2^23 give or
take 1.5 bits.

ATTRIBUTION.  Neither degeneracy is new on its own.  The absorption property of
the choice function -- Ch(0xFFFFFFFF, y, z) = y, so one argument disappears --
is used for "message stealing" by Aoki, Guo, Matusiewicz, Sasaki and Wang,
Preimages for Step-Reduced SHA-2, ASIACRYPT 2009.  The majority degeneracy,
Maj(x, x, y) = x, is likewise standard and is formalised as an absorption rule
by Guo, Li, Liu, Wang and Zhang, Dual-Syncopation Meet-in-the-Middle Attacks,
EUROCRYPT 2026.  The companion note of this project already records both levers
and the identical e8 = 0xFFFFFFFF-via-a8 construction, in its section "A Context
Family that Removes a_2 from C0", and dismisses it there because the second
condition it used tied a context word to an unknown.  What is new here is
tying two CONTEXT words together instead (a4 = a5), which costs nothing per
candidate, and the observation that the same choice collapses the final 32-bit
modular consistency check into a bitwise one.  We have found no precedent for
that second effect.

Usage:
    python3 submask_family.py                      # verify the algebra (CPU, no table)
    python3 submask_family.py --attack --hash <64 hex> [--rounds 19] [--a0 N]
    python3 submask_family.py --measure [--rounds 19] [--targets N] [--a0 N]

--attack and --measure need the sigma0(u)-u table on disk; build it with
build_sigma0_table.py and pass --table or set SIGMA0_TABLE.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys
import time

import numpy as np

M = 0xFFFFFFFF
U32 = np.uint32
MISS = U32(M)
ZERO = U32(0)

K = [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
     0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
     0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
     0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
     0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
     0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
     0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
     0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
     0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
     0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
     0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]
IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]


# --------------------------------------------------------------------------
# primitives (accept python ints and uint32 numpy arrays alike)
# --------------------------------------------------------------------------
def rotr(x, n):
    if isinstance(x, np.ndarray):
        return ((x >> U32(n)) | (x << U32(32 - n))) & MISS
    return ((x >> n) | (x << (32 - n))) & M


def S0(x):
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)


def S1(x):
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)


def s0(x):
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> (U32(3) if isinstance(x, np.ndarray) else 3))


def s1(x):
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> (U32(10) if isinstance(x, np.ndarray) else 10))


def Ch(e, f, g):
    if isinstance(e, np.ndarray):
        return (e & f) ^ (~e & g)
    return ((e & f) ^ ((~e) & g)) & M


def Maj(a, b, c_):
    return (a & b) ^ (a & c_) ^ (b & c_)


def T2(a, b, c_):
    return (S0(a) + Maj(a, b, c_)) & (MISS if isinstance(a, np.ndarray) else M)


def u32(x):
    """python int -> uint32 scalar, reduced first so numpy never sees an overflow."""
    return U32(x & M)


# --------------------------------------------------------------------------
# reference reduced-round compression function, in the a/e representation
# --------------------------------------------------------------------------
def forward(W, R):
    a = {-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]}
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    Wf = list(W)
    for t in range(16, R):
        Wf.append((s1(Wf[t - 2]) + Wf[t - 7] + s0(Wf[t - 15]) + Wf[t - 16]) & M)
    for r in range(R):
        T1 = (e[r - 4] + S1(e[r - 1]) + Ch(e[r - 1], e[r - 2], e[r - 3]) + K[r] + Wf[r]) & M
        a[r] = (T1 + T2(a[r - 1], a[r - 2], a[r - 3])) & M
        e[r] = (a[r - 4] + T1) & M
    return a, e, Wf


def digest(W, R):
    a, e, _ = forward(W, R)
    s = [a[R - 1], a[R - 2], a[R - 3], a[R - 4], e[R - 1], e[R - 2], e[R - 3], e[R - 4]]
    return b"".join(struct.pack(">I", (x + y) & M) for x, y in zip(s, IV))


def recover_W(a, e, r):
    return (a[r] - T2(a[r - 1], a[r - 2], a[r - 3]) - e[r - 4] - S1(e[r - 1])
            - Ch(e[r - 1], e[r - 2], e[r - 3]) - K[r]) & M


def backward_chain(h, R):
    """Recover a_{R-8}..a_{R-1} and e_{R-4}..e_{R-1} from the R-round digest."""
    s = [(struct.unpack(">I", h[4 * i:4 * i + 4])[0] - IV[i]) & M for i in range(8)]
    a = {R - 1: s[0], R - 2: s[1], R - 3: s[2], R - 4: s[3]}
    e = {R - 1: s[4], R - 2: s[5], R - 3: s[6], R - 4: s[7]}
    for r in (R - 1, R - 2, R - 3, R - 4):
        T1 = (a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M
        a[r - 4] = (e[r] - T1) & M
    return a, e


# --------------------------------------------------------------------------
# the family
# --------------------------------------------------------------------------
def make_context(rng, R, v=None):
    """A context in the submask family: a4 = a5 = v, e8 = e9 = 0xFFFFFFFF."""
    if v is None:
        v = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    ctx = {4: v & M, 5: v & M}
    ctx[6] = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    ctx[7] = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    a4, a5, a6, a7 = ctx[4], ctx[5], ctx[6], ctx[7]
    ctx[8] = (M - a4 + S0(a7) + Maj(a7, a6, a5)) & M       # e8 = a4 + a8 - T2(a7,a6,a5) = -1
    a8 = ctx[8]
    ctx[9] = (M - a5 + S0(a8) + Maj(a8, a7, a6)) & M       # e9 = a5 + a9 - T2(a8,a7,a6) = -1
    ctx[10] = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    if R >= 20:
        ctx[11] = int(rng.integers(0, 1 << 32, dtype=np.uint64))
    return ctx


def verify_algebra(trials=2000, seed=20260903):
    """Check both collapses numerically, against the unrestricted formulas.

    Needs no table and no target: it perturbs a full state directly.
    """
    rng = np.random.default_rng(seed)
    print("Reference model against hashlib (64 rounds):", end=" ")
    blk = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
    pad = blk + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
    Wt = [struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)]
    ok_ref = digest(Wt, 64) == hashlib.sha256(blk).digest()
    print("OK" if ok_ref else "FAIL")

    bad_D = bad_eps = 0
    for _ in range(trials):
        v = int(rng.integers(0, 1 << 32, dtype=np.uint64))
        ctx = make_context(rng, 19, v)
        a4, a5, a6, a7, a8, a9 = (ctx[i] for i in range(4, 10))
        e8 = (a4 + a8 - T2(a7, a6, a5)) & M
        e9 = (a5 + a9 - T2(a8, a7, a6)) & M
        assert e8 == M and e9 == M, "context construction is wrong"
        T1_7 = (a7 - T2(a6, a5, a4)) & M
        a2 = int(rng.integers(0, 1 << 32, dtype=np.uint64))
        a3 = int(rng.integers(0, 1 << 32, dtype=np.uint64))

        # (1) the a3-dependent part of the C1 target must not move with a3
        def D(x):
            e7 = (x + T1_7) & M
            return ((a6 - S0(a5) - Maj(a5, a4, x)) + Ch(e9, e8, e7)) & M
        if D(a3) != D(0):
            bad_D += 1

        # (2) eps from the unrestricted formula, versus the collapsed one
        def W9conv(x2, x3):
            T1_5 = (a5 - S0(a4) - Maj(a4, x3, x2)) & M
            e7 = (x3 + T1_7) & M
            e6 = (x2 + a6 - S0(a5) - Maj(a5, a4, x3)) & M
            return (-T1_5 - S1(e8) - Ch(e8, e7, e6)) & M     # W9base cancels in eps
        eps_general = (W9conv(a2, a3) - W9conv(0, 0)) & M
        eps_closed = (Maj(v, a3, a2) - a3) & M
        if eps_general != eps_closed:
            bad_eps += 1
        # borrow-free decomposition and the compact bitwise form
        P = a3 & ~a2 & ~v & M
        Q = a2 & ~a3 & v & M
        if (P & Q) or (P & a3) != P or (Q & a3) or eps_closed != ((Q - P) & M):
            bad_eps += 1
        if (eps_closed == 0) != (((a2 ^ a3) & (a3 ^ v) & M) == 0):
            bad_eps += 1

    print(f"C1 target independent of a3 in the family: "
          f"{trials - bad_D}/{trials} {'OK' if bad_D == 0 else 'FAIL'}")
    print(f"eps == Maj(v,a3,a2) - a3 exactly:          "
          f"{trials - bad_eps}/{trials} {'OK' if bad_eps == 0 else 'FAIL'}")

    # (3) the collapsed condition is bitwise, so its density is (3/4)^32
    n = 1 << 22
    a2 = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(U32)
    a3 = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(U32)
    for v in (0, M, 0x6a09e667):
        hits = int((Maj(U32(v), a3, a2) == a3).sum())
        print(f"P(eps==0) at v=0x{v:08x}: {hits/n:.3e} over {n:,} random (a2,a3) "
              f"[predicted (3/4)^32 = {0.75**32:.4e}, uniform 2^-32 = {2**-32:.3e}]")
    # (4) the backward chain really inverts the digest: plant a message, hash it,
    # and check the chain recovers the state words the forward pass produced.
    # (Comparing the chain's own e_r against itself is a tautology; this is not.)
    bad_bc = 0
    for R in (19, 20):
        for _ in range(50):
            m = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
            pad = m + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
            Wt = [struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)]
            a_true, e_true, _ = forward(Wt, R)
            ab, eb = backward_chain(digest(Wt, R), R)
            for r in range(R - 8, R):
                if ab[r] != a_true[r]:
                    bad_bc += 1
            for r in range(R - 4, R):
                if eb[r] != e_true[r]:
                    bad_bc += 1
    print(f"backward chain recovers the true state words:   "
          f"{'OK' if bad_bc == 0 else str(bad_bc) + ' MISMATCHES'}")

    ok = ok_ref and bad_D == 0 and bad_eps == 0 and bad_bc == 0
    print("\n" + ("ALL ALGEBRAIC CHECKS PASS" if ok else "FAILURES PRESENT"))
    return ok


# --------------------------------------------------------------------------
# the attack
# --------------------------------------------------------------------------
def load_table(path):
    if path is None:
        path = os.environ.get("SIGMA0_TABLE", "sigma0_u_table.npy")
    if not os.path.exists(path):
        sys.exit(f"sigma0 table not found at {path}; build it with build_sigma0_table.py "
                 f"or set SIGMA0_TABLE")
    return np.load(path, mmap_mode="r").view(np.uint32)


def attack_context(tbl, h, ctx, R, n_a0, rng, collect=None):
    """Sweep n_a0 random a0 in one context.  Returns counters and any preimages."""
    ab, eb = backward_chain(h, R)
    a = dict(ab)
    a.update(ctx)
    a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(8, R):
        e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M

    a4, a5, a6, a7, a8, a9, a10 = (a[i] for i in range(4, 11))
    am1, am2, am3, am4 = a[-1], a[-2], a[-3], a[-4]
    em1, em2, em3, em4 = e[-1], e[-2], e[-3], e[-4]
    e8, e9, e10 = e[8], e[9], e[10]
    assert e8 == M and e9 == M and a4 == a5, "context is not in the family"
    # note: comparing the recomputed e_r against the backward chain's would be a
    # tautology (the chain derives a_{r-4} from that same relation), so the real
    # check on the chain is the round-trip in self_test().
    T1_7 = (a7 - T2(a6, a5, a4)) & M
    c6 = (a6 - S0(a5)) & M
    W9base = ((a9 - T2(a8, a7, a6)) - K[9]) & M
    W10base = ((a10 - T2(a9, a8, a7)) - S1(e9) - K[10]) & M
    W11base = ((a[11] - T2(a10, a9, a8)) - S1(e10) - K[11]) & M
    Wr = {r: recover_W(a, e, r) for r in range(12, R)}
    K0p = (Wr[16] - s1(Wr[14])) & M
    K1p = (Wr[17] - s1(Wr[15])) & M
    K2p = (Wr[18] - s1(Wr[16])) & M
    K3p = ((Wr[19] - s1(Wr[17]) - Wr[12]) & M) if R >= 20 else None
    T2iv = T2(am1, am2, am3)
    C0c = (-T2iv - em4 - S1(em1) - Ch(em1, em2, em3) - K[0]) & M
    Ce0 = (am4 - T2iv) & M
    # provisional W9 at a2 = a3 = 0, and the constant C1 offset
    W9hat = (W9base - (a5 - S0(a4) - Maj(a4, 0, 0)) - S1(e8)
             - Ch(e8, T1_7, (c6 - Maj(a5, a4, 0)) & M)) & M
    D = ((a6 - S0(a5) - Maj(a5, a4, 0)) + Ch(e9, e8, T1_7)) & M
    # fold the per-context constants in python ints, so numpy never sees a
    # uint32 scalar sum that overflows (correct either way, but it warns)
    KC0 = (K0p - W9hat) & M
    KC1 = (K1p - W10base + D) & M
    KC2 = (K2p - W11base + T1_7 + Ch(e10, e9, e8)) & M

    st = dict(a0=0, surv=0, sol=0, eps0=0, c3=0, ver=0)
    out = []
    B = 1 << 20
    done = 0
    while done < n_a0:
        b = min(B, n_a0 - done)
        done += b
        st['a0'] += b
        A0 = rng.integers(0, 1 << 32, size=b, dtype=np.uint64).astype(U32)
        E0 = (A0 + u32(Ce0)) & MISS
        W0 = (A0 + u32(C0c)) & MISS
        G = (-(S0(A0) + Maj(A0, u32(am1), u32(am2))) - u32(em3) - S1(E0)
             - Ch(E0, u32(em1), u32(em2)) - u32(K[1])) & MISS      # W1 - a1
        W1 = np.asarray(tbl[(u32(KC0) - W0 - G) & MISS])
        keep = W1 != MISS
        A0, E0, W0, G, W1 = (x[keep] for x in (A0, E0, W0, G, W1))
        st['surv'] += A0.size
        A1 = (W1 - G) & MISS
        E1 = (u32(am3) + A1 - (S0(A0) + Maj(A0, u32(am1), u32(am2)))) & MISS
        F12 = (-(S0(A1) + Maj(A1, A0, u32(am1))) - u32(em2) - S1(E1)
               - Ch(E1, E0, u32(em1)) - u32(K[2])) & MISS          # W2 - a2
        W2 = np.asarray(tbl[(u32(KC1) - W1 - F12) & MISS])
        keep = W2 != MISS
        A0, A1, E0, E1, W1, F12, W2 = (x[keep] for x in (A0, A1, E0, E1, W1, F12, W2))
        A2 = (W2 - F12) & MISS
        e2 = (u32(am2) + A2 - (S0(A1) + Maj(A1, A0, u32(am1)))) & MISS
        F23 = (-(S0(A2) + Maj(A2, A1, A0)) - u32(em1) - S1(e2)
               - Ch(e2, E1, E0) - u32(K[3])) & MISS                # W3 - a3
        W3 = np.asarray(tbl[(u32(KC2) - W2 - F23) & MISS])
        keep = W3 != MISS
        A0, A1, A2, E0, E1, e2, F23, W3 = (x[keep] for x in
                                           (A0, A1, A2, E0, E1, e2, F23, W3))
        A3 = (W3 - F23) & MISS
        st['sol'] += A0.size
        if A0.size == 0:
            continue
        hit = Maj(u32(a4), A3, A2) == A3            # eps == 0, the collapsed form
        st['eps0'] += int(hit.sum())
        idx = np.nonzero(hit)[0]
        if R >= 20 and idx.size:
            e3 = (u32(am1) + A3[idx] - (S0(A2[idx]) + Maj(A2[idx], A1[idx], A0[idx]))) & MISS
            W4 = (u32(a4) - (S0(A3[idx]) + Maj(A3[idx], A2[idx], A1[idx])) - E0[idx]
                  - S1(e3) - Ch(e3, e2[idx], E1[idx]) - u32(K[4])) & MISS
            c3 = (s0(W4) + W3[idx] - u32(K3p)) & MISS
            st['c3'] += int((c3 == ZERO).sum())
            idx = idx[np.nonzero(c3 == ZERO)[0]]
        for j in idx:
            aa = dict(a)
            aa.update({0: int(A0[j]), 1: int(A1[j]), 2: int(A2[j]), 3: int(A3[j])})
            ee = dict(e)
            for rr in range(R):
                ee[rr] = (aa[rr - 4] + aa[rr] - T2(aa[rr - 1], aa[rr - 2], aa[rr - 3])) & M
            Wm = [recover_W(aa, ee, rr) for rr in range(16)]
            if digest(Wm, R) == h:                  # independent forward check
                st['ver'] += 1
                if collect is not None and len(collect) < collect.maxlen:
                    collect.append(Wm)
                out.append(Wm)
    return st, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--hash", default=None, help="64-hex target digest")
    ap.add_argument("--rounds", type=int, default=19)
    ap.add_argument("--a0", type=int, default=1 << 22, help="swept a0 per context")
    ap.add_argument("--contexts", type=int, default=1)
    ap.add_argument("--targets", type=int, default=4, help="--measure only")
    ap.add_argument("--table", default=None)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--trials", type=int, default=2000)
    args = ap.parse_args()

    if not args.attack and not args.measure:
        sys.exit(0 if verify_algebra(args.trials, args.seed) else 1)

    tbl = load_table(args.table)
    rng = np.random.default_rng(args.seed)
    R = args.rounds

    if args.attack:
        if not args.hash:
            ap.error("--attack needs --hash")
        h = bytes.fromhex(args.hash.strip())
        if len(h) != 32:
            ap.error("--hash must be 64 hex characters")
        t0 = time.time()
        tot = dict(a0=0, sol=0, eps0=0, ver=0)
        for ci in range(args.contexts):
            st, out = attack_context(tbl, h, make_context(rng, R), R, args.a0, rng)
            for k in tot:
                tot[k] += st[k]
            for Wm in out:
                print("PREIMAGE " + " ".join(f"{w:08x}" for w in Wm), flush=True)
            print(f"ctx {ci}: a0 {st['a0']:,}  solutions {st['sol']:,}  "
                  f"eps==0 {st['eps0']:,}  verified {st['ver']}", flush=True)
        el = time.time() - t0
        print(f"\n{tot['ver']} verified {R}-round preimages from {tot['a0']:,} swept a0 "
              f"in {el:.1f}s")
        if tot['ver']:
            print(f"  {tot['a0'] // tot['ver']:,} swept a0 per preimage, {el/tot['ver']:.3f}s each")
        return

    # --measure
    tot = dict(a0=0, surv=0, sol=0, eps0=0, c3=0, ver=0)
    t0 = time.time()
    for ti in range(args.targets):
        msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
        pad = msg + b"\x80" + b"\x00" * (56 - 1 - 55) + struct.pack(">Q", 55 * 8)
        h = digest([struct.unpack(">I", pad[4 * i:4 * i + 4])[0] for i in range(16)], R)
        st, _ = attack_context(tbl, h, make_context(rng, R), R, args.a0, rng)
        for k in tot:
            tot[k] += st[k]
        print(f"target {ti} {h.hex()[:16]}...: solutions {st['sol']:,} "
              f"({st['sol']/max(st['surv'],1):.4f}/survivor) eps==0 {st['eps0']:,} "
              f"verified {st['ver']}", flush=True)
    el = time.time() - t0
    sol = max(tot['sol'], 1)
    print(f"\nR={R}, {args.targets} targets, {tot['a0']:,} swept a0, {el:.1f}s")
    print(f"  triangular solutions {tot['sol']:,}  = {tot['sol']/max(tot['surv'],1):.4f} "
          f"per C0 survivor (predicted 0.634^2 = 0.4019)")
    print(f"  eps == 0             {tot['eps0']:,}  = {tot['eps0']/sol:.4e} per solution "
          f"(predicted (3/4)^32 = {0.75**32:.4e}, uniform {2**-32:.3e})")
    if R >= 20:
        print(f"  C3 == 0              {tot['c3']:,}")
    print(f"  verified preimages   {tot['ver']:,}")
    if tot['ver']:
        print(f"  cost: {tot['a0']/tot['ver']:,.0f} swept a0 per preimage "
              f"= 2^{np.log2(tot['a0']/tot['ver']):.1f}")


if __name__ == "__main__":
    main()
