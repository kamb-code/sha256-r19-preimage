#!/usr/bin/env python3
"""ANGLE B: frames parameterised by MESSAGE words rather than state words.

Five exact/numerical measurements, all CPU, seconds:

  1. FRAME EQUIVALENCE.  The map (W_0..W_15) -> (a_0..a_15) is a prefix-
     triangular bijection.  Verified both directions.  Consequence: fixing a
     PREFIX of W is the same as fixing a prefix of a, so every "W-coordinate
     frame" with a prefix context is a translate of the published a-frame.

  2. WHAT THE DIGEST FIXES.  With the digest (a_12..a_19) and a context
     a_4..a_11, the words W_12..W_19 are determined and independent of the
     unknowns a_0..a_3.  Four of them (W_16..W_19) are also determined by the
     schedule -> 4 constraints.  So: 8 message words determined, of which 4
     are over-determined.  12 free words vs 4 constraints -- IDENTICAL count
     to the a-domain, because it IS the a-domain.

  3. LINEAR ENTRY CENSUS, W-domain.  The a-domain absorber exists because
     a_k enters exactly two recovered message words linearly, W_k with +1 and
     W_{k+8} with -1, and the schedule pairs W_{t-15} (under sigma0) with
     W_{t-7} (linear), which are exactly 8 apart.  We census the analogous
     structure for W_i: coefficient of W_i in each of the 8 digest residuals,
     and in the schedule.

  4. EDGE WEIGHTS, W-domain.  16 message words x 8 digest residuals at
     R = 19, 20, 21: flip one random bit of W_i, Hamming weight of the change
     in residual j.  Absorption needs <= 2 bits (mild) or the sigma0(u)-u
     self-edge shape (~8 bits with an exact +1/-1 linear pair).

  5. ADDITIVE SEPARABILITY.  A word W_i is absorbable by residual j only if
     R_j = F(W_i) + G(rest) for some one-input F (F may depend on the context
     -- that would only need a per-context 2^32 table).  Necessary and
     testable exactly: the discrete derivative R_j(x+d) - R_j(x) must not
     depend on the other message words.  Positive control: the a-domain
     C_0 / a_1 absorber, which must PASS.
"""
import sys, struct, random
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, forward,
                            digest, recover_W, backward_chain)


def popcount(x):
    return bin(x & M).count("1")


def state_from_W(W16, R):
    """Full a/e state and expanded schedule from 16 message words."""
    a, e, Wf = forward(W16, R)
    return a, e, Wf


def digest_words(W16, R):
    a, e, _ = state_from_W(W16, R)
    return [a[R-1], a[R-2], a[R-3], a[R-4], e[R-1], e[R-2], e[R-3], e[R-4]]


def residuals(W16, R, tgt):
    return [(x - y) & M for x, y in zip(digest_words(W16, R), tgt)]


def a_to_W(a15, R=16):
    """Invert: given a_0..a_15 (list), recover W_0..W_15."""
    a = {-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]}
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(16):
        a[r] = a15[r]
        e[r] = (a[r-4] + a[r] - T2(a[r-1], a[r-2], a[r-3])) & M
    return [recover_W(a, e, r) for r in range(16)]


rng = random.Random(20260908)
RW = lambda: rng.getrandbits(32)

print("=" * 78)
print("1. PREFIX-TRIANGULAR BIJECTION  W_0..W_15  <->  a_0..a_15")
print("=" * 78)
bad = 0
prefix_bad = 0
for _ in range(300):
    W = [RW() for _ in range(16)]
    a, e, _ = forward(W, 16)
    a15 = [a[r] for r in range(16)]
    if a_to_W(a15) != W:
        bad += 1
    # prefix property: changing W_k..W_15 leaves a_0..a_{k-1} fixed and v.v.
    k = rng.randrange(1, 16)
    W2 = W[:k] + [RW() for _ in range(16 - k)]
    a2, _, _ = forward(W2, 16)
    if [a2[r] for r in range(k)] != [a[r] for r in range(k)]:
        prefix_bad += 1
print(f"  round-trip W -> a -> W exact          : {300-bad}/300 "
      f"{'OK' if bad==0 else 'FAIL'}")
print(f"  a_0..a_(k-1) depends only on W_0..W_(k-1): {300-prefix_bad}/300 "
      f"{'OK' if prefix_bad==0 else 'FAIL'}")
print("  => the two parameterisations are related by a PREFIX-triangular")
print("     bijection.  Fixing a prefix of W == fixing a prefix of a.")
print("     Every W-coordinate frame with a prefix context is therefore a")
print("     translate of the published a-frame (already priced at 45.3-45.4).")

print()
print("=" * 78)
print("2. WHAT THE DIGEST FIXES IN THE MESSAGE DOMAIN")
print("=" * 78)
for R in (19, 20, 21):
    # random target from a real message
    Wt = [RW() for _ in range(16)]
    h = digest(Wt, R)
    ab, eb = backward_chain(h, R)
    dep = {r: set() for r in range(R)}
    for trial in range(60):
        base = {i: RW() for i in range(R - 8)}          # free a-words a_0..a_{R-9}
        def build(free):
            a = dict(ab); a.update(free)
            a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
            e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
            for r in range(R):
                e[r] = (a[r-4] + a[r] - T2(a[r-1], a[r-2], a[r-3])) & M
            return {r: recover_W(a, e, r) for r in range(R)}
        W0 = build(base)
        for k in range(R - 8):
            f2 = dict(base); f2[k] = RW()
            W1 = build(f2)
            for r in range(R):
                if W1[r] != W0[r]:
                    dep[r].add(k)
    unk = {0, 1, 2, 3}
    det = [r for r in range(R) if not (dep[r] & unk)]
    print(f"  R={R}: message words determined by (digest + context a_4..a_{R-9}), "
          f"i.e. free of a_0..a_3: {det}")
    print(f"        of these, {[r for r in det if r >= 16]} are ALSO determined by "
          f"the schedule  ->  {len([r for r in det if r>=16])} constraints")
    nfree = R - 8
    print(f"        free a-words {nfree}, constraints {R-16};  "
          f"free W-words 16, digest constraints 8;  "
          f"solution dim {nfree-(R-16)} == {16-8} words")

print()
print("=" * 78)
print("3. LINEAR ENTRY CENSUS")
print("=" * 78)
print("  (a) a-domain control: coefficient of a_k in each recovered W_r")
R = 20
Wt = [RW() for _ in range(16)]
h = digest(Wt, R)
ab, eb = backward_chain(h, R)
def build_a(free, R, ab):
    a = dict(ab); a.update(free)
    a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(R):
        e[r] = (a[r-4] + a[r] - T2(a[r-1], a[r-2], a[r-3])) & M
    return {r: recover_W(a, e, r) for r in range(R)}, a, e
for k in (1, 2, 3, 4):
    lin = []
    for r in range(R):
        ok = True
        for _ in range(12):
            base = {i: RW() for i in range(R - 8)}
            d = RW()
            W0, _, _ = build_a(base, R, ab)
            f2 = dict(base); f2[k] = (base[k] + d) & M
            W1, _, _ = build_a(f2, R, ab)
            dd = (W1[r] - W0[r]) & M
            if dd == d:
                c = +1
            elif dd == (-d) & M:
                c = -1
            else:
                ok = False; break
        if ok:
            lin.append((r, c))
    print(f"    a_{k}: linear in W_r for r,coef = {lin}")

print("  (b) W-domain: coefficient of W_i in each of the 8 digest residuals")
tgt = digest_words([RW() for _ in range(16)], R)
anylin = 0
for i in range(16):
    lin = []
    for j in range(8):
        ok = True
        for _ in range(8):
            W = [RW() for _ in range(16)]
            d = RW()
            r0 = residuals(W, R, tgt)[j]
            W2 = list(W); W2[i] = (W[i] + d) & M
            r1 = residuals(W2, R, tgt)[j]
            dd = (r1 - r0) & M
            if dd == d: c = +1
            elif dd == (-d) & M: c = -1
            else:
                ok = False; break
        if ok:
            lin.append((j, c)); anylin += 1
    if lin:
        print(f"    W_{i}: linear in residual {lin}")
print(f"    total (W_i, residual) pairs with an exact +-1 linear entry: {anylin}")
print("  (c) W-domain schedule: W_i enters W_{i+7} and W_{i+16} with +1 (SAME sign),")
print("      W_{i+2} under sigma1 and W_{i+15} under sigma0.  There is no")
print("      opposite-sign pair, so no sigma0(u)-u / sigma1(u)-u atom forms.")
# verify (c)
sc = []
for _ in range(20):
    W = [RW() for _ in range(34)]
    for t in range(16, 34):
        W[t] = (s1(W[t-2]) + W[t-7] + s0(W[t-15]) + W[t-16]) & M
    i = rng.randrange(9, 16)          # need i+7 >= 16 for the +1 entry to be visible
    d = RW()
    W2 = list(W); W2[i] = (W[i] + d) & M
    for t in range(16, 34):
        W2[t] = (s1(W2[t-2]) + W2[t-7] + s0(W2[t-15]) + W2[t-16]) & M
    sc.append(((W2[i+7]-W[i+7]) & M == d, (W2[i+16]-W[i+16]) & M == d))
print(f"      verified: coef(W_i in W_(i+7)) = +1 in {sum(x for x,_ in sc)}/20, "
      f"coef(W_i in W_(i+16)) = +1 in {sum(y for _,y in sc)}/20")

print()
print("=" * 78)
print("4. EDGE WEIGHTS IN THE W-DOMAIN (bits moved per flipped bit)")
print("=" * 78)
print("  rows: message word flipped; cols: the 8 digest residuals")
print("  absorption needs <= 2 bits (mild).  full avalanche ~ 16.")
NT = 120
for R in (19, 20, 21):
    tgt = digest_words([RW() for _ in range(16)], R)
    print(f"  R={R}")
    print("    W_i  " + " ".join(f"{n:>5}" for n in
          ["a-1", "a-2", "a-3", "a-4", "e-1", "e-2", "e-3", "e-4"]))
    mn = 99.0
    for i in range(16):
        tot = [0] * 8
        for _ in range(NT):
            W = [RW() for _ in range(16)]
            r0 = residuals(W, R, tgt)
            W2 = list(W); W2[i] ^= 1 << rng.randrange(32)
            r1 = residuals(W2, R, tgt)
            for j in range(8):
                tot[j] += popcount(r0[j] ^ r1[j])
        row = [t / NT for t in tot]
        mn = min(mn, min(row))
        print(f"    W_{i:<2} " + " ".join(f"{x:5.1f}" for x in row))
    print(f"    minimum entry over the whole matrix: {mn:.2f} bits "
          f"(absorption threshold 2.0)")

print()
print("=" * 78)
print("5. ADDITIVE SEPARABILITY  R_j = F(W_i) + G(rest) ?")
print("=" * 78)
print("  Exact necessary test: the discrete derivative R_j(x+d) - R_j(x) must be")
print("  the same for two different settings of the other 15 message words.")
R = 20
tgt = digest_words([RW() for _ in range(16)], R)
NS = 200
worst = 0.0
for i in range(16):
    hits = [0] * 8
    for _ in range(NS):
        x = RW(); d = RW()
        A = [RW() for _ in range(16)]
        B = [RW() for _ in range(16)]
        for W in (A, B):
            W[i] = x
        A2 = list(A); A2[i] = (x + d) & M
        B2 = list(B); B2[i] = (x + d) & M
        dA = [(u - v) & M for u, v in zip(residuals(A2, R, tgt), residuals(A, R, tgt))]
        dB = [(u - v) & M for u, v in zip(residuals(B2, R, tgt), residuals(B, R, tgt))]
        for j in range(8):
            if dA[j] == dB[j]:
                hits[j] += 1
    best = max(hits) / NS
    worst = max(worst, best)
    print(f"    W_{i:<2} separable fraction per residual: "
          + " ".join(f"{h/NS:.2f}" for h in hits))
print(f"  best separability anywhere: {worst:.3f}  (1.00 = absorbable, "
      f"0.00 = fully entangled)")

print()
print("  POSITIVE CONTROL: the same test on the a-domain absorber C_0 <- a_1.")
ab, eb = backward_chain(digest([RW() for _ in range(16)], R), R)
def C_resid(free, R, ab):
    Wd, a, e = build_a(free, R, ab)
    Wf = dict(Wd)
    out = []
    for j in range(R - 16):
        out.append((Wf[16+j] - s1(Wf[14+j]) - Wf[9+j] - s0(Wf[1+j]) - Wf[j]) & M)
    return out
for k in (1, 2, 3, 4, 5):
    hits = [0] * (R - 16)
    for _ in range(NS):
        x = RW(); d = RW()
        A = {i: RW() for i in range(R - 8)}; A[k] = x
        B = {i: RW() for i in range(R - 8)}; B[k] = x
        A2 = dict(A); A2[k] = (x + d) & M
        B2 = dict(B); B2[k] = (x + d) & M
        dA = [(u - v) & M for u, v in zip(C_resid(A2, R, ab), C_resid(A, R, ab))]
        dB = [(u - v) & M for u, v in zip(C_resid(B2, R, ab), C_resid(B, R, ab))]
        for j in range(R - 16):
            if dA[j] == dB[j]:
                hits[j] += 1
    print(f"    a_{k} separable fraction per constraint C0..C{R-17}: "
          + " ".join(f"{h/NS:.2f}" for h in hits))
