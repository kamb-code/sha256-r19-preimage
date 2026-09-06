#!/usr/bin/env python3
"""Part (a)/(b): the a/e duality of SHA-256 and mixed frames.

A FRAME assigns to every round r < R-8 one free 32-bit variable x_r which is
either a_r ('a') or e_r ('e'); a_{R-8..R-1} come from the backward chain.
Given the variables, the trajectory is rebuilt forward from the IV:
    'a': a_r = x_r
    'e': a_r = x_r - a_{r-4} + T2(a_{r-1},a_{r-2},a_{r-3})   (so that e_r = x_r)
and e_r = a_{r-4} + a_r - T2(...) for all r, W_r by the recovery formula, and
the schedule constraints F_j = W_{16+j} - s1(W_{14+j}) - W_{9+j} - s0(W_{1+j}) - W_j.

Measurements (all numeric, random states, vectorised):
  1. linear-entry pattern of x_k in W_{k+m} for the a-frame, the e-frame and
     the backward e-frame (variables e_r, a's rebuilt from the digest end).
  2. single-table absorber test: (x_k, C_j) admits a one-lookup absorber iff
     x_k enters exactly two of the five words of C_j, both exactly linearly,
     one of them the sigma argument -- exhaustively over frames.
  3. edge weights (bits of F_j moved per flipped bit of x_k) for all 2^9
     mixed frames of frame B at R=20 (unknown rounds 0..4, context 5..11).
"""
import itertools, sys
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import (M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, u32, U32, MISS)

np.seterr(over="ignore")


def popcnt(x):
    x = x.astype(np.uint64)
    c = np.zeros_like(x)
    for _ in range(32):
        c += x & 1
        x >>= 1
    return c


def c(v):
    return U32(v & M)


def build(kind, x, bc, R):
    a = {-1: c(IV[0]), -2: c(IV[1]), -3: c(IV[2]), -4: c(IV[3])}
    e = {-1: c(IV[4]), -2: c(IV[5]), -3: c(IV[6]), -4: c(IV[7])}
    for r in range(R):
        if r >= R - 8:
            a[r] = bc[r]
        elif kind[r] == 'a':
            a[r] = x[r]
        else:
            a[r] = (x[r] - a[r - 4] + T2(a[r - 1], a[r - 2], a[r - 3]))
        e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3]))
    W = {}
    for r in range(R):
        W[r] = (a[r] - T2(a[r - 1], a[r - 2], a[r - 3]) - e[r - 4] - S1(e[r - 1])
                - Ch(e[r - 1], e[r - 2], e[r - 3]) - c(K[r]))
    return a, e, W


def constraints(W, R):
    return [(W[16 + j] - s1(W[14 + j]) - W[9 + j] - s0(W[1 + j]) - W[j]) for j in range(R - 16)]


def build_backward_e(x, dig, R):
    """Backward e-frame: variables e_0..e_{R-5}; a's rebuilt downward from the
    digest words a_{R-4..R-1}, e_{R-4..R-1} via a_{r-4} = e_r - a_r + T2(a_{r-1},a_{r-2},a_{r-3}).
    The IV then becomes a 4-word constraint at the bottom (not imposed here)."""
    a = {r: dig['a'][r] for r in range(R - 4, R)}
    e = {r: dig['e'][r] for r in range(R - 4, R)}
    for r in range(R - 5, -1, -1):
        e[r] = x[r]
    for r in range(R - 1, -1, -1):      # a_{R-5} .. a_{-4}
        a[r - 4] = (e[r] - a[r] + T2(a[r - 1], a[r - 2], a[r - 3]))
    e.update({-1: c(IV[4]), -2: c(IV[5]), -3: c(IV[6]), -4: c(IV[7])})
    W = {}
    for r in range(R):
        W[r] = (a[r] - T2(a[r - 1], a[r - 2], a[r - 3]) - e[r - 4] - S1(e[r - 1])
                - Ch(e[r - 1], e[r - 2], e[r - 3]) - c(K[r]))
    return a, e, W


def rand_words(rng, n):
    return rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(U32)


def classify(dW, d):
    """'+' if dW == +d for all, '-' if -d, '.' if 0, 'N' otherwise."""
    if np.all(dW == 0):
        return '.'
    if np.all(dW == d):
        return '+'
    if np.all(dW == (-d)):
        return '-'
    return 'N'


def linear_entries(rng, R=20, n=64):
    print(f"=== 1. linear entries: how x_k enters W_(k+m)  (R={R}; + / - exact unit coefficient, N nonlinear, . none)")
    for name in ("a-frame", "e-frame (forward)", "e-frame (backward)"):
        print(f"\n  {name}")
        print("   m:     " + " ".join(f"{m:>2}" for m in range(0, 13)))
        for k in range(0, 6):
            row = []
            for m in range(0, 13):
                r = k + m
                if r >= R:
                    row.append(' .'); continue
                if name == "e-frame (backward)":
                    x = {i: rand_words(rng, n) for i in range(R - 4)}
                    dig = {'a': {i: rand_words(rng, n) for i in range(R - 4, R)},
                           'e': {i: rand_words(rng, n) for i in range(R - 4, R)}}
                    _, _, W0 = build_backward_e(x, dig, R)
                    d = rand_words(rng, n) | U32(1)
                    x2 = dict(x); x2[k] = x[k] + d
                    _, _, W1 = build_backward_e(x2, dig, R)
                else:
                    kind = {i: ('a' if name.startswith('a') else 'e') for i in range(R - 8)}
                    x = {i: rand_words(rng, n) for i in range(R - 8)}
                    bc = {i: rand_words(rng, n) for i in range(R - 8, R)}
                    _, _, W0 = build(kind, x, bc, R)
                    d = rand_words(rng, n) | U32(1)
                    x2 = dict(x); x2[k] = x[k] + d
                    _, _, W1 = build(kind, x2, bc, R)
                row.append(' ' + classify(W1[r] - W0[r], d))
            print(f"   x{k}:    " + " ".join(row))


def absorber_test(rng, R=20, n=48):
    """For every frame (2^(R-8) a/e choices is too many: use pure a, pure e, and all
    2^9 choices on rounds 0..8 with a-context above) list (x_k, C_j) pairs that admit a
    single-table absorber: exactly two words of C_j move, both exactly linearly (unit
    coefficient, opposite or equal sign), one of them a sigma argument."""
    print(f"\n=== 2. single-table absorber test at R={R}: pairs (x_k, C_j) with x_k exactly linear in two")
    print("       words of C_j (one under a sigma) and absent from the other three")
    words_of = lambda j: [(16 + j, 'lin'), (14 + j, 's1'), (9 + j, 'lin'), (1 + j, 's0'), (j, 'lin')]
    found = {}
    frames = []
    for bits in itertools.product('ae', repeat=9):
        kind = {i: (bits[i] if i < 9 else 'a') for i in range(R - 8)}
        frames.append(kind)
    frames.append({i: 'e' for i in range(R - 8)})
    for kind in frames:
        key = ''.join(kind[i] for i in range(R - 8))
        for k in range(0, R - 8):
            x = {i: rand_words(rng, n) for i in range(R - 8)}
            bc = {i: rand_words(rng, n) for i in range(R - 8, R)}
            _, _, W0 = build(kind, x, bc, R)
            d = rand_words(rng, n) | U32(1)
            x2 = dict(x); x2[k] = x[k] + d
            _, _, W1 = build(kind, x2, bc, R)
            cls = {r: classify(W1[r] - W0[r], d) for r in range(R)}
            for j in range(R - 16):
                ent = [(r, typ, cls[r]) for r, typ in words_of(j) if cls[r] != '.']
                if len(ent) == 2 and all(cc in '+-' for _, _, cc in ent) and any(t in ('s0', 's1') for _, t, _ in ent):
                    found.setdefault((k, j, tuple(ent)), []).append(key)
    for (k, j, ent), keys in sorted(found.items()):
        desc = ", ".join(f"W{r}({typ}){cc}" for r, typ, cc in ent)
        kinds = sorted(set(kk[k] for kk in keys))
        print(f"   x{k} -> C{j}: {desc}   in {len(keys)} frames; x{k} is {'/'.join(kinds)}-type there")
    print(f"   ({len(frames)} frames x {R-8} variables x {R-16} constraints tested)")
    return found


def edge_weights_frameB(rng, R=20, n=200):
    """Frame B: unknown rounds 0..4 (x0 swept, x1..x4 absorbed by C0..C3), context 5..11.
    All 2^9 a/e choices on rounds 0..8 (rounds 9..11 are context-only either way)."""
    print(f"\n=== 3. edge weights, frame B at R={R}, all 2^9 mixed frames on rounds 0..8 ({n} trials/cell)")
    results = []
    for bits in itertools.product('ae', repeat=9):
        kind = {i: (bits[i] if i < 9 else 'a') for i in range(R - 8)}
        key = ''.join(bits)
        Wt = np.zeros((5, R - 16))
        for k in range(5):
            x = {i: rand_words(rng, n) for i in range(R - 8)}
            bc = {i: rand_words(rng, n) for i in range(R - 8, R)}
            _, _, W0 = build(kind, x, bc, R)
            F0 = constraints(W0, R)
            flip = (U32(1) << rng.integers(0, 32, n, dtype=np.uint64).astype(U32))
            x2 = dict(x); x2[k] = x[k] ^ flip
            _, _, W1 = build(kind, x2, bc, R)
            F1 = constraints(W1, R)
            for j in range(R - 16):
                Wt[k, j] = popcnt(F0[j] ^ F1[j]).mean()
        heavy_fb = [(k, j, Wt[k, j]) for k in range(1, 5) for j in range(R - 16) if j < k - 1 and Wt[k, j] > 4.0]
        results.append((key, Wt, heavy_fb))
    results.sort(key=lambda t: (len(t[2]), sum(w for _, _, w in t[2])))
    print("   frames with the fewest heavy feedback edges (k -> C_j with j < k-1, > 4 bits):")
    for key, Wt, hf in results[:8]:
        print(f"   frame {key} (rounds 0..8): {len(hf)} heavy feedback edge(s): " +
              ", ".join(f"x{k}->C{j} {w:.1f}" for k, j, w in hf))
    key, Wt, hf = results[0]
    print(f"\n   full matrix of the best frame {key}:")
    print("          " + " ".join(f"   C{j}" for j in range(R - 16)))
    for k in range(5):
        print(f"   x{k}:    " + " ".join(f"{Wt[k, j]:6.1f}" for j in range(R - 16)))
    ref = [r for r in results if r[0] == 'a' * 9][0]
    print(f"\n   reference a-frame: {len(ref[2])} heavy feedback edge(s): " +
          ", ".join(f"a{k}->C{j} {w:.1f}" for k, j, w in ref[2]))
    hist = {}
    for _, _, hf in results:
        hist[len(hf)] = hist.get(len(hf), 0) + 1
    print(f"   histogram of #heavy feedback edges over 512 frames: {dict(sorted(hist.items()))}")
    # minimum weight of the lightest feedback edge into C0 from x4 across frames
    best = min(results, key=lambda t: t[1][4, 0])
    print(f"   lightest x4->C0 edge over all frames: {best[1][4,0]:.2f} bits in frame {best[0]}")
    return results


if __name__ == "__main__":
    rng = np.random.default_rng(20260906)
    linear_entries(rng)
    absorber_test(rng)
    edge_weights_frameB(rng)
