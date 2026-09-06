"""Shifted frame (a1 guessed, a4 absorbed by C3; context a5=a6=v, e9=e10=-1):
numerically confirm that C1's dependence on (a3,a4) is exactly Maj(v,a4,a3)-a4, i.e. the
provisional-a4 freeze in C1 passes with probability (3/4)^32, and that C2 is free of a4."""
import sys, numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, recover_W, U32, MISS, u32
R = 20
rng = np.random.default_rng(3)
rnd = lambda: int(rng.integers(0, 1 << 32, dtype=np.uint64))
def rhs(a, e, j):
    W = {r: recover_W(a, e, r) for r in range(R)}
    return (W[16 + j] - s1(W[14 + j]) - W[9 + j] - W[j]) & M
bad = 0; c2dep = 0; hits = 0; tot = 0
for t in range(200):
    v = rnd(); a7 = rnd(); a8 = rnd()
    ctx = {5: v, 6: v, 7: a7, 8: a8}
    ctx[9] = (M - v + S0(a8) + Maj(a8, a7, v)) & M          # e9 = a5 + a9 - T2(a8,a7,a6) = -1
    ctx[10] = (M - v + S0(ctx[9]) + Maj(ctx[9], a8, a7)) & M  # e10 = a6 + a10 - T2(a9,a8,a7) = -1
    ctx[11] = rnd()
    base = {i: rnd() for i in range(0, 5)}
    chain = {i: rnd() for i in range(12, R)}
    def full(unk):
        a = dict(ctx); a.update(chain); a.update(unk)
        a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
        e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
        for r in range(R): e[r] = (a[r-4] + a[r] - T2(a[r-1], a[r-2], a[r-3])) & M
        assert e[9] == M and e[10] == M
        return a, e
    a, e = full(base); r1 = rhs(a, e, 1); r2 = rhs(a, e, 2)
    u2 = dict(base); u2[3] = rnd(); u2[4] = rnd()
    a2_, e2_ = full(u2); r1b = rhs(a2_, e2_, 1); r2b = rhs(a2_, e2_, 2)
    # predicted change of C1's rhs: W10 = const - e6 - S1(e9) - e8 with e6 = a2 + ... - Maj(v,a4,a3), e8 = a4 + c
    pred = ((Maj(v, u2[4], u2[3]) - u2[4]) - (Maj(v, base[4], base[3]) - base[4])) & M
    if ((r1b - r1) & M) != ((-pred) & M): bad += 1   # rhs = ... - W10, W10 = const + Maj(v,a4,a3) - a4
    if r2b != r2: c2dep += 1
    # freeze pass rate: a4 provisional = v ; true (a3,a4) random
    n = 1 << 18
    A3 = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(U32); A4 = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(U32)
    hits += int(((Maj(u32(v), A4, A3) - A4) == U32(0)).sum()); tot += n
print(f"C1 rhs moves with (a3,a4) exactly as Maj(v,a4,a3)-a4 : {200-bad}/200 OK")
print(f"C2 rhs moves with (a3,a4)?  changed in {c2dep}/200 trials (a3 should move it, a4 alone tested next)")
c2a4 = 0
for t in range(100):
    v = rnd(); a7 = rnd(); a8 = rnd()
    ctx = {5: v, 6: v, 7: a7, 8: a8}
    ctx[9] = (M - v + S0(a8) + Maj(a8, a7, v)) & M; ctx[10] = (M - v + S0(ctx[9]) + Maj(ctx[9], a8, a7)) & M; ctx[11] = rnd()
    base = {i: rnd() for i in range(0, 5)}; chain = {i: rnd() for i in range(12, R)}
    def full(unk):
        a = dict(ctx); a.update(chain); a.update(unk); a.update({-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]})
        e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
        for r in range(R): e[r] = (a[r-4] + a[r] - T2(a[r-1], a[r-2], a[r-3])) & M
        return a, e
    a, e = full(base); u2 = dict(base); u2[4] = rnd(); b, f = full(u2)
    if rhs(a, e, 2) != rhs(b, f, 2): c2a4 += 1
print(f"C2 rhs moves with a4 alone: {c2a4}/100 trials (expected 0 in the shifted family)")
print(f"P(Maj(v,a4,a3) - a4 == 0) over {tot:,} random pairs: {hits/tot:.3e}  vs (3/4)^32 = {0.75**32:.3e}")
