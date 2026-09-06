#!/usr/bin/env python3
"""Independent scalar re-measurement of the a5 -> C_j edge weights for one context
condition (pure-Python ints, no shared code with tie_scan.py beyond the SHA-256
primitives of code/submask_family.py).

usage: python3 reproduce.py "<defs dict>" [states] [seed]
   e.g. python3 reproduce.py "{7: ('eq', 6), 8: ('sat_r', 0xFFFFFFFF)}"

Frame: a5 unknown; a4, a6..a11 context (defined by defs, else random); a12..a19
random digest words; a0..a3 random.  For each state and each bit of a5: flip it,
re-derive the context words that are defined in terms of other words, recompute
the four lookup targets T_j = W_{16+j} - s1(W_{14+j}) - W_{9+j} - W_j, and count
changed bits.  Also the a4 -> C0 control (flip a4, re-derive context).
"""
import sys, ast, random
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import M, K, IV, S0, S1, s0, s1, Ch, Maj, T2, rotr

R = 20


def build(defs, a5, a4, base, unk, dig):
    """Return the a-dict for one state.  base: random values of free context words."""
    a = {-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]}
    a.update(unk)
    a.update(dig)
    a[5] = a5
    pend = {i: defs.get(i, ('free',)) for i in range(4, 12) if i != 5}
    done = set()
    while len(done) < len(pend):
        prog = False
        for i, d in pend.items():
            if i in done:
                continue
            k = d[0]
            if k == 'free':
                a[i] = a4 if i == 4 else base[i]
            else:
                if k in ('eq', 'neq', 'rotr', 'xor', 'add', 'S0add', 'S1add', 'maj', 'lin'):
                    deps = [d[1]]
                elif k == 'sat_r':
                    deps = [i - 4, i - 1, i - 2, i - 3]
                elif k == 'sat_h':
                    deps = [i + 4, i + 3, i + 2, i + 1]
                else:
                    raise ValueError(d)
                if not all(j in a for j in deps):
                    continue
                if k == 'eq':
                    a[i] = a[d[1]]
                elif k == 'neq':
                    a[i] = a[d[1]] ^ M
                elif k == 'rotr':
                    a[i] = rotr(a[d[1]], d[2]) if d[2] else a[d[1]]
                elif k == 'xor':
                    a[i] = a[d[1]] ^ (d[2] & M)
                elif k == 'add':
                    a[i] = (a[d[1]] + d[2]) & M
                elif k == 'S0add':
                    a[i] = (S0(a[d[1]]) + d[2]) & M
                elif k == 'S1add':
                    a[i] = (S1(a[d[1]]) + d[2]) & M
                elif k == 'maj':
                    a[i] = Maj(a[d[1]], d[2] & M, d[3] & M)
                elif k == 'lin':
                    a[i] = 0
                    for n in d[2]:
                        a[i] ^= rotr(a[d[1]], n) if n else a[d[1]]
                elif k == 'sat_r':
                    a[i] = (d[1] - a[i - 4] + T2(a[i - 1], a[i - 2], a[i - 3])) & M
                elif k == 'sat_h':
                    r = i + 4
                    a[i] = (d[1] - a[r] + T2(a[r - 1], a[r - 2], a[r - 3])) & M
            done.add(i)
            prog = True
        if not prog:
            raise RuntimeError("cyclic definitions")
    return a


def targets(a):
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(R):
        e[r] = (a[r - 4] + a[r] - T2(a[r - 1], a[r - 2], a[r - 3])) & M
    W = {}
    for r in range(R):
        W[r] = (a[r] - T2(a[r - 1], a[r - 2], a[r - 3]) - e[r - 4] - S1(e[r - 1])
                - Ch(e[r - 1], e[r - 2], e[r - 3]) - K[r]) & M
    # sanity: the recovered W reproduce the state when run forward
    return e, W, [(W[16 + j] - s1(W[14 + j]) - W[9 + j] - W[j]) & M for j in range(4)]


def main():
    defs = ast.literal_eval(sys.argv[1]) if len(sys.argv) > 1 else {7: ('eq', 6), 8: ('sat_r', M)}
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    rnd = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 20260906)
    r32 = lambda: rnd.getrandbits(32)
    ham5 = [0] * 4; ham4 = [0] * 4; n5 = n4 = 0; ctxdep5 = ctxdep4 = 0
    for _ in range(N):
        unk = {i: r32() for i in range(4)}
        dig = {i: r32() for i in range(12, R)}
        base = {i: r32() for i in range(6, 12)}
        a4, a5 = r32(), r32()
        a = build(defs, a5, a4, base, unk, dig)
        _, _, T0 = targets(a)
        for bit in range(32):
            b = build(defs, a5 ^ (1 << bit), a4, base, unk, dig)
            ctxdep5 += any(b[i] != a[i] for i in range(4, 12) if i != 5)
            _, _, T1 = targets(b)
            for j in range(4):
                ham5[j] += bin(T0[j] ^ T1[j]).count('1')
            n5 += 1
            if defs.get(4, ('free',))[0] == 'free':
                b = build(defs, a5, a4 ^ (1 << bit), base, unk, dig)
                ctxdep4 += any(b[i] != a[i] for i in range(5, 12))
                _, _, T1 = targets(b)
                for j in range(4):
                    ham4[j] += bin(T0[j] ^ T1[j]).count('1')
                n4 += 1
    print("defs =", defs)
    print("a5 -> " + "  ".join(f"C{j} {ham5[j]/n5:5.2f}" for j in range(4)) + f"   bits per flipped bit ({N} states x 32 bits)"
          + ("   [context depends on a5]" if ctxdep5 else "   [context independent of a5]"))
    if n4:
        print("a4 -> " + "  ".join(f"C{j} {ham4[j]/n4:5.2f}" for j in range(4)) + "   (control)"
              + ("   [context depends on a4]" if ctxdep4 else ""))


if __name__ == "__main__":
    main()
