"""Reduced-width SHA-256 analogue: w-bit words, scaled rotation constants."""
import numpy as np

# (r1, r2, s) for sigma0 ; (R1,R2,s) for sigma1 ; Sigma0/Sigma1 rotations
PARAMS = {
    8:  dict(s0=(2, 5, 1), s1=(4, 5, 3), S0=(1, 3, 6), S1=(2, 3, 6)),
    12: dict(s0=(3, 7, 1), s1=(6, 7, 4), S0=(1, 5, 8), S1=(2, 4, 9)),
    16: dict(s0=(4, 9, 2), s1=(9, 10, 5), S0=(1, 7, 11), S1=(3, 6, 13)),
    32: dict(s0=(7, 18, 3), s1=(17, 19, 10), S0=(2, 13, 22), S1=(6, 11, 25)),
}


def mk(w):
    M = (1 << w) - 1
    p = PARAMS[w]

    def rotr(x, n):
        n %= w
        return ((x >> n) | (x << (w - n))) & M

    def s0(x):
        a, b, c = p['s0']
        return rotr(x, a) ^ rotr(x, b) ^ (x >> c)

    def s1(x):
        a, b, c = p['s1']
        return rotr(x, a) ^ rotr(x, b) ^ (x >> c)

    def S0(x):
        a, b, c = p['S0']
        return rotr(x, a) ^ rotr(x, b) ^ rotr(x, c)

    def S1(x):
        a, b, c = p['S1']
        return rotr(x, a) ^ rotr(x, b) ^ rotr(x, c)

    return dict(M=M, w=w, rotr=rotr, s0=s0, s1=s1, S0=S0, S1=S1, p=p)


def s0_bit_sources(w):
    """For each output bit i of sigma0, the list of input bit indices XORed."""
    a, b, c = PARAMS[w]['s0']
    src = []
    for i in range(w):
        # rotr(x,a) bit i = x bit (i+a) mod w ; (x>>c) bit i = x bit i+c (if < w)
        t = [(i + a) % w, (i + b) % w]
        if i + c < w:
            t.append(i + c)
        src.append(t)
    return src


def brute_s0_minus_u(w):
    """Table of u -> (s0(u)-u) mod 2^w, and inverse map."""
    f = mk(w)
    M = f['M']
    u = np.arange(1 << w, dtype=np.int64)
    val = (f['s0'](u) - u) & M
    inv = {}
    for uu, vv in zip(u.tolist(), val.tolist()):
        inv.setdefault(vv, []).append(uu)
    return val, inv
