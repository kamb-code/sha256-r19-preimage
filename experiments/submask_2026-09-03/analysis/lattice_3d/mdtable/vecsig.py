"""Vectorised context -> signature map, for fibre enumeration."""
import numpy as np
from wmodel import W


def make_vecsig(w: W, h, R=20):
    M = w.M
    ab, _ = w.backward_chain(h, R)
    A = {r: ab[r] for r in range(R - 8, R)}          # a12..a19, scalars
    U = np.uint64

    def f(v, A6, A7, A10, A11):
        a4 = a5 = U(v)
        a6, a7, a10, a11 = A6, A7, A10, A11
        a8 = (U(M) - a4 + w.S0(a7) + w.Maj(a7, a6, a5)) & M
        a9 = (U(M) - a5 + w.S0(a8) + w.Maj(a8, a7, a6)) & M
        a12, a13, a14, a15, a16, a17, a18, a19 = (U(A[r]) for r in range(12, 20))
        e8 = U(M); e9 = U(M)
        e10 = (a6 + a10 - w.T2(a9, a8, a7)) & M
        e11 = (a7 + a11 - w.T2(a10, a9, a8)) & M
        e12 = (a8 + a12 - w.T2(a11, a10, a9)) & M
        e13 = (a9 + a13 - w.T2(a12, a11, a10)) & M
        e14 = (a10 + a14 - w.T2(a13, a12, a11)) & M
        e15 = (a11 + a15 - w.T2(a14, a13, a12)) & M
        e16 = (a12 + a16 - w.T2(a15, a14, a13)) & M
        e17 = (a13 + a17 - w.T2(a16, a15, a14)) & M
        e18 = (a14 + a18 - w.T2(a17, a16, a15)) & M
        K = [U(x) for x in w.K]
        W12 = (a12 - w.T2(a11, a10, a9) - e8 - w.S1(e11) - w.Ch(e11, e10, e9) - K[12]) & M
        W14 = (a14 - w.T2(a13, a12, a11) - e10 - w.S1(e13) - w.Ch(e13, e12, e11) - K[14]) & M
        W15 = (a15 - w.T2(a14, a13, a12) - e11 - w.S1(e14) - w.Ch(e14, e13, e12) - K[15]) & M
        W16 = (a16 - w.T2(a15, a14, a13) - e12 - w.S1(e15) - w.Ch(e15, e14, e13) - K[16]) & M
        W17 = (a17 - w.T2(a16, a15, a14) - e13 - w.S1(e16) - w.Ch(e16, e15, e14) - K[17]) & M
        W18 = (a18 - w.T2(a17, a16, a15) - e14 - w.S1(e17) - w.Ch(e17, e16, e15) - K[18]) & M
        W19 = (a19 - w.T2(a18, a17, a16) - e15 - w.S1(e18) - w.Ch(e18, e17, e16) - K[19]) & M
        K0p = (W16 - w.s1(W14)) & M
        K1p = (W17 - w.s1(W15)) & M
        K2p = (W18 - w.s1(W16)) & M
        K3p = (W19 - w.s1(W17) - W12) & M
        T1_7 = (a7 - w.T2(a6, a5, a4)) & M
        c6 = (a6 - w.S0(a5)) & M
        W9base = (a9 - w.T2(a8, a7, a6) - K[9]) & M
        W10base = (a10 - w.T2(a9, a8, a7) - w.S1(e9) - K[10]) & M
        W11base = (a11 - w.T2(a10, a9, a8) - w.S1(e10) - K[11]) & M
        W9hat = (W9base - (a5 - w.S0(a4) - w.Maj(a4, U(0), U(0))) - w.S1(e8)
                 - w.Ch(e8, T1_7, (c6 - w.Maj(a5, a4, U(0))) & M)) & M
        D = ((a6 - w.S0(a5) - w.Maj(a5, a4, U(0))) + w.Ch(e9, e8, T1_7)) & M
        KC0 = (K0p - W9hat) & M
        KC1 = (K1p - W10base + D) & M
        KC2 = (K2p - W11base + T1_7 + w.Ch(e10, e9, e8)) & M
        return KC0, KC1, KC2, K3p
    return f
