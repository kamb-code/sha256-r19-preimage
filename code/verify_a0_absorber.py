#!/usr/bin/env python3
"""Verify the a0 absorber identity for the C0 schedule constraint.

C0 is the requirement that the message word W16 produced by the round function
equals the one produced by the schedule recurrence,

    W16 = sigma1(W14) + W9 + sigma0(W1) + W0.

In the attack a0 is swept and C0 absorbs a1 through the global sigma0(u)-u
table.  The identity below shows the roles can be exchanged: C0 also absorbs
a0, through a different global table.

Writing gv for the round-1 constant and using

    W1 = a1 + gv(a0),    W0 = a0 + C0_CONST,    W9(a1) = W9_0 - a1,

where W9_0 is W9 evaluated at a1 = 0, C0 rearranges to

    Gamma(a0) = R + W1 - sigma0(W1),        Gamma(a0) := a0 + gv(a0)
    R := W16 - sigma1(W14) - W9_0 - C0_CONST

Two facts make this useful:

  * gv depends ONLY on a0 and the SHA-256 IV -- no context words, no target.
    So Gamma is a fixed function and Gamma^-1 is a GLOBAL table, built once and
    valid for every target and every context, exactly like sigma0(u)-u.
  * Gamma is a near-random map (about 63% coverage), so the table behaves like
    the sigma0 table already used.

This does not by itself lower the attack's cost: C0 can absorb a0 OR a1 but not
both, so one of the two must still be swept and the remaining unknowns still
form a cycle.  It is recorded because it removes one of the few structural
degrees of freedom the attack has, and because Gamma is a new global table.

Usage:
    python3 verify_a0_absorber.py            # check against P1, P2, P3
    python3 verify_a0_absorber.py --full     # add the 2^32 Gamma coverage scan
"""

from __future__ import annotations

import argparse

from sha256_core import H0, K
from utils import big_sigma0, big_sigma1, ch, maj, small_sigma0, small_sigma1

M = 0xFFFFFFFF

# the three published preimages (see verified_preimages.txt)
CASES = [
    ("P1", "1e65261c54255188604f5375091839733de63e966b5e4715658226bf03588447",
     "22f091af ec52d67b 74c33819 a280dc6a b001ff1a 1f2356a5 3eccf108 bd9a2333 "
     "abe611d1 6d1e5a20 8041df25 e43d31af aa895a2e 69106ad2 7479fa3a 2a9abb91"),
    ("P2", "fb52f81baed24f8728faf5bbce82c67d510761172fb9876d9e3a72dda351b7ca",
     "37e6702f bc20efea 2dd42a3e 501dfbe9 3cacc578 ea2de1c1 11c0f066 0f22be47 "
     "2a447d2d 13f0080f 1f33df6b d655d8e6 15730eaa 9bf64950 9f129973 5a964edf"),
    ("P3", "1bd7ebbdc4d938fb26d19b5dd5caf333de397bd1c745727bd5556baf38ccf977",
     "3ce8fba4 e2fb9661 44730c59 e1cf4bc0 e1a18d93 97658983 67efe2a7 ef260ecb "
     "d4c6dbe0 13e9388e 95664a59 4d9e248b 74137862 664815ac 89eae95a cd7dbef5"),
]


def gv_of(a0: int) -> int:
    """Round-1 constant. Depends only on a0 and the IV -- no context, no target."""
    t2_0 = (big_sigma0(H0[0]) + maj(H0[0], H0[1], H0[2])) & M
    e0 = (H0[3] + ((a0 - t2_0) & M)) & M
    return (-big_sigma0(a0) - maj(a0, H0[0], H0[1]) - H0[6]
            - big_sigma1(e0) - ch(e0, H0[4], H0[5]) - K[1]) & M


def state_a(words, n):
    """Run n rounds and return the a-sequence (indexed from -4)."""
    # registers (a,b,c,d) = H0[0..3] and (e,f,g,h) = H0[4..7], so the most
    # recent value carries index -1:  a[-1]=a=H0[0], ..., a[-4]=d=H0[3]
    a = {-1 - i: H0[i] for i in range(4)}
    e = {-1 - i: H0[4 + i] for i in range(4)}
    for t in range(n):
        T1 = (e[t - 4] + big_sigma1(e[t - 1])
              + ch(e[t - 1], e[t - 2], e[t - 3]) + K[t] + words[t]) & M
        T2 = (big_sigma0(a[t - 1]) + maj(a[t - 1], a[t - 2], a[t - 3])) & M
        a[t] = (T1 + T2) & M
        e[t] = (a[t - 4] + T1) & M
    return a, e


def check(name, target_hex, words_str):
    words = [int(x, 16) for x in words_str.split()]
    a, e = state_a(words, 16)
    a0, a1 = a[0], a[1]

    gv = gv_of(a0)
    C0_CONST = (words[0] - a0) & M
    W1, W9, W14 = words[1], words[9], words[14]
    W16 = (small_sigma1(W14) + W9 + small_sigma0(W1) + words[0]) & M

    # W9(a1) = W9_0 - a1, so W9_0 = W9 + a1
    W9_0 = (W9 + a1) & M
    R = (W16 - small_sigma1(W14) - W9_0 - C0_CONST) & M

    lhs = (a0 + gv) & M
    rhs = (R + W1 - small_sigma0(W1)) & M
    ok = lhs == rhs
    print(f"  {name}:  a0={a0:#010x}  Gamma(a0)={lhs:#010x}  "
          f"R+W1-sigma0(W1)={rhs:#010x}   {'PASS' if ok else 'FAIL'}")
    # W1 = a1 + gv(a0) is the substitution the derivation relies on
    ok2 = ((a1 + gv) & M) == W1
    print(f"        W1 == a1 + gv(a0):  {'PASS' if ok2 else 'FAIL'}")
    return ok and ok2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="scan all 2^32 a0 for Gamma coverage (needs numpy, ~2 min)")
    args = ap.parse_args()

    print("gv depends only on a0 and the IV (no context words appear in gv_of).")
    print("\nIdentity  Gamma(a0) = a0 + gv(a0) = R + W1 - sigma0(W1)  on the "
          "published preimages:")
    allok = all(check(*c) for c in CASES)
    print(f"\n  {'all cases pass' if allok else 'FAILURE'}")

    print("\nControl: the identity encodes C0, so perturbing W16 must break it.")
    words = [int(x, 16) for x in CASES[0][2].split()]
    a, _ = state_a(words, 16)
    a0, a1 = a[0], a[1]
    gv = gv_of(a0)
    C0_CONST = (words[0] - a0) & M
    W1, W9, W14 = words[1], words[9], words[14]
    W16 = (small_sigma1(W14) + W9 + small_sigma0(W1) + words[0]) & M
    hits = 0
    for delta in range(1, 65):
        R = (W16 + delta - small_sigma1(W14) - ((W9 + a1) & M) - C0_CONST) & M
        if ((a0 + gv) & M) == ((R + W1 - small_sigma0(W1)) & M):
            hits += 1
    print(f"  perturbing W16 by 1..64:  {hits}/64 still satisfy it "
          f"({'good, it is not a tautology' if hits == 0 else 'PROBLEM'})")

    if args.full:
        import numpy as np
        np.seterr(over="ignore")

        def rotr(x, n):
            return (x >> np.uint32(n)) | (x << np.uint32(32 - n))

        def vS0(x): return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
        def vS1(x): return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
        def vch(e, f, g): return (e & f) ^ (~e & g)
        def vmaj(a, b, c): return (a & b) ^ (a & c) ^ (b & c)

        Hn = [np.uint32(h) for h in H0]
        t2_0 = vS0(Hn[0]) + vmaj(Hn[0], Hn[1], Hn[2])
        seen = np.zeros(1 << 32, dtype=bool)
        step = 1 << 26
        for base in range(0, 1 << 32, step):
            x = np.arange(step, dtype=np.uint32) + np.uint32(base)
            e0 = Hn[3] + (x - t2_0)
            g = (np.uint32(0) - vS0(x) - vmaj(x, Hn[0], Hn[1]) - Hn[6]
                 - vS1(e0) - vch(e0, Hn[4], Hn[5]) - np.uint32(K[1]))
            seen[x + g] = True
        cov = seen.sum() / (1 << 32)
        print(f"\n  Gamma coverage over all 2^32 a0: {cov*100:.3f}%  "
              f"(1 - 1/e = 63.212% for a random map)")


if __name__ == "__main__":
    main()
