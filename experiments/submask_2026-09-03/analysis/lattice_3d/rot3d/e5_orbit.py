#!/usr/bin/env python3
"""E5: the orbit / stabiliser argument, and a concrete test on the published
20-round preimages.

A symmetry sigma of the compression maps a preimage of H to a preimage of
sigma(H).  For a FIXED target it buys nothing unless sigma(H) = H.  Because of
the feed-forward h_i = IV_i + s_i, a bit-rotation of the STATE acts on the
digest as   h_i -> rot_g(h_i - IV_i) + IV_i,   which fixes H iff every
s_i = h_i - IV_i is rotation-invariant of period gcd(g,32).
"""
import struct

M = 0xFFFFFFFF
IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
K = [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
     0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
     0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
     0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da]


def rotr(x, n): n %= 32; return ((x >> n) | (x << (32 - n))) & M
def rotl(x, n): return rotr(x, (32 - n % 32) % 32)
def S0(x): return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
def S1(x): return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
def s0(x): return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)
def s1(x): return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)
def Ch(e, f, g): return ((e & f) ^ (~e & g)) & M
def Maj(a, b, c): return (a & b) ^ (a & c) ^ (b & c)


def compress(W, R):
    Wf = list(W)
    for t in range(16, R):
        Wf.append((s1(Wf[t - 2]) + Wf[t - 7] + s0(Wf[t - 15]) + Wf[t - 16]) & M)
    a = {-1: IV[0], -2: IV[1], -3: IV[2], -4: IV[3]}
    e = {-1: IV[4], -2: IV[5], -3: IV[6], -4: IV[7]}
    for r in range(R):
        T1 = (e[r-4] + S1(e[r-1]) + Ch(e[r-1], e[r-2], e[r-3]) + K[r] + Wf[r]) & M
        a[r] = (T1 + S0(a[r-1]) + Maj(a[r-1], a[r-2], a[r-3])) & M
        e[r] = (a[r-4] + T1) & M
    return a, e, Wf


def digest(W, R):
    a, e, _ = compress(W, R)
    st = [a[R-1], a[R-2], a[R-3], a[R-4], e[R-1], e[R-2], e[R-3], e[R-4]]
    return [(x + y) & M for x, y in zip(st, IV)]


def hexs(v): return "".join(f"{x:08x}" for x in v)


P1 = [0xa36f4238, 0xf2c9204e, 0xb5b6653b, 0x070401f7, 0x5928e4f3, 0xfe766be2,
      0x52026907, 0xf7ee1812, 0x01344603, 0xea505012, 0x86cbe6cb, 0x6ce73d0b,
      0x5c91de6a, 0x43355e3b, 0xff3d5e88, 0xc1ad0b54]
T1H = "d962ca30635f9b74ac6c8c1243a1a9cf800e81bc05f1d2e40764c68c795f7388"
P2 = [0x0b0187bf, 0xb9aea692, 0xb66effa5, 0x087dca3a, 0x0caea827, 0x1d2f9916,
      0x739a224e, 0xa87c0eae, 0x7f9ef4a7, 0xb318a7de, 0xa8848c61, 0xa7a141a4,
      0x11ac114b, 0x952299be, 0x97aaa67f, 0xc6acfe57]
T2H = "ff" * 32

print("=== sanity: the two published 20-round preimages verify ===")
for nm, W, T in (("solver digest", P1, T1H), ("all-ones", P2, T2H)):
    got = hexs(digest(W, 20))
    print(f"  {nm:<14} recomputed digest == published target: {got == T}")

print()
print("=== stabiliser of a target under a bit-rotation of the state ===")
print("required: s_i = h_i - IV_i is invariant under rot_g, for all 8 words")
for nm, T in (("solver digest", T1H), ("all-ones ff..ff", T2H)):
    h = [int(T[8*i:8*i+8], 16) for i in range(8)]
    s = [(x - IV[i]) & M for i, x in enumerate(h)]
    fixed = [g for g in range(1, 32) if all(rotl(x, g) == x for x in s)]
    print(f"  {nm:<16}: s = {[f'{x:08x}' for x in s]}")
    print(f"  {'':<16}  nontrivial g fixing the target: {fixed}  -> stabiliser size "
          f"{1 + len(fixed)}")

print()
print("  For the all-ones digest s_i = ~IV_i.  Is any ~IV_i rotation-invariant?")
for i, v in enumerate(IV):
    c = (~v) & M
    inv = [g for g in range(1, 32) if rotl(c, g) == c]
    print(f"    ~IV[{i}] = {c:08x}: nontrivial rotational periods {inv}")

print()
print("=== how many targets have a nontrivial rotational stabiliser? ===")
for g in (1, 2, 4, 8, 16):
    d = 32 // (32 // __import__('math').gcd(g, 32))  # period = gcd(g,32)
    per = __import__('math').gcd(g, 32)
    print(f"  g={g:2d}: period gcd(g,32)={per}; each s_i has 2^{per} choices; "
          f"fraction of digests fixed = 2^-{8*(32-per)}")

print()
print("=== concrete: rotate a published preimage and count surviving constraints ===")
for nm, W, T in (("solver digest", P1, T1H), ("all-ones", P2, T2H)):
    a, e, Wf = compress(W, 20)
    print(f"  {nm}:")
    for g in (1, 16, 31):
        Wr = [rotl(x, g) for x in W]
        ar, er, Wfr = compress(Wr, 20)
        sch = sum(1 for t in range(16, 20) if Wfr[t] == rotl(Wf[t], g))
        st = sum(1 for r in range(20) if ar[r] == rotl(a[r], g)) \
            + sum(1 for r in range(20) if er[r] == rotl(e[r], g))
        hr = hexs(digest(Wr, 20))
        want = hexs([(rotl((x - IV[i]) & M, g) + IV[i]) & M
                     for i, x in enumerate([int(T[8*i:8*i+8], 16) for i in range(8)])])
        print(f"    g={g:2d}: expanded words matching rot: {sch}/4;  "
              f"state words matching rot: {st}/40;  "
              f"digest == rotated target: {hr == want}")
