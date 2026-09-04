#!/usr/bin/env python3
"""
INDEPENDENT SHA-256 implementation, written from FIPS 180-4 only.
No imports from, and no code copied out of, /home/administrator/sha or the
scratchpad scripts.  Constants are DERIVED from the specification
(square roots / cube roots of the first primes), not transcribed.

Then: reduced-round compression digest  h = IV + compress_R(IV, W0..W15)
and audit of the three claimed R=19 preimages.
"""

import hashlib
import os
import struct

M32 = 0xFFFFFFFF


# ----------------------------------------------------------------------------
# 0. Derive the constants from the spec (do not transcribe them)
# ----------------------------------------------------------------------------
def primes(n):
    out, c = [], 2
    while len(out) < n:
        if all(c % p for p in out if p * p <= c):
            out.append(c)
        c += 1
    return out


def iroot(x, k):
    """floor(x ** (1/k)) for integers, exactly."""
    if x < 2:
        return x
    lo, hi = 1, 1 << ((x.bit_length() + k - 1) // k + 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid ** k <= x:
            lo = mid
        else:
            hi = mid - 1
    return lo


P = primes(64)
# IV: first 32 bits of the fractional parts of the square roots of primes 1..8
IV = tuple((iroot(p << 64, 2)) & M32 for p in P[:8])
# K: first 32 bits of the fractional parts of the cube roots of primes 1..64
K = tuple((iroot(p << 96, 3)) & M32 for p in P[:64])


# ----------------------------------------------------------------------------
# 1. Core primitives (FIPS 180-4 sec. 4.1.2)
# ----------------------------------------------------------------------------
def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & M32


def shr(x, n):
    return (x & M32) >> n


def Ch(x, y, z):
    return ((x & y) ^ ((x ^ M32) & z)) & M32


def Maj(x, y, z):
    return (x & y) ^ (x & z) ^ (y & z)


def BSig0(x):                      # Sigma0  (state)
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)


def BSig1(x):                      # Sigma1  (state)
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)


def sSig0(x):                      # sigma0  (schedule)
    return rotr(x, 7) ^ rotr(x, 18) ^ shr(x, 3)


def sSig1(x):                      # sigma1  (schedule)
    return rotr(x, 17) ^ rotr(x, 19) ^ shr(x, 10)


# ----------------------------------------------------------------------------
# 2. Message schedule + compression, parameterised by round count R
# ----------------------------------------------------------------------------
def expand(W16, R):
    """Full schedule W_0..W_{R-1} from the 16 given words."""
    W = list(W16)
    for t in range(16, R):
        W.append((sSig1(W[t - 2]) + W[t - 7] + sSig0(W[t - 15]) + W[t - 16]) & M32)
    return W


def compress(H, W16, R=64, trace=False):
    """Returns (digest_words, trace) where digest = H + state after R rounds."""
    W = expand(W16, max(R, 16))
    a, b, c, d, e, f, g, h = H
    a_hist, e_hist = {}, {}
    a_hist[-1], a_hist[-2], a_hist[-3], a_hist[-4] = H[0], H[1], H[2], H[3]
    e_hist[-1], e_hist[-2], e_hist[-3], e_hist[-4] = H[4], H[5], H[6], H[7]
    for t in range(R):
        T1 = (h + BSig1(e) + Ch(e, f, g) + K[t] + W[t]) & M32
        T2 = (BSig0(a) + Maj(a, b, c)) & M32
        h, g, f, e = g, f, e, (d + T1) & M32
        d, c, b, a = c, b, a, (T1 + T2) & M32
        a_hist[t], e_hist[t] = a, e
    digest = tuple((x + y) & M32 for x, y in zip(H, (a, b, c, d, e, f, g, h)))
    if trace:
        return digest, a_hist, e_hist
    return digest


def hexdig(words):
    return "".join("%08x" % w for w in words)


# ----------------------------------------------------------------------------
# 3. Self-test at 64 rounds against hashlib
# ----------------------------------------------------------------------------
def selftest():
    # constants
    assert IV[0] == 0x6A09E667 and IV[7] == 0x5BE0CD19, [hex(x) for x in IV]
    assert K[0] == 0x428A2F98 and K[63] == 0xC67178F2, hex(K[63])

    # empty string, one padded block
    blk = b"\x80" + b"\x00" * 62 + b"\x00"
    blk = bytearray(64)
    blk[0] = 0x80
    W = list(struct.unpack(">16I", bytes(blk)))
    assert hexdig(compress(IV, W, 64)) == hashlib.sha256(b"").hexdigest()

    # random short messages (single padded block)
    for _ in range(400):
        n = os.urandom(1)[0] % 56
        msg = os.urandom(n)
        pad = msg + b"\x80" + b"\x00" * (55 - n) + struct.pack(">Q", 8 * n)
        assert len(pad) == 64
        W = list(struct.unpack(">16I", pad))
        assert hexdig(compress(IV, W, 64)) == hashlib.sha256(msg).hexdigest()

    # two-block message, chained, to exercise a non-IV chaining value
    msg = os.urandom(100)
    pad = msg + b"\x80" + b"\x00" * (119 - len(msg)) + struct.pack(">Q", 8 * len(msg))
    assert len(pad) == 128
    H = IV
    for i in (0, 64):
        H = compress(H, list(struct.unpack(">16I", pad[i:i + 64])), 64)
    assert hexdig(H) == hashlib.sha256(msg).hexdigest()
    print("SELF-TEST OK: derived IV/K correct; 64-round compression matches "
          "hashlib on 402 messages (1- and 2-block).")


# ----------------------------------------------------------------------------
# 4. The claimed preimages
# ----------------------------------------------------------------------------
CASES = [
    ("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
     "955105a3 4013e529 1b25f747 32d49736 686c1d29 11badccf 6367aa18 1ea9aecc "
     "dcd580e6 82244f11 ed5f5e1f 8931971b e1aecc0c 1b6b1a39 7226607e 2b9bc7a3"),
    ("0000000000000000000000000000000000000000000000000000000000000000",
     "ae7fef2c fe70ad27 9243d601 9c80f4fa d5928838 9e6f5549 e53c9b31 e5ad11d5 "
     "aaa545a8 376d065d 9ed81fe3 1f7c88b8 3e580ca0 0b4accdf 9dd81f31 4181bb1e"),
    ("bd4ed1b236c94e728eac234e57ce2371169f94aedb5aecad4b589fee1ccd9ce9",
     "5f95b1b6 66a8ded7 114daecb d0ccbc7a a25a7a9e 5f3b6807 264758a0 bae5dd72 "
     "3999f89e 44a10243 6927763e 630f21cd 26ffc552 07bb55bc fee16d43 f4578256"),
]


def main():
    selftest()
    print()
    allok = True
    for idx, (target, wtxt) in enumerate(CASES, 1):
        W = [int(x, 16) for x in wtxt.split()]
        assert len(W) == 16
        print("=" * 74)
        print("CASE %d  target %s" % (idx, target))
        for R in (18, 19, 20):
            d = hexdig(compress(IV, W, R))
            mark = "MATCH  <<<" if d == target else "no match"
            print("   R=%2d  %s   %s" % (R, d, mark))
            if R == 19 and d != target:
                allok = False
            if R in (18, 20) and d == target:
                allok = False
        # state extraction at R=19
        _, ah, eh = compress(IV, W, 19, trace=True)
        a4, a5, e8, e9 = ah[4], ah[5], eh[8], eh[9]
        print("   state: a4=%08x a5=%08x  e8=%08x e9=%08x" % (a4, a5, e8, e9))
        print("   a4==a5: %s   a4==a5==0: %s   e8==e9==0xFFFFFFFF: %s"
              % (a4 == a5, a4 == a5 == 0, e8 == e9 == 0xFFFFFFFF))
        # the claimed residual identity eps = Maj(v,a3,a2) - a3 == 0
        a2, a3 = ah[2], ah[3]
        eps = (Maj(a4, a3, a2) - a3) & M32
        print("   a2=%08x a3=%08x   Maj(a4,a3,a2)-a3 = %08x  (zero: %s)"
              % (a2, a3, eps, eps == 0))
        # with a4=a5=0, Maj(0,a3,a2) = a3 & a2, so eps==0 <=> a3 submask of a2
        print("   a3 & a2 == a3 (a3 submask of a2): %s" % (((a3 & a2) == a3)))
        # full 64-round hash of the same block, for contrast
        print("   (full 64-round compression of same W: %s)"
              % hexdig(compress(IV, W, 64)))
    print("=" * 74)
    print("VERDICT: all three valid at R=19 and invalid at R=18,20:", allok)


if __name__ == "__main__":
    main()
