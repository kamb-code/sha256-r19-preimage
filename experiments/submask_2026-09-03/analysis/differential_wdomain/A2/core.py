"""Vectorised reduced-round SHA-256 compression (numpy uint32) + diff laws."""
import numpy as np

M = np.uint32(0xFFFFFFFF)

H0 = np.array([0x6A09E667,0xBB67AE85,0x3C6EF372,0xA54FF53A,
               0x510E527F,0x9B05688C,0x1F83D9AB,0x5BE0CD19], dtype=np.uint32)

K = np.array([
 0x428A2F98,0x71374491,0xB5C0FBCF,0xE9B5DBA5,0x3956C25B,0x59F111F1,0x923F82A4,0xAB1C5ED5,
 0xD807AA98,0x12835B01,0x243185BE,0x550C7DC3,0x72BE5D74,0x80DEB1FE,0x9BDC06A7,0xC19BF174,
 0xE49B69C1,0xEFBE4786,0x0FC19DC6,0x240CA1CC,0x2DE92C6F,0x4A7484AA,0x5CB0A9DC,0x76F988DA,
 0x983E5152,0xA831C66D,0xB00327C8,0xBF597FC7,0xC6E00BF3,0xD5A79147,0x06CA6351,0x14292967,
 0x27B70A85,0x2E1B2138,0x4D2C6DFC,0x53380D13,0x650A7354,0x766A0ABB,0x81C2C92E,0x92722C85,
 0xA2BFE8A1,0xA81A664B,0xC24B8B70,0xC76C51A3,0xD192E819,0xD6990624,0xF40E3585,0x106AA070,
 0x19A4C116,0x1E376C08,0x2748774C,0x34B0BCB5,0x391C0CB3,0x4ED8AA4A,0x5B9CCA4F,0x682E6FF3,
 0x748F82EE,0x78A5636F,0x84C87814,0x8CC70208,0x90BEFFFA,0xA4506CEB,0xBEF9A3F7,0xC67178F2],
 dtype=np.uint32)


def rotr(x, n):
    n = np.uint32(n)
    return ((x >> n) | (x << np.uint32(32 - n))).astype(np.uint32)


def S0(x): return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
def S1(x): return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
def s0(x): return rotr(x, 7) ^ rotr(x, 18) ^ (x >> np.uint32(3))
def s1(x): return rotr(x, 17) ^ rotr(x, 19) ^ (x >> np.uint32(10))
def ch(x, y, z): return (x & y) ^ (~x & z)
def maj(x, y, z): return (x & y) ^ (x & z) ^ (y & z)


def expand(W, rounds):
    """W: (n,16) uint32 -> (n,rounds) uint32"""
    n = W.shape[0]
    out = np.zeros((n, max(rounds, 16)), dtype=np.uint32)
    out[:, :16] = W
    for t in range(16, rounds):
        out[:, t] = (s1(out[:, t-2]) + out[:, t-7] + s0(out[:, t-15]) + out[:, t-16])
    return out[:, :rounds]


def compress(W, rounds, iv=None):
    """W: (n,16) uint32 -> digest (n,8) uint32, feed-forward included."""
    Wx = expand(W, rounds)
    n = W.shape[0]
    if iv is None:
        iv = H0
    st = [np.full(n, iv[i], dtype=np.uint32) for i in range(8)]
    a, b, c, d, e, f, g, h = st
    for t in range(rounds):
        t1 = (h + S1(e) + ch(e, f, g) + K[t] + Wx[:, t]).astype(np.uint32)
        t2 = (S0(a) + maj(a, b, c)).astype(np.uint32)
        h, g, f, e = g, f, e, (d + t1).astype(np.uint32)
        d, c, b, a = c, b, a, (t1 + t2).astype(np.uint32)
    fin = [a, b, c, d, e, f, g, h]
    return np.stack([(fin[i] + iv[i]).astype(np.uint32) for i in range(8)], axis=1)


def popcnt32(x):
    x = x.astype(np.uint32)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 4), axis=1).sum(axis=1)
