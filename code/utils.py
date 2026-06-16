"""Shared constants and helpers for SHA-256 cryptanalysis."""

import struct

# SHA-256 initial hash values — first 32 bits of fractional parts of square roots of first 8 primes
H0 = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]

# SHA-256 round constants — first 32 bits of fractional parts of cube roots of first 64 primes
K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]

# Rotation/shift amounts
SIGMA0_ROTATIONS = (2, 13, 22)   # Σ0(a) = ROTR2 ^ ROTR13 ^ ROTR22
SIGMA1_ROTATIONS = (6, 11, 25)   # Σ1(e) = ROTR6 ^ ROTR11 ^ ROTR25
LSIGMA0_PARAMS = (7, 18, 3)     # σ0(x) = ROTR7 ^ ROTR18 ^ SHR3
LSIGMA1_PARAMS = (17, 19, 10)   # σ1(x) = ROTR17 ^ ROTR19 ^ SHR10

MASK32 = 0xFFFFFFFF


def rotr32(x, n):
    """Right rotate 32-bit integer by n positions."""
    return ((x >> n) | (x << (32 - n))) & MASK32


def shr32(x, n):
    """Right shift 32-bit integer by n positions."""
    return (x >> n) & MASK32


def add32(*args):
    """Add multiple 32-bit integers with mod 2^32."""
    s = 0
    for a in args:
        s = (s + a) & MASK32
    return s


def ch(e, f, g):
    """Ch(e,f,g) = (e AND f) XOR (NOT e AND g)."""
    return (e & f) ^ (~e & g) & MASK32


def maj(a, b, c):
    """Maj(a,b,c) = (a AND b) XOR (a AND c) XOR (b AND c)."""
    return (a & b) ^ (a & c) ^ (b & c)


def big_sigma0(a):
    """Σ0(a) = ROTR2(a) XOR ROTR13(a) XOR ROTR22(a)."""
    return rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22)


def big_sigma1(e):
    """Σ1(e) = ROTR6(e) XOR ROTR11(e) XOR ROTR25(e)."""
    return rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25)


def small_sigma0(x):
    """σ0(x) = ROTR7(x) XOR ROTR18(x) XOR SHR3(x)."""
    return rotr32(x, 7) ^ rotr32(x, 18) ^ shr32(x, 3)


def small_sigma1(x):
    """σ1(x) = ROTR17(x) XOR ROTR19(x) XOR SHR10(x)."""
    return rotr32(x, 17) ^ rotr32(x, 19) ^ shr32(x, 10)


def pad_message(message_bytes):
    """Pad message to 512-bit blocks per SHA-256 spec.
    Returns list of blocks, each block is 16 uint32 words."""
    msg = bytearray(message_bytes)
    length_bits = len(message_bytes) * 8

    # Append bit '1' (0x80 byte)
    msg.append(0x80)

    # Pad with zeros until length ≡ 448 mod 512 (56 mod 64 bytes)
    while len(msg) % 64 != 56:
        msg.append(0x00)

    # Append original length as 64-bit big-endian
    msg += struct.pack('>Q', length_bits)

    # Split into 512-bit (64-byte) blocks of 16 uint32 words
    blocks = []
    for i in range(0, len(msg), 64):
        block = list(struct.unpack('>16I', bytes(msg[i:i+64])))
        blocks.append(block)

    return blocks


def uint32_to_bits(val, nbits=32):
    """Convert uint32 to list of bits (MSB first)."""
    return [(val >> (nbits - 1 - i)) & 1 for i in range(nbits)]


def bits_to_uint32(bits):
    """Convert list of 32 bits (MSB first) to uint32."""
    val = 0
    for b in bits:
        val = (val << 1) | b
    return val
