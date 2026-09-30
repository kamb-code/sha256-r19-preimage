#!/usr/bin/env python3
"""Independent verifier for raw-block reduced-round SHA-256 compression.

This deliberately does not import the attack code.  It verifies claims of the
form:

  SHA256_compress_R(IV, W0..W15) + IV == target

The block is interpreted as sixteen arbitrary big-endian 32-bit words.  This is
the reduced-round one-block compression convention used by the research code,
not the standard padded SHA-256 message-hash convention.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Iterable

MASK32 = 0xFFFFFFFF

H0 = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
]

K = [
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
]


def rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & MASK32


def big_s0(x: int) -> int:
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)


def big_s1(x: int) -> int:
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)


def small_s0(x: int) -> int:
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)


def small_s1(x: int) -> int:
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)


def ch(x: int, y: int, z: int) -> int:
    return ((x & y) ^ (~x & z)) & MASK32


def maj(x: int, y: int, z: int) -> int:
    return (x & y) ^ (x & z) ^ (y & z)


def parse_hex_words(text: str, expected: int | None = None) -> list[int]:
    values = [int(tok, 16) for tok in re.findall(r"(?:0x)?[0-9a-fA-F]{1,8}", text)]
    if expected is not None and len(values) != expected:
        raise SystemExit(f"expected {expected} words, got {len(values)}")
    return [v & MASK32 for v in values]


def parse_hash(text: str) -> list[int]:
    compact = re.sub(r"[^0-9a-fA-F]", "", text)
    if len(compact) == 64:
        return [int(compact[i:i + 8], 16) for i in range(0, 64, 8)]
    return parse_hex_words(text, expected=8)


def sha256_reduced_raw_block(words: Iterable[int], rounds: int) -> list[int]:
    w = [int(x) & MASK32 for x in words]
    if len(w) != 16:
        raise ValueError("exactly 16 message words are required")
    if not (0 <= rounds <= 64):
        raise ValueError("round count must be between 0 and 64")

    for t in range(16, rounds):
        w.append((small_s1(w[t - 2]) + w[t - 7] + small_s0(w[t - 15]) + w[t - 16]) & MASK32)

    a, b, c, d, e, f, g, h = H0
    for t in range(rounds):
        t1 = (h + big_s1(e) + ch(e, f, g) + K[t] + w[t]) & MASK32
        t2 = (big_s0(a) + maj(a, b, c)) & MASK32
        h, g, f, e = g, f, e, (d + t1) & MASK32
        d, c, b, a = c, b, a, (t1 + t2) & MASK32

    return [(H0[i] + v) & MASK32 for i, v in enumerate([a, b, c, d, e, f, g, h])]


def fmt_words(words: Iterable[int]) -> str:
    return " ".join(f"{int(w) & MASK32:08x}" for w in words)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", "-r", type=int, required=True)
    parser.add_argument("--words", "-w", required=True, help="sixteen 32-bit words")
    parser.add_argument("--hash", "-t", dest="target", help="target hash as 64 hex chars or 8 words")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    words = parse_hex_words(args.words, expected=16)
    computed = sha256_reduced_raw_block(words, args.rounds)
    target = parse_hash(args.target) if args.target else None
    ok = target == computed if target is not None else None

    if args.json:
        print(json.dumps({
            "rounds": args.rounds,
            "computed": fmt_words(computed),
            "target": fmt_words(target) if target is not None else None,
            "ok": ok,
        }, indent=2))
        return

    print(f"rounds:   {args.rounds}")
    print(f"computed: {fmt_words(computed)}")
    if target is not None:
        print(f"target:   {fmt_words(target)}")
        print("result:   " + ("OK" if ok else "FAIL"))


if __name__ == "__main__":
    main()
