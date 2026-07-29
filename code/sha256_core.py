"""Phase 1: SHA-256 reference implementation with full state exposure.

Every internal state at every round and every sub-operation is captured
for cryptanalysis purposes.
"""

import hashlib
import os
import struct
from dataclasses import dataclass, field

from utils import (
    H0, K, MASK32, add32, big_sigma0, big_sigma1, ch, maj,
    pad_message, small_sigma0, small_sigma1,
)


@dataclass
class SubOpState:
    """Intermediate values within a single round."""
    ch_out: int = 0
    maj_out: int = 0
    sigma0_out: int = 0  # Σ0(a)
    sigma1_out: int = 0  # Σ1(e)
    temp1: int = 0
    temp2: int = 0
    w_t: int = 0         # message schedule word for this round


@dataclass
class TraceResult:
    """Full execution trace of SHA-256 on a single 512-bit block."""
    round_states: list = field(default_factory=list)   # [64][8] — (a,b,c,d,e,f,g,h) after each round
    sub_op_states: list = field(default_factory=list)   # [64] SubOpState
    message_schedule: list = field(default_factory=list) # [64] W values
    initial_state: list = field(default_factory=list)    # [8] — H0 or chained state
    final_hash: bytes = b''
    final_state: list = field(default_factory=list)      # [8] — state after adding H0


def sha256_message_schedule(block):
    """Expand 16-word block to 64-word message schedule."""
    W = list(block[:16])
    for t in range(16, 64):
        w = add32(small_sigma1(W[t-2]), W[t-7], small_sigma0(W[t-15]), W[t-16])
        W.append(w)
    return W


def sha256_full_trace(message_bytes, num_rounds=64):
    """Compute SHA-256 with full state trace.

    Args:
        message_bytes: Input message as bytes.
        num_rounds: Number of rounds (default 64, can be reduced for testing).

    Returns:
        TraceResult with complete internal state history.
    """
    blocks = pad_message(message_bytes)

    # For multi-block messages, chain the state
    h = list(H0)

    # We trace only the last block for simplicity in analysis
    # (most analysis uses single-block messages anyway)
    all_traces = []

    for block in blocks:
        trace = TraceResult()
        trace.initial_state = list(h)

        # Message schedule
        W = sha256_message_schedule(block)
        trace.message_schedule = list(W)

        # Working variables
        a, b, c, d, e, f, g, hh = h

        for t in range(num_rounds):
            sub = SubOpState()

            sub.sigma1_out = big_sigma1(e)
            sub.ch_out = ch(e, f, g)
            sub.w_t = W[t]
            sub.temp1 = add32(hh, sub.sigma1_out, sub.ch_out, K[t], W[t])

            sub.sigma0_out = big_sigma0(a)
            sub.maj_out = maj(a, b, c)
            sub.temp2 = add32(sub.sigma0_out, sub.maj_out)

            # Update working variables
            hh = g
            g = f
            f = e
            e = add32(d, sub.temp1)
            d = c
            c = b
            b = a
            a = add32(sub.temp1, sub.temp2)

            trace.round_states.append([a, b, c, d, e, f, g, hh])
            trace.sub_op_states.append(sub)

        # Add to initial hash values
        final = [add32(h[i], [a, b, c, d, e, f, g, hh][i]) for i in range(8)]
        trace.final_state = final
        trace.final_hash = struct.pack('>8I', *final)

        h = final
        all_traces.append(trace)

    # Return trace of last block
    result = all_traces[-1]
    result.final_hash = struct.pack('>8I', *h)
    return result


def sha256_hash(message_bytes):
    """Simple SHA-256 hash returning bytes (no trace)."""
    trace = sha256_full_trace(message_bytes)
    return trace.final_hash


def validate(num_tests=1000):
    """Validate against hashlib.sha256."""
    import random
    random.seed(42)

    passed = 0
    for i in range(num_tests):
        # Random message of random length (0 to 128 bytes)
        length = random.randint(0, 128)
        msg = os.urandom(length)

        our_hash = sha256_hash(msg)
        ref_hash = hashlib.sha256(msg).digest()

        if our_hash == ref_hash:
            passed += 1
        else:
            print(f"MISMATCH at test {i}: len={length}")
            print(f"  ours: {our_hash.hex()}")
            print(f"  ref:  {ref_hash.hex()}")

    print(f"Validation: {passed}/{num_tests} passed")
    return passed == num_tests


if __name__ == '__main__':
    # Quick test
    test_msg = b"abc"
    result = sha256_full_trace(test_msg)
    print(f"SHA-256('abc') = {result.final_hash.hex()}")
    print(f"Expected:        ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    print()
    validate()
