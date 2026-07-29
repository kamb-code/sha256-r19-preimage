"""Extended reduced-round SHA-256 pre-image solver for R=2..20+.

KEY INSIGHT: For R ≤ 16, the free a-values are UNCONSTRAINED.
Any choice produces a valid pre-image. No brute force needed!

For R = 2..8:   Backward chain reaches a_0. All W recovered exactly.
For R = 9..16:  Free a-values (a_0..a_{R-9}). CHOOSE any value.
                Compute e's from a's, then W's from a's and e's.
                All W[0]..W[R-1] recovered. W[R]..W[15] are free.
                No brute force! Still O(1)!

For R = 17+:    W[16]+ from message schedule creates NONLINEAR constraints.
                The chosen a-values must produce W[0]..W[15] such that
                the schedule expansion is consistent with the state evolution.
                This is where the real cryptographic hardness begins.

The boundary is R=16, not R=8!
"""

import os
import struct
import time
import numpy as np
from tqdm import tqdm

from sha256_core import sha256_full_trace
from utils import (H0, K, MASK32, add32, big_sigma0, big_sigma1,
                   ch, maj, small_sigma0, small_sigma1, pad_message)


def get_iv(kind, idx):
    """Get IV value. kind='a': a_{idx}, kind='e': e_{idx}. idx in [-4,-1]."""
    if kind == 'a':
        return H0[-idx - 1] if -4 <= idx <= -1 else None
    else:
        return H0[4 + (-idx - 1)] if -4 <= idx <= -1 else None


def backward_chain(hash_bytes, R):
    """Recover a-values from hash via backward chain.

    Returns known_a dict (from shift register + chain).
    """
    hash_words = struct.unpack('>8I', hash_bytes[:32])
    final_state = [(hash_words[i] - H0[i]) & MASK32 for i in range(8)]

    known_a = {}
    known_e = {}

    # Shift register: 4 a-values, 4 e-values
    for offset in range(min(4, R)):
        r = R - 1 - offset
        known_a[r] = final_state[offset]
        known_e[r] = final_state[4 + offset]

    # IV
    for idx in range(-4, 0):
        known_a[idx] = get_iv('a', idx)
        known_e[idx] = get_iv('e', idx)

    # Backward chain: extend 4 more a-values using the 4 known e-values
    for r in range(R - 1, max(-1, R - 5), -1):
        target = r - 4
        if target in known_a:
            continue
        if not all(k in known_a for k in [r, r-1, r-2, r-3]):
            break
        if r not in known_e:
            break

        T2 = (big_sigma0(known_a[r-1]) + maj(known_a[r-1], known_a[r-2], known_a[r-3])) & MASK32
        T1 = (known_a[r] - T2) & MASK32
        known_a[target] = (known_e[r] - T1) & MASK32

    return known_a, known_e


def compute_e_from_a(known_a, R):
    """Compute all e-values from a-values.

    e_r = a_r + a_{r-4} - Σ0(a_{r-1}) - Maj(a_{r-1}, a_{r-2}, a_{r-3})
    """
    known_e = {}
    for idx in range(-4, 0):
        known_e[idx] = get_iv('e', idx)

    for r in range(R):
        needed = [r, r-1, r-2, r-3, r-4]
        if all(n in known_a for n in needed):
            T2 = (big_sigma0(known_a[r-1]) + maj(known_a[r-1], known_a[r-2], known_a[r-3])) & MASK32
            T1 = (known_a[r] - T2) & MASK32
            known_e[r] = (known_a[r-4] + T1) & MASK32

    return known_e


def recover_W(known_a, known_e, R):
    """Recover W values from known a and e values."""
    W = {}
    for r in range(R):
        needed_a = [r, r-1, r-2, r-3]
        needed_e = [r-1, r-2, r-3, r-4]
        if not all(n in known_a for n in needed_a):
            continue
        if not all(n in known_e for n in needed_e):
            continue

        T2 = (big_sigma0(known_a[r-1]) + maj(known_a[r-1], known_a[r-2], known_a[r-3])) & MASK32
        T1 = (known_a[r] - T2) & MASK32
        W[r] = (T1 - known_e[r-4] - big_sigma1(known_e[r-1])
                - ch(known_e[r-1], known_e[r-2], known_e[r-3]) - K[r]) & MASK32
    return W


def solve_preimage(hash_bytes, R, free_a_values=None):
    """Full pre-image solver for R-round SHA-256.

    Args:
        hash_bytes: 32-byte hash
        R: number of rounds
        free_a_values: dict {round: value} for the free a-parameters.
                       If None, uses random values.

    Returns:
        message_words: list of 16 uint32 (the message block), or None
        info: dict with solver details
    """
    # Step 1: Backward chain
    known_a, known_e_shift = backward_chain(hash_bytes, R)

    # Step 2: Determine which a-values are free
    chain_min = min(known_a.keys())  # Lowest recovered a-index
    free_indices = [r for r in range(max(0, chain_min), R) if r not in known_a]

    # Step 3: Fill free a-values
    if free_a_values is None:
        free_a_values = {}
    for r in free_indices:
        if r not in free_a_values:
            # Default: random value
            free_a_values[r] = int.from_bytes(os.urandom(4), 'big')
        known_a[r] = free_a_values[r]

    # Step 4: Compute all e-values from a-values
    known_e = compute_e_from_a(known_a, R)

    # Step 5: Recover W values
    W = recover_W(known_a, known_e, R)

    # Step 6: Build message block
    message_words = [0] * 16
    for r in range(min(R, 16)):
        if r in W:
            message_words[r] = W[r]
    # W[R]..W[15] are free (unused in computation)
    for r in range(R, 16):
        message_words[r] = 0  # or any value

    # Step 7: For R >= 17, check message schedule consistency
    schedule_ok = True
    schedule_violations = []
    if R > 16:
        full_W = list(message_words)
        for t in range(16, R):
            expanded = (small_sigma1(full_W[t-2]) + full_W[t-7]
                       + small_sigma0(full_W[t-15]) + full_W[t-16]) & MASK32
            if t in W:
                if expanded != W[t]:
                    schedule_ok = False
                    schedule_violations.append(t)
            full_W.append(expanded)

    return message_words, {
        'free_indices': free_indices,
        'num_free': len(free_indices),
        'W_recovered': sorted(W.keys()),
        'num_W': len(W),
        'schedule_ok': schedule_ok,
        'schedule_violations': schedule_violations,
    }


def verify_preimage(message_words, hash_bytes, R):
    """Verify a candidate pre-image by forward computation."""
    # Build message bytes from words
    msg_bytes = struct.pack('>16I', *message_words)

    # Run R-round SHA-256 with this as a raw block
    trace = sha256_full_trace(msg_bytes[:55], num_rounds=R)

    # But we need to use the EXACT message_words, not padded.
    # Re-run with direct block injection.
    from sha256_core import sha256_message_schedule
    W = sha256_message_schedule(message_words)

    a, b, c, d, e, f, g, hh = H0
    for t in range(R):
        s1 = big_sigma1(e)
        ch_val = ch(e, f, g)
        temp1 = add32(hh, s1, ch_val, K[t], W[t])
        s0 = big_sigma0(a)
        maj_val = maj(a, b, c)
        temp2 = add32(s0, maj_val)
        hh = g
        g = f
        f = e
        e = add32(d, temp1)
        d = c
        c = b
        b = a
        a = add32(temp1, temp2)

    final = [add32(H0[i], v) for i, v in enumerate([a, b, c, d, e, f, g, hh])]
    computed_hash = struct.pack('>8I', *final)

    return computed_hash == hash_bytes


def run_tests():
    """Test the extended solver across all round counts."""
    print("=" * 70)
    print("EXTENDED SHA-256 PRE-IMAGE SOLVER")
    print("=" * 70)
    print()
    print("Testing R=2 through R=20...")
    print()

    results = {}

    for R in range(2, 21):
        num_tests = 500 if R <= 16 else 100
        correct = 0
        w_counts = []
        free_counts = []
        schedule_pass = 0

        for _ in tqdm(range(num_tests), desc=f"R={R:2d}", leave=True):
            msg = os.urandom(55)
            trace = sha256_full_trace(msg, num_rounds=R)
            hash_bytes = trace.final_hash

            # Solve
            message_words, info = solve_preimage(hash_bytes, R)
            w_counts.append(info['num_W'])
            free_counts.append(info['num_free'])

            if R <= 16 or info['schedule_ok']:
                schedule_pass += 1

            # Verify: does our pre-image actually hash to the target?
            if verify_preimage(message_words, hash_bytes, R):
                correct += 1

            # For R <= 8: also check W values match original message
            if R <= 8:
                for r in info['W_recovered']:
                    if r < 16:
                        assert message_words[r] == trace.message_schedule[r], \
                            f"W[{r}] mismatch at R={R}"

        results[R] = {
            'correct': correct,
            'total': num_tests,
            'avg_W': np.mean(w_counts),
            'avg_free': np.mean(free_counts),
            'schedule_pass': schedule_pass,
        }

        status = "PASS" if correct == num_tests else f"FAIL({correct}/{num_tests})"
        sched = "" if R <= 16 else f" sched={schedule_pass}/{num_tests}"
        print(f"        → {status}: {correct}/{num_tests} verified, "
              f"W={np.mean(w_counts):.0f}, free_a={np.mean(free_counts):.0f}{sched}")

    # Summary table
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'R':>3} | {'Verified':>10} | {'W recovered':>12} | {'Free a':>7} | {'Sched OK':>9} | {'Method':>20}")
    print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*12}-+-{'-'*7}-+-{'-'*9}-+-{'-'*20}")

    for R in sorted(results.keys()):
        r = results[R]
        pct = r['correct'] / r['total'] * 100
        verified = f"{r['correct']}/{r['total']}"
        w_rec = f"{r['avg_W']:.0f}/{R}"
        free_a = f"{r['avg_free']:.0f}"
        sched = "N/A" if R <= 16 else f"{r['schedule_pass']}/{r['total']}"
        if R <= 8:
            method = "chain (exact)"
        elif R <= 16:
            method = "chain + free a"
        else:
            method = "chain + free a + sched"
        print(f"  {R:>3} | {verified:>10} | {w_rec:>12} | {free_a:>7} | {sched:>9} | {method:>20}")

    # Analysis of R=17+ boundary
    print()
    print("=" * 70)
    print("R=17+ ANALYSIS: MESSAGE SCHEDULE CONSTRAINT")
    print("=" * 70)
    print()
    print("  For R ≤ 16: ALL message words are direct (W[r] = M[r]).")
    print("  Any choice of free a-values produces a valid pre-image.")
    print("  The solver is O(1) — no search needed.")
    print()
    print("  For R = 17: W[16] = σ1(W[14]) + W[9] + σ0(W[1]) + W[0]")
    print("  This expanded word must be consistent with the round-16 equation.")
    print("  The round-16 equation is a nonlinear constraint on the free a-values.")
    print("  Each additional round past 16 adds one more constraint.")
    print()
    print("  Free a-values:    max(0, R-8) values = max(0, R-8) × 32 bits")
    print("  Schedule constraints: max(0, R-16) equations = max(0, R-16) × 32 bits")
    print("  Net freedom:      always 256 bits (= hash output size)")
    print()
    print("  For R=17: 9 free a-values, 1 constraint → 8 effective free (256 bits)")
    print("  For R=64: 56 free a-values, 48 constraints → 8 effective free (256 bits)")
    print("  But the constraints are NONLINEAR — finding a solution requires solving")
    print("  a system of nonlinear equations over GF(2^32).")


if __name__ == '__main__':
    t0 = time.time()
    run_tests()
    print(f"\nTotal time: {time.time() - t0:.1f}s")
