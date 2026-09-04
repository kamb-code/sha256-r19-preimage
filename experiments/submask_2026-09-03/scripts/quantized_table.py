#!/usr/bin/env python3
"""Quantized inversion tables: trade cheap arithmetic for scarce memory.

The table inverting u -> sigma0(u) - u stores a full 32-bit root per target,
16 GB, so three roots need an 80 GB card.  It does not have to.  Store only the
low B bits of each root and recover the rest at lookup time by trying all
2^(32-B) completions and keeping the ones that actually map back to the target:

    candidates u_i = p + i * 2^B,  i = 0 .. 2^(32-B) - 1
    accept u_i iff (sigma0(u_i) - u_i) & 0xFFFFFFFF == target

The check is exact, so the result is never wrong -- a returned root is a root by
construction.  It is also never lossy in practice: a wrong completion is a root
of the same target only with probability 2^-32, so over the 2^(32-B) - 1 wrong
completions the expected number of spurious extra roots is 2^(32-B) * 2^-32,
about 6e-8 at B = 24.  Recall of the stored root is exactly 1.

    B = 24:  12 GB per table, 256 completions per lookup
    B = 20:  10 GB per table, 4096 completions
    B = 28:  14 GB per table, 16 completions

The attack is bound by random memory reads, not arithmetic, so the extra work is
close to free on a GPU while the memory saving is what decides how many root
tables fit.  Three 24-bit tables are 36 GB and fit a 48 GB card (A6000, L40S,
RTX 6000 Ada); two fit a 24 GB card.

This validates the scheme against the real 16 GB table and measures the cost.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import M, U32, MISS, s0  # noqa: E402

TABLE = "/nvme0n1-disk/Kamvid/sigma0_u_table.npy"


def complete(part, target, B):
    """Recover full roots from B-bit partials.  Vectorised over a batch.

    part, target: uint32 arrays of equal length.
    Returns a uint32 array of the same length: the recovered root, or MISS.
    """
    out = np.full(part.shape, MISS, dtype=U32)
    lo = part & U32((1 << B) - 1)
    for i in range(1 << (32 - B)):
        u = (lo + U32(i << B)) & MISS
        hit = ((s0(u) - u) & MISS) == target
        out = np.where(hit & (out == MISS), u, out)
    return out


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    n = int(sys.argv[2]) if len(sys.argv) > 2 else (1 << 20)
    tbl = np.load(TABLE, mmap_mode="r").view(np.uint32)
    rng = np.random.default_rng(7)

    targets = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(U32)
    full = np.asarray(tbl[targets])
    have = full != MISS
    print(f"B = {B} bits stored per entry -> {(1 << 32) * B / 8 / 2**30:.1f} GB per table "
          f"(against 16.0 GB), {1 << (32 - B)} completions per lookup")
    print(f"probe: {n:,} random targets, {have.sum():,} with a stored root "
          f"({have.mean()*100:.2f}%)")

    t0 = time.time()
    rec = complete(full, targets, B)
    el = time.time() - t0

    exact = (rec == full)
    print(f"recovered the stored root exactly: {exact[have].sum():,} / {have.sum():,} "
          f"({'OK' if exact[have].all() else 'MISMATCHES'})")
    # every recovered value must genuinely invert
    ok = rec != MISS
    verify = (((s0(rec[ok]) - rec[ok]) & MISS) == targets[ok])
    print(f"every recovered root satisfies sigma0(u)-u == target: "
          f"{verify.sum():,} / {ok.sum():,} {'OK' if verify.all() else 'FAIL'}")
    # entries with no stored root must stay empty (a spurious hit is allowed but
    # must still be a genuine root; count them)
    spurious = ok & ~have
    print(f"targets with no stored root that nevertheless yielded a valid root: "
          f"{spurious.sum():,} (expected about {(1 << (32-B)) * n * 2**-32:.2g})")
    print(f"completion cost: {el:.2f}s for {n:,} lookups on one CPU core "
          f"= {n/el:.2e} lookups/s")
    print(f"  arithmetic per lookup ~ {(1 << (32-B)) * 10:,} integer ops; an H100 at "
          f"~6e13 int32 ops/s sustains ~{6e13 / ((1 << (32-B)) * 10):.2e} lookups/s,")
    print(f"  against the {1.74e9:.2e} random-read lookups/s it actually achieves -- "
          f"{'arithmetic is not the bottleneck' if 6e13/((1 << (32-B))*10) > 1.74e9 else 'ARITHMETIC WOULD DOMINATE'}")
    print()
    print("three-root memory: "
          f"{3 * (1 << 32) * B / 8 / 2**30:.0f} GB quantized vs 48 GB full "
          f"-> fits a {'48' if 3 * (1 << 32) * B / 8 / 2**30 <= 46 else '80'} GB card")


if __name__ == "__main__":
    main()
