#!/usr/bin/env python3
"""Build the sigma0(u)-u representative table as a .npy file (CPU, 16 GiB).

The GPU solver (h100_extended.py) rebuilds this table in GPU memory on every
run and never writes it out.  The CPU analysis scripts
(measure_w9_filters.py, context_screening.py, screening_validation.py) read
it from disk, so this utility produces that file.

Format (the one the CPU scripts expect, and that of the published table):
    numpy array of 2**32 int32 values, indexed by v = (sigma0(u) - u) mod 2**32,
    holding one representative u with that difference (as the int32 with the
    same bit pattern), or -1, i.e. 0xFFFFFFFF, when no u maps to v.  The CPU
    scripts read it with .view(np.uint32).  Coverage is
    2,721,603,628 / 2**32 = 63.37 % (the paper's figure); the remaining
    36.63 % of entries are the miss sentinel.

Note on the sentinel: u = 0xFFFFFFFF itself is a legitimate preimage
(sigma0(0xFFFFFFFF) - 0xFFFFFFFF = 0x20000000), so that single entry is
indistinguishable from a miss under this format.  Counting non-sentinel
entries in the file therefore gives 2,721,603,627, one below the true
coverage of 2,721,603,628 quoted in the paper; the missing entry is
v = 0x20000000.  The GPU build uses the int32 minimum as its sentinel
instead and does not have this collision; the effect on any statistic is
one entry in 2.7e9 and is ignored throughout.  A table built by this script
with --policy max is identical, entry for entry, to the table used for the
published runs (checked 2026-09-03).

Representative policy: with --policy max (default) the LARGEST u mapping to
each v is stored; with --policy min the smallest.  The paper's E3 experiment
compares the two and finds the filter rates unchanged within Poisson error.

Requirements: numpy, about 17 GiB of free RAM, ~3 minutes on one core.

Usage:
    python3 build_sigma0_table.py --out sigma0_u_table.npy
    python3 build_sigma0_table.py --out sigma0_u_table_MIN.npy --policy min
    python3 build_sigma0_table.py --check sigma0_u_table.npy   # verify 10^6 random entries
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

M = np.uint32(0xFFFFFFFF)
MISS = np.uint32(0xFFFFFFFF)
CHUNK = 1 << 24


def rotr(x, n):
    return (x >> np.uint32(n)) | (x << np.uint32(32 - n))


def s0(x):
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> np.uint32(3))


def build(policy: str) -> np.ndarray:
    tbl = np.full(1 << 32, MISS, dtype=np.uint32)
    starts = range(0, 1 << 32, CHUNK)
    if policy == "min":                       # iterate descending so the smallest u writes last
        starts = reversed(list(starts))
    t0 = time.time()
    for i, s in enumerate(starts):
        u = np.arange(s, s + CHUNK, dtype=np.uint64).astype(np.uint32)
        if policy == "min":
            u = u[::-1]
        v = (s0(u) - u) & M
        tbl[v] = u                              # last writer wins within the chunk order
        if i % 32 == 31:
            print(f"  {(i + 1) * CHUNK / 2**32 * 100:5.1f} %  {time.time() - t0:6.1f} s", flush=True)
    return tbl


def check(path: str, n: int = 1_000_000, seed: int = 1) -> bool:
    tbl = np.load(path, mmap_mode="r")
    print(f"{path}: dtype {tbl.dtype}, {tbl.shape[0]:,} entries")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 1 << 32, size=n, dtype=np.uint64)
    vals = np.asarray(tbl[idx]).view(np.uint32)
    hit = vals != MISS
    u = vals[hit].astype(np.uint32)
    ok = np.all(((s0(u) - u) & M) == idx[hit].astype(np.uint32))
    cov = hit.mean()
    print(f"checked {n:,} random entries: {hit.sum():,} hits ({cov * 100:.2f} % coverage, expect 63.37 %), "
          f"all consistent: {bool(ok)}")
    return bool(ok) and abs(cov - 0.6337) < 0.002


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output .npy path")
    ap.add_argument("--policy", choices=("max", "min"), default="max")
    ap.add_argument("--check", default=None, help="verify an existing table instead of building")
    ap.add_argument("--dry-run", action="store_true",
                    help="build in RAM and report coverage without writing the file")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check(a.check) else 1)
    if not a.out and not a.dry_run:
        ap.error("--out is required unless --check or --dry-run is given")
    print(f"Building sigma0(u)-u table, policy={a.policy} (2^32 entries, 16 GiB) ...", flush=True)
    tbl = build(a.policy)
    n = int((tbl != MISS).sum())
    print(f"coverage: {n:,} / {1 << 32:,} = {n / 2**32 * 100:.2f} %  "
          f"(paper: 2,721,603,628 = 63.37 %)")
    if a.dry_run:
        return
    np.save(a.out, tbl.view(np.int32))          # same bit patterns; sentinel becomes -1
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
