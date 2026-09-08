"""S3: per-round XOR differential cost model + best-characteristic search for
the SHA-256 COMPRESSION FUNCTION with FIXED IV and FREE MESSAGE, 19/20/21 rounds.

Preimage setting: the chaining input difference is 0 (fixed IV) and the output
difference must be 0 (same digest), so any usable characteristic is a
fixed-IV collision characteristic.

Cost model (exact, from S1):
  Sigma0/Sigma1/sigma0/sigma1 : 0 bits          (GF(2)-linear)
  modular add u+v -> u^v      : hw((du|dv) & 0x7fffffff)   [Lipmaa-Moriai]
  Ch  per bit                 : 0 if (de,df,dg) in {000,011} else 1
  Maj per bit                 : 0 if (da,db,dc) in {000,111} else 1
p_characteristic = 2^-cost.
"""
import time, sys
import numpy as np
from core import S0, S1, s0, s1
from s2_witness import lin_compress_map, kernel_basis, check_kernel, int_to_words

M32 = 0xFFFFFFFF
LOW31 = 0x7FFFFFFF


def _rotr(x, n): return ((x >> n) | (x << (32 - n))) & M32
def _S0(x): return _rotr(x, 2) ^ _rotr(x, 13) ^ _rotr(x, 22)
def _S1(x): return _rotr(x, 6) ^ _rotr(x, 11) ^ _rotr(x, 25)
def _s0(x): return _rotr(x, 7) ^ _rotr(x, 18) ^ (x >> 3)
def _s1(x): return _rotr(x, 17) ^ _rotr(x, 19) ^ (x >> 10)
def hw(x): return bin(x & M32).count("1")
def addcost(du, dv): return hw((du | dv) & LOW31)


def bitcost_ch(de, df, dg):
    """bits that must be fixed so that dCh = dg."""
    free = (~de & ~df & ~dg) | (~de & df & dg)      # patterns 000 and 011
    return hw(~free & M32)


def bitcost_maj(da, db, dc):
    """bits that must be fixed so that dMaj = da."""
    free = (~da & ~db & ~dc) | (da & db & dc)       # patterns 000 and 111
    return hw(~free & M32)


def trail_cost(dw16, rounds, want_zero_out=True):
    """Linearised XOR trail from a 16-word message difference; returns
    (cost_bits, final_state_difference, per_round_costs)."""
    dw = [int(x) & M32 for x in dw16]
    cost = 0
    for t in range(16, rounds):
        e1 = _s1(dw[t-2]); e2 = dw[t-7]; e3 = _s0(dw[t-15]); e4 = dw[t-16]
        cost += addcost(e1, e2)
        p = e1 ^ e2
        cost += addcost(p, e3)
        p ^= e3
        cost += addcost(p, e4)
        dw.append(p ^ e4)
    da = db = dc = dd = de = df = dg = dh = 0
    per = []
    for t in range(rounds):
        c0 = cost
        dch = dg                       # Ch ~ z
        dmj = da                       # Maj ~ x
        cost += bitcost_ch(de, df, dg)
        cost += bitcost_maj(da, db, dc)
        u = _S1(de)
        cost += addcost(dh, u); p = dh ^ u
        cost += addcost(p, dch); p ^= dch
        # + K : no difference, free
        cost += addcost(p, dw[t]); dt1 = p ^ dw[t]
        v = _S0(da)
        cost += addcost(v, dmj); dt2 = v ^ dmj
        cost += addcost(dt1, dt2)
        cost += addcost(dd, dt1)
        dh, dg, df, de = dg, df, de, (dd ^ dt1)
        dd, dc, db, da = dc, db, da, (dt1 ^ dt2)
        per.append(cost - c0)
    out = (da, db, dc, dd, de, df, dg, dh)
    return cost, out, per


def main():
    t0 = time.time()
    rng = np.random.default_rng(7)
    results = {}
    for R in (19, 20, 21):
        cols, nout = lin_compress_map(R, "z", "x")
        kern = kernel_basis(cols, nout)
        check_kernel(cols, nout, kern)
        print(f"\n=== R = {R} rounds ===")
        print(f"  zero-output message-difference space: dim {len(kern)} "
              f"(2^{len(kern)} candidate characteristics)")

        def cost_of(vec):
            c, out, _ = trail_cost(int_to_words(vec), R)
            assert all(o == 0 for o in out), "not a zero-output trail"
            return c

        # seeds: basis vectors, then randomised greedy descent
        best = None
        seeds = list(kern)
        for v in seeds:
            c = cost_of(v)
            if best is None or c < best[0]:
                best = (c, v)
        print(f"  best over {len(kern)} basis vectors alone: {best[0]} bits", flush=True)

        budget = 200.0 if R == 21 else 120.0
        tstart = time.time()
        cur_c, cur_v = best
        restarts = 0
        while time.time() - tstart < budget:
            improved = True
            while improved and time.time() - tstart < budget:
                improved = False
                order = rng.permutation(len(kern))
                for idx in order:
                    cand = cur_v ^ kern[int(idx)]
                    if cand == 0:
                        continue
                    c = cost_of(cand)
                    if c < cur_c:
                        cur_c, cur_v = c, cand
                        improved = True
                if cur_c < best[0]:
                    best = (cur_c, cur_v)
            restarts += 1
            # random restart
            v = 0
            for _ in range(int(rng.integers(1, 6))):
                v ^= kern[int(rng.integers(0, len(kern)))]
            if v == 0:
                v = kern[0]
            cur_c, cur_v = cost_of(v), v
        c, out, per = trail_cost(int_to_words(best[1]), R)
        dwords = int_to_words(best[1])
        print(f"  BEST characteristic found ({restarts} restarts, {budget:.0f}s):")
        print(f"     cost = {c} bits   ->  p_char = 2^-{c}")
        print(f"     active message words: "
              f"{[i for i in range(16) if dwords[i]]}  "
              f"hw(dW0..15) = {sum(hw(int(x)) for x in dwords)}")
        print(f"     per-round cost: {per}")
        results[R] = c
    print("\n" + "=" * 66)
    print("best zero-in / zero-out XOR characteristic, fixed IV, free message")
    for R in (19, 20, 21):
        print(f"   R = {R:2d} rounds :  p_char <= 2^-{results[R]}"
              f"   (bar to beat at R=21 is 2^-77)")
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
