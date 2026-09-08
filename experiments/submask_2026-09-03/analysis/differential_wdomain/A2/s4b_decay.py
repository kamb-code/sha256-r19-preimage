"""S4b: validate the per-round XOR cost model and measure how fast a
GF(2)-linear trail decays in the real compression function.

For low-weight message differences d, compare
   measured survival fraction after t rounds   (over random messages)
against the model prediction  2^-(cumulative cost through round t).
"""
import numpy as np, time
from core import S0, S1, s0, s1, ch, maj, H0, K, expand
from s3_charsearch import trail_cost, hw
from s4_decay import lin_states, real_states

M = np.uint32(0xFFFFFFFF)


def cum_cost(dw16, rounds):
    c, out, per = trail_cost(dw16, rounds)
    cum = np.cumsum(per)
    return cum, per, out


def main():
    t0 = time.time()
    rng = np.random.default_rng(5)
    R = 21
    NM = 1 << 22          # 4.2 M random messages per difference
    print(f"messages per difference: {NM:,}")
    cases = []
    for w in range(16):
        cases.append(("W%d bit0" % w, [(1 << 0) if i == w else 0 for i in range(16)]))
    cases.append(("W15 bit31 (MSB)", [(1 << 31) if i == 15 else 0 for i in range(16)]))
    cases.append(("W15 bit0", [(1 << 0) if i == 15 else 0 for i in range(16)]))

    print(f"\n{'difference':<18} {'rnds survived (meas)':>21} {'model 2^-cum':>14}"
          f" {'cum cost@last surv':>19}")
    for name, dw in cases:
        cum, per, out = cum_cost(dw, R)
        Wr = rng.integers(0, 1 << 32, (NM, 16), dtype=np.uint64).astype(np.uint32)
        A = real_states(Wr, R)
        B = real_states((Wr ^ np.array(dw, dtype=np.uint32)[None, :]).astype(np.uint32), R)
        pred = lin_states(dw, R)
        alive = np.ones(NM, dtype=bool)
        last = -1
        frac = []
        for t in range(R):
            dif = (A[t] ^ B[t]).astype(np.uint32)
            p = np.array(pred[t], dtype=np.uint32)
            alive &= np.all(dif == p[None, :], axis=1)
            f = alive.mean()
            frac.append(f)
            if alive.any():
                last = t
        meas = -np.log2(frac[last]) if last >= 0 and frac[last] > 0 else np.inf
        print(f"{name:<18} {last+1:>21d} {cum[last] if last>=0 else 0:>14d}"
              f" {meas:>19.2f}")
        if name in ("W15 bit0", "W0 bit0"):
            print("      per-round measured -log2(survival):",
                  [f"{-np.log2(f):.1f}" if f > 0 else "inf" for f in frac[:12]])
            print("      per-round model cumulative cost   :", list(cum[:12]))
    print(f"\nelapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
