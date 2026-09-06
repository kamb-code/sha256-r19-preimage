#!/usr/bin/env python3
"""Exact decomposition of the a5 -> C1 edge, and what each atom weighs.

Claim (from the recovered-W formula): with a0..a3 and the context fixed, the C1
lookup target T_1 = W17 - s1(W15) - W10 - W1 depends on a5 only through W10, as

    T_1 = const + f(a5),
    f(a5) = -S0(a5) - Maj(a5, a4, a3) + S1(a5 + c9) + Ch(a5 + c9, e8, e7),
    c9 = a9 - S0(a8) - Maj(a8, a7, a6),
    e8 = a4 + a8 - S0(a7) - Maj(a7, a6, a5),      e7 = a3 + a7 - S0(a6) - Maj(a6, a5, a4).

Checks:
  (1) exactness: T_1(a5') - T_1(a5) == f(a5') - f(a5) for every flip, every state;
  (2) zero effect of a10, a11 (hence of e10, e11, e12..e15 and the digest words):
      the 32-vector of dT_1 is bit-identical when a10, a11, a12..a19 are replaced;
  (3) per-atom weights in the best condition and in the base frame: the weight of
      the change of each atom alone, of the pair S1(a5+c9) - S0(a5) alone, and of f;
  (4) a broad scan of c9 on the reduced pair g(x) = S1(x + c) - S0(x): 200,000 random
      c and all c with <= 2 set bits or <= 2 clear bits, 64 states x 32 flips each,
      to bound the minimum of the dominant term over the whole 2^32 range of c9.
"""
import sys, json
import numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import M, S0, S1, Ch, Maj
from edge_bulk import realise, targets, U, u, BITS, NS

pc = np.bitwise_count


def f_atoms(a):
    c9 = a[9] - (S0(a[8]) + Maj(a[8], a[7], a[6]))
    e8 = a[4] + a[8] - (S0(a[7]) + Maj(a[7], a[6], a[5]))
    e7 = a[3] + a[7] - (S0(a[6]) + Maj(a[6], a[5], a[4]))
    y = a[5] + c9
    return dict(S0=-S0(a[5]), Maj=-Maj(a[5], a[4], a[3]), S1=S1(y), Ch=Ch(y, e8, e7))


def study(cond, seed, label):
    rng = np.random.default_rng(seed)
    a = realise(cond, rng, NS)
    T0, _, _ = targets(a)
    af = {r: a[r][:, None] for r in a}
    af[5] = a[5][:, None] ^ BITS[None, :]
    Tf, _, _ = targets(af)
    dT1 = Tf[1] ^ T0[1][:, None]
    A0 = f_atoms(a); Af = f_atoms(af)
    f0 = sum(A0.values()); ff = sum(Af.values())
    exact = bool(np.all((Tf[1] - T0[1][:, None]) == (ff - f0[:, None])))
    # (2) replace a10, a11 and the digest words by fresh random words
    b = dict(a)
    for r in (10, 11) + tuple(range(12, 20)):
        b[r] = rng.integers(0, 1 << 32, NS, dtype=np.uint64).astype(U)
    if cond.get('e10') is not None or cond.get('e11') is not None:
        pass  # a10/a11 no longer satisfy the condition; the point is that dT_1 does not care
    Tb0, _, _ = targets(b)
    bf = {r: b[r][:, None] for r in b}; bf[5] = af[5]
    Tbf, _, _ = targets(bf)
    # the MODULAR difference of T_1 under the flip must be bit-identical (the XOR pattern
    # is not, since it depends on the base value through carries)
    same = bool(np.all((Tbf[1] - Tb0[1][:, None]) == (Tf[1] - T0[1][:, None])))
    w = lambda x0, x1: float(pc(x1 ^ x0[:, None]).mean())
    out = dict(label=label, cond=cond, exact_decomposition=exact, invariant_under_a10_a11_digest=same,
               w_T1=w(T0[1], Tf[1]), w_f=w(f0, ff),
               w_S0=w(A0['S0'], Af['S0']), w_S1=w(A0['S1'], Af['S1']),
               w_Maj=w(A0['Maj'], Af['Maj']), w_Ch=w(A0['Ch'], Af['Ch']),
               w_S1_minus_S0=w(A0['S0'] + A0['S1'], Af['S0'] + Af['S1']),
               w_Maj_plus_Ch=w(A0['Maj'] + A0['Ch'], Af['Maj'] + Af['Ch']),
               xor_support_S1_minus_S0=float(pc((Af['S0'] + Af['S1']) - (A0['S0'] + A0['S1'])[:, None]).mean()))
    print(f"[{label}] exact={exact} invariant(a10,a11,digest)={same}  "
          f"wt: T1 {out['w_T1']:.2f} = f {out['w_f']:.2f}; S0 {out['w_S0']:.2f} S1 {out['w_S1']:.2f} "
          f"S1-S0 {out['w_S1_minus_S0']:.2f} (modular diff popcount {out['xor_support_S1_minus_S0']:.2f}); "
          f"Maj {out['w_Maj']:.2f} Ch {out['w_Ch']:.2f} Maj+Ch {out['w_Maj_plus_Ch']:.2f}")
    return out


def scan_c(ncand=200_000, ns=64, seed=5):
    """min over c of E[wt(S1(x'+c) - S0(x') - S1(x+c) + S0(x))] on the reduced pair."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 1 << 32, ns, dtype=np.uint64).astype(U)
    xf = x[:, None] ^ BITS[None, :]
    dS0 = S0(xf) - S0(x)[:, None]                     # (ns, 32)
    cands = [int(v) for v in rng.integers(0, 1 << 32, ncand, dtype=np.uint64)]
    low = [0, M] + [1 << i for i in range(32)] + [M ^ (1 << i) for i in range(32)]
    low += [(1 << i) | (1 << j) for i in range(32) for j in range(i)]
    low += [M ^ ((1 << i) | (1 << j)) for i in range(32) for j in range(i)]
    cands = low + cands
    best = (99, None); worst = (0, None); acc = []
    B = 256
    for i in range(0, len(cands), B):
        cc = np.array(cands[i:i + B], dtype=np.uint64).astype(U)      # (B,)
        y0 = x[None, :] + cc[:, None]                                  # (B, ns)
        yf = xf[None, :, :] + cc[:, None, None]                        # (B, ns, 32)
        g0 = S1(y0) - S0(x)[None, :]
        gf = S1(yf) - S0(xf)[None, :, :]
        wt = pc(gf ^ g0[:, :, None]).mean(axis=(1, 2))                # (B,)
        acc.extend(wt.tolist())
        k = int(wt.argmin())
        if wt[k] < best[0]: best = (float(wt[k]), int(cc[k]))
        k = int(wt.argmax())
        if wt[k] > worst[0]: worst = (float(wt[k]), int(cc[k]))
    acc = np.array(acc)
    print(f"[scan c9 on S1(x+c)-S0(x)] {len(cands)} values of c ({len(low)} structured + {ncand} random), "
          f"{ns} states x 32 flips each: min {best[0]:.3f} at c=0x{best[1]:08x}, mean {acc.mean():.3f}, "
          f"max {worst[0]:.3f} at c=0x{worst[1]:08x}; #c below 8 bits: {(acc < 8).sum()}")
    return dict(n=len(cands), min=best, mean=float(acc.mean()), max=worst, below8=int((acc < 8).sum()))


if __name__ == '__main__':
    res = []
    res.append(study(dict(a7='eq6', e8=M, e11=0), 11, 'closest-miss frame (a7=a6, e8=-1, e11=0)'))
    res.append(study(dict(), 12, 'random context'))
    res.append(study(dict(a6='eq4', a7='eq6', e8=M, c9=0, e10=0, e11=M), 13, 'grid minimum (a6=a4, a7=a6, e8=-1, c9=0, e10=0, e11=-1)'))
    res.append(study(dict(a7='eq6', e8=M, c9=0, e11=0), 14, 'c9 = 0'))
    res.append(study(dict(a7='eq6', e8=M, c9=0x80000000, e11=0), 15, 'c9 = 2^31'))
    res.append(study(dict(a7='eq6', e8=M, c9=M, e11=0), 16, 'c9 = -1'))
    sc = scan_c()
    with open('results/atoms.json', 'w') as fh:
        json.dump(dict(studies=res, scan_c9=sc), fh, indent=1)
