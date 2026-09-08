"""E6: could the system be written WORD-level instead of bit-level?

A word-level lattice needs the constraints to be Z-linear mod 2^w up to a SMALL
error, i.e. sigma0(x) = alpha*x + beta + e(x) with e(x) confined to a short arc.
Then c3 = 0 becomes a modular linear equation with a small unknown, exactly the
Coppersmith / truncated-LCG / BDD regime.

Measured here: over ALL alpha, the shortest arc mod 2^w containing e(x), and the
largest modular-linear correlation |E[exp(2 pi i (sigma0(x)-alpha x)/2^w)]|.
"""
import json, math
import numpy as np
from wsha import mk


def min_arc(w, alphas=None, frac=1.0):
    """min over alpha of the shortest arc mod 2^w containing frac of e(x)."""
    f = mk(w); M = (1 << w) - 1
    x = np.arange(1 << w, dtype=np.int64)
    s = f['s0'](x).astype(np.int64)
    su = (s - x) & M                        # the map the table inverts
    if alphas is None:
        alphas = range(1 << w)
    best = (1 << w) + 1, None
    best_su = (1 << w) + 1, None
    n = 1 << w
    keep = int(round(frac * n))
    for a in alphas:
        for tag, base in (('s0', s), ('s0mu', su)):
            e = np.sort((base - a * x) & M)
            if keep >= n:
                gaps = np.diff(np.concatenate([e, [e[0] + (1 << w)]]))
                arc = (1 << w) - int(gaps.max())
            else:
                d = e[keep - 1:] - e[:n - keep + 1]
                arc = int(d.min())
            if tag == 's0' and arc < best[0]:
                best = (arc, a)
            if tag == 's0mu' and arc < best_su[0]:
                best_su = (arc, a)
    return dict(w=w,
                min_arc_sigma0=best[0], best_alpha_sigma0=int(best[1]),
                bits_sigma0=round(math.log2(max(best[0], 1)), 2),
                min_arc_sigma0_minus_u=best_su[0], best_alpha_su=int(best_su[1]),
                bits_sigma0_minus_u=round(math.log2(max(best_su[0], 1)), 2),
                full_width_bits=w)


def max_modular_correlation(w):
    """max_alpha |(1/2^w) sum_x exp(2 pi i (sigma0(x) - alpha x)/2^w)|, via FFT."""
    f = mk(w); M = (1 << w) - 1
    x = np.arange(1 << w, dtype=np.int64)
    s = f['s0'](x).astype(np.int64)
    su = (s - x) & M
    out = {}
    for tag, base in (('sigma0', s), ('sigma0_minus_u', su)):
        h = np.exp(2j * np.pi * base / (1 << w))
        F = np.abs(np.fft.fft(h)) / (1 << w)
        F[0] = 0 if tag == 'x' else F[0]
        out[f'max_corr_{tag}'] = float(F.max())
        out[f'argmax_{tag}'] = int(F.argmax())
    out['noise_floor_2^-w/2'] = 2 ** (-w / 2)
    out['expected_max_of_2^w_gaussians'] = math.sqrt(math.log(1 << w)) * 2 ** (-w / 2)
    out['w'] = w
    return out


if __name__ == "__main__":
    res = []
    for w in (8, 12, 16):
        r = min_arc(w); print(json.dumps(r), flush=True); res.append(r)
        r = max_modular_correlation(w); print(json.dumps(r), flush=True); res.append(r)
    # w = 32: exhaustive alpha is 2^64 ops; sample alpha instead, on a random x subset
    f = mk(32); M = 0xFFFFFFFF
    rng = np.random.default_rng(0)
    x = rng.integers(0, 1 << 32, 1 << 20, dtype=np.uint64).astype(np.int64)
    s = f['s0'](x.astype(np.uint32)).astype(np.int64)
    su = (s - x) & M
    best = (1 << 32) + 1, None; best_su = (1 << 32) + 1, None
    cand = list(rng.integers(0, 1 << 32, 20000, dtype=np.uint64).astype(np.int64))
    cand += [(1 << 25) + (1 << 14), (1 << 25) + (1 << 14) - (1 << 3), 1, 0, M,
             (1 << 25) + (1 << 14) + (1 << 29)]
    for a in cand:
        for tag, base in (('s0', s), ('s0mu', su)):
            e = np.sort((base - a * x) & M)
            gaps = np.diff(np.concatenate([e, [e[0] + (1 << 32)]]))
            arc = (1 << 32) - int(gaps.max())
            if tag == 's0' and arc < best[0]:
                best = (arc, a)
            if tag == 's0mu' and arc < best_su[0]:
                best_su = (arc, a)
    r = dict(w=32, sampled_alphas=len(cand), x_sample=1 << 20,
             min_arc_sigma0=best[0], bits_sigma0=round(math.log2(best[0]), 2),
             min_arc_sigma0_minus_u=best_su[0],
             bits_sigma0_minus_u=round(math.log2(best_su[0]), 2),
             full_width_bits=32)
    print(json.dumps(r), flush=True); res.append(r)
    json.dump(res, open("e6_linear.json", "w"), indent=1)
