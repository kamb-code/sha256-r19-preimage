import json
import numpy as np
import math
from wsha import mk
from e6_linear import min_arc, max_modular_correlation

out = []
with open('e6_linear.log', 'w') as fh:
    def emit(r):
        fh.write(json.dumps(r) + "\n"); fh.flush(); print(json.dumps(r), flush=True); out.append(r)
    for w in (8, 12, 16):
        emit(min_arc(w))
        emit(max_modular_correlation(w))
    # w=32: sampled alphas, exhaustive-in-x is impossible; 2^17 x-sample
    f = mk(32); M = 0xFFFFFFFF
    rng = np.random.default_rng(0)
    x = rng.integers(0, 1 << 32, 1 << 17, dtype=np.uint64).astype(np.int64)
    s = f['s0'](x.astype(np.uint32)).astype(np.int64)
    su = (s - x) & M
    best = (1 << 32) + 1, None; best_su = (1 << 32) + 1, None
    cand = list(rng.integers(0, 1 << 32, 4000, dtype=np.uint64).astype(np.int64))
    cand += [(1 << 25) + (1 << 14), (1 << 25) + (1 << 14) - (1 << 3), 1, 0, M,
             (1 << 25) + (1 << 14) + (1 << 29), (1 << 25), (1 << 14), (1 << 29)]
    for a in cand:
        for tag, base in (('s0', s), ('s0mu', su)):
            e = np.sort((base - a * x) & M)
            gaps = np.diff(np.concatenate([e, [e[0] + (1 << 32)]]))
            arc = (1 << 32) - int(gaps.max())
            if tag == 's0' and arc < best[0]:
                best = (arc, a)
            if tag == 's0mu' and arc < best_su[0]:
                best_su = (arc, a)
    emit(dict(w=32, sampled_alphas=len(cand), x_sample=1 << 17,
              min_arc_sigma0=best[0], bits_sigma0=round(math.log2(best[0]), 2),
              min_arc_sigma0_minus_u=best_su[0],
              bits_sigma0_minus_u=round(math.log2(best_su[0]), 2),
              full_width_bits=32))
json.dump(out, open("e6_linear2.json", "w"), indent=1)
