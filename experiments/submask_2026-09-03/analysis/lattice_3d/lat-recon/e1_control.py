"""E1: positive controls -- regimes where lattices DO work, through the same code.

(i) low-density subset sum, via the identical embed.solve() 0/1 machinery.
(ii) truncated LCG, the textbook "small unknowns mod q" lattice.
"""
import json, math, time, sys
import numpy as np
from lat import lll
import embed


def subset_sum(n, density, trials=20, seed=3):
    """weights of B bits with B = n/density; find the 0/1 vector."""
    import random
    rng = np.random.default_rng(seed)
    rr = random.Random(seed)
    B = int(round(n / density))
    ok = 0
    diag = []
    for _ in range(trials):
        wts = [rr.getrandbits(B) | 1 for _ in range(n)]
        z = [int(rng.integers(0, 2)) for _ in range(n)]
        s = sum(w * zz for w, zz in zip(wts, z))
        A = [wts]
        sols, red, N, M = embed.solve(A, [s])
        d = embed.diagnostics(red, N, M)
        diag.append(d)
        if any(sum(w * zz for w, zz in zip(wts, sol)) == s for sol in sols):
            ok += 1
    return dict(kind='subset_sum', n=n, density=density, bits=B, trials=trials,
                success=ok, rate=ok / trials,
                dim_L0=diag[0]['dim_L0'],
                target_norm=diag[0]['target_norm'],
                gh=float(np.nanmean([d.get('gh_L0', np.nan) for d in diag])),
                ratio=float(np.nanmean([d.get('ratio', np.nan) for d in diag])),
                log2_count=float(np.nanmean([d.get('log2_count', np.nan) for d in diag])))


def truncated_lcg(nout=8, k=8, q=1 << 32, trials=20, seed=5):
    """x_{i+1} = a x_i + b mod q, top q-k bits published, recover the k-bit tails."""
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(trials):
        a = int(rng.integers(1, q)) | 1
        b = int(rng.integers(0, q))
        x = [int(rng.integers(0, q))]
        for i in range(nout - 1):
            x.append((a * x[-1] + b) % q)
        hi = [xx >> k for xx in x]
        lo = [xx & ((1 << k) - 1) for xx in x]
        # y_i = lo_i ; relation:  hi_{i+1}*2^k + y_{i+1} = a(hi_i*2^k + y_i)+b mod q
        # unknowns y_0..y_{n-1} small (< 2^k)
        m = nout - 1
        Arows = []
        rhs = []
        for i in range(m):
            row = [0] * nout
            row[i] = a % q
            row[i + 1] = (-1) % q
            Arows.append(row)
            rhs.append((-(a * (hi[i] << k) + b) + (hi[i + 1] << k)) % q)
        # lattice:  [ I*scale | rows^T ] with q-rows, standard CVP embedding
        n = nout
        S = 1  # coordinate scale: unknowns bounded by 2^k
        KK = 1 << (k + 4)
        rows = []
        for j in range(n):
            r = [0] * (n + 1) + [KK * (Arows[i][j] % q) for i in range(m)]
            r[j] = 1
            rows.append(r)
        for i in range(m):
            r = [0] * (n + 1) + [0] * m
            r[n + 1 + i] = KK * q
            rows.append(r)
        r = [0] * (n + 1) + [-KK * rhs[i] for i in range(m)]
        r[n] = 1 << (k - 1)
        rows.append(r)
        red = lll(rows)
        found = False
        for v in red:
            for sg in (1, -1):
                vv = [sg * t for t in v]
                if any(t != 0 for t in vv[n + 1:]):
                    continue
                if vv[n] != (1 << (k - 1)):
                    continue
                cand = vv[:n]
                if all(0 <= c < (1 << k) for c in cand) and cand == lo:
                    found = True
        ok += found
    return dict(kind='truncated_lcg', nout=nout, k=k, trials=trials,
                success=ok, rate=ok / trials,
                density=(nout * k) / ((nout - 1) * 32))


if __name__ == "__main__":
    res = []
    for dens in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.2):
        r = subset_sum(40, dens, trials=20)
        print(json.dumps(r), flush=True); res.append(r)
    for k in (4, 8, 12, 16, 20, 24, 28):
        r = truncated_lcg(nout=8, k=k, trials=20)
        print(json.dumps(r), flush=True); res.append(r)
    json.dump(res, open("e1_control.json", "w"), indent=1)
