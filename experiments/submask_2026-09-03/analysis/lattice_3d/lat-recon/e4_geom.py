"""E4: lattice geometry of the blasted C0..C3 system, without reducing it.

covol(ker_Z(Abar)) = sqrt(det(Abar Abar^T)) / [Z^M : Abar Z^(N+1)]
Validated against the LLL-measured det(L0) on the small sigma0 instances,
where the index turns out to be 1.
"""
import json, math
import numpy as np
from wsha import mk
import e2_sigma0, embed, blast


def geom(A, b, name=""):
    A = np.array(A, dtype=np.float64)
    M, N = A.shape
    ones = A.sum(axis=1)
    bp = 2 * np.array(b, dtype=np.float64) - ones
    Ab = np.hstack([A, -bp.reshape(-1, 1)])          # M x (N+1)
    G = Ab @ Ab.T
    sign, logdet = np.linalg.slogdet(G)
    rank = np.linalg.matrix_rank(A)
    d0 = N + 1 - rank
    log2_det = 0.5 * logdet / math.log(2) if sign > 0 else float('nan')
    tgt = math.sqrt(N + 1)
    gh = math.sqrt(d0 / (2 * math.pi * math.e)) * 2 ** (log2_det / d0)
    return dict(name=name, N=int(N), M=int(M), rank=int(rank), dim_L0=int(d0),
                log2_det_L0=round(log2_det, 2),
                target_norm=round(tgt, 2), gh=round(gh, 3),
                ratio=round(tgt / gh, 3),
                log2_points_shorter=round(d0 * math.log2(tgt / gh), 1))


if __name__ == "__main__":
    out = []
    print("--- validation of the formula against LLL-measured det(L0) (sigma0, enc B) ---")
    for w in (8, 12, 16):
        rng = np.random.default_rng(11)
        f = mk(w); Mm = f['M']
        u = int(rng.integers(0, 1 << w)); c = (f['s0'](u) - u) & Mm
        A, b, idx, N, fixed = e2_sigma0.build_system_B(w, c, 0, u)
        g = geom(A, b, f"sigma0 w={w}")
        sols, red, Nn, Mm2 = embed.solve(A, b)
        d = embed.diagnostics(red, Nn, Mm2)
        g['lll_log2_det'] = round(math.log2(d['det_L0']), 2)
        g['lll_gh'] = round(d['gh_L0'], 3)
        print(json.dumps(g)); out.append(g)

    print("--- blasted SHA-256 constraint systems ---")
    for w in (8, 12, 16, 32):
        for subset in ('c3', 'c0', 'all'):
            A, b, truth, meta = blast.build_R20(w, seed=1, subset=subset)
            g = geom(A, b, f"R20 {subset} w={w}")
            print(json.dumps(g), flush=True); out.append(g)
    json.dump(out, open("e4_geom.json", "w"), indent=1)
