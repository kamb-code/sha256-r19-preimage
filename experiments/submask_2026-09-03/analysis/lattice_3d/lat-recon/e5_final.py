"""E5: (a) my w=32 model IS the real sigma0, checked against the 16 GB table;
        (b) BKZ does not help -- the target is not the shortest vector;
        (c) direct count of lattice points shorter than the target at w=8.
"""
import json, math, sys, time
import numpy as np
from wsha import mk
import e2_sigma0, embed
from lat import lll, bkz, _gso

TBL = "/nvme0n1-disk/Kamvid/sigma0_u_table.npy"


def check_table():
    f = mk(32); M = f['M']
    tbl = np.load(TBL, mmap_mode="r").view(np.uint32)
    rng = np.random.default_rng(0)
    u = rng.integers(0, 1 << 32, 200000, dtype=np.uint64).astype(np.uint32)
    c = (f['s0'](u) - u) & np.uint32(M)
    root = np.asarray(tbl[c])
    hit = root != np.uint32(M)
    ok = int((((f['s0'](root[hit]) - root[hit]) & np.uint32(M)) == c[hit]).all())
    return dict(check="w=32 model vs 16GB table", probed=200000,
                stored_fraction=float(hit.mean()), all_roots_valid=bool(ok))


def count_shorter(w=8, trials=5, cap=400000, seed=7):
    """Enumerate lattice points of L with norm <= ||target||, cap the count."""
    rng = np.random.default_rng(seed)
    f = mk(w); M = f['M']
    out = []
    for _ in range(trials):
        u = int(rng.integers(0, 1 << w)); c = (f['s0'](u) - u) & M
        A, b, idx, N, fixed = e2_sigma0.build_system_B(w, c, 0, u)
        rows, Nn, Mm, K = embed.build(A, b)
        red = lll(rows)
        mu, nrm = _gso(red)
        d = len(red)
        R2 = (Nn + 1) * 1.0000001      # squared norm of the target
        cnt = [0]

        def rec(k, part, coefs):
            if cnt[0] >= cap:
                return
            if part > R2:
                return
            if k < 0:
                if any(coefs):
                    cnt[0] += 1
                return
            center = -sum(mu[i, k] * coefs[i] for i in range(k + 1, d) if coefs[i])
            rem = R2 - part
            if rem < 0 or nrm[k] <= 0:
                return
            rad = (rem / nrm[k]) ** 0.5
            lo = int(math.ceil(center - rad - 1e-12)); hi = int(math.floor(center + rad + 1e-12))
            if hi - lo > 5000:
                return
            for v in range(lo, hi + 1):
                coefs[k] = v
                rec(k - 1, part + nrm[k] * (v - center) ** 2, coefs)
                if cnt[0] >= cap:
                    break
            coefs[k] = 0

        rec(d - 1, 0.0, [0] * d)
        out.append(cnt[0])
    return dict(check=f"lattice points with norm <= target, w={w}",
                target_norm2=Nn + 1, counts=out, capped_at=cap)


if __name__ == "__main__":
    res = []
    r = check_table(); print(json.dumps(r), flush=True); res.append(r)
    for w in (8, 12):
        for use in (False, True):
            t = time.time()
            a = e2_sigma0.run(w, trials=15, enc='B', use_bkz=use)
            a['reduction'] = 'BKZ-20' if use else 'LLL'
            a['secs'] = round(time.time() - t, 1)
            print(json.dumps(a), flush=True); res.append(a)
    r = count_shorter(8, trials=5); print(json.dumps(r), flush=True); res.append(r)
    json.dump(res, open("e5_final.json", "w"), indent=1)
