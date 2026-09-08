"""Validate the width-w model: sweep hits are genuine R-round preimages."""
import sys, numpy as np
sys.path.insert(0, '/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/lattice/mdtable')
from wmodel import W

def rebuild_msg(w, h, ctx, R, a0123):
    """Full message words from the recovered state; returns None if inconsistent."""
    M = w.M
    ab, eb = w.backward_chain(h, R)
    a = dict(ab); a.update(ctx)
    for i, x in zip(range(4), a0123):
        a[i] = int(x)
    a.update({-1: w.IV[0], -2: w.IV[1], -3: w.IV[2], -4: w.IV[3]})
    e = {-1: w.IV[4], -2: w.IV[5], -3: w.IV[6], -4: w.IV[7]}
    for r in range(R):
        e[r] = (a[r - 4] + a[r] - w.T2(a[r - 1], a[r - 2], a[r - 3])) & M
    return [w.recover_W(a, e, r) for r in range(16)]


def run(width, R=20, ntrials=200, seed=7):
    w = W(width)
    tbl = w.build_table()
    M = w.M
    img = int((tbl != M + 1).sum())
    rng = np.random.default_rng(seed)
    print(f"w={width}: sigma0(u)-u image fraction {img/(M+1):.6f}  (32-bit value 0.633673)")
    tot = dict(a0=0, surv=0, sol=0, eps0=0, c3=0, ver=0)
    for _ in range(ntrials):
        msg = [int(x) for x in rng.integers(0, M + 1, 16, dtype=np.uint64)]
        h = w.digest(msg, R)
        v, a6, a7, a10, a11 = (int(x) for x in rng.integers(0, M + 1, 5, dtype=np.uint64))
        ctx = w.context(v, a6, a7, a10, a11)
        sig = w.signature(h, ctx, R)
        out = w.sweep(tbl, sig, R)
        tot['a0'] += M + 1
        for k in ('surv', 'sol', 'eps0'):
            tot[k] += out[k]
        z = np.nonzero(out['c3'] == 0)[0]
        tot['c3'] += z.size
        for j in z:
            a0123 = [out['a'][i][j] for i in range(4)]
            Wm = rebuild_msg(w, h, ctx, R, a0123)
            if w.digest(Wm, R) == h:
                tot['ver'] += 1
    n = tot['a0']
    print(f"  swept a0        {n:,}")
    print(f"  C0 survivors    {tot['surv']:,}  ({tot['surv']/n:.4f}; predicted c=0.634)")
    print(f"  triangular sols {tot['sol']:,}  ({tot['sol']/max(tot['surv'],1):.4f} per survivor; c^2=0.401)")
    print(f"  collapse eps=0  {tot['eps0']:,}  ({tot['eps0']/max(tot['sol'],1):.3e}; (3/4)^w={0.75**width:.3e})")
    print(f"  c3 == 0         {tot['c3']:,}  ({tot['c3']/max(tot['eps0'],1):.3e}; 2^-w={2.0**-width:.3e})")
    print(f"  VERIFIED preimages {tot['ver']:,}   consistency {'OK' if tot['ver']==tot['c3'] else 'FAIL'}")
    return tot


if __name__ == '__main__':
    for width in (6, 8, 10, 12):
        run(width, ntrials={6: 400, 8: 400, 10: 200, 12: 60}[width])
        print()
