"""Does ANY meet-in-the-middle split exist?

(A) context side: is the signature map (a6,a7,a10,a11) -> (KC0,KC1,KC2,kappa3)
    additively (or XOR-) separable across any 2-way split of its four inputs?
    A split S(x,y) = f(x) + g(y) is detected by the mixed second difference
    S(x,y) - S(x,y') - S(x',y) + S(x',y') == 0.

(B) unknown side: are the constraint residuals c0..c3 separable across any
    split of (a0,a1,a2,a3)?  Also: how many LOW bits are separable (the
    carry-free part), since sigma0 is GF(2)-linear and only the carries of the
    modular sums break separability.
"""
import sys, itertools, numpy as np
sys.path.insert(0, '/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/lattice/mdtable')
from wmodel import W
from vecsig import make_vecsig

U = np.uint64
N = 1 << 16


def bits(x, w):
    return np.unpackbits(np.ascontiguousarray(x.astype('>u8')).view(np.uint8)
                         ).reshape(-1, 64)[:, 64 - w:].sum(1)


def contextside(WD, seed=1):
    w = W(WD); M = w.M
    rng = np.random.default_rng(seed)
    msg = [int(x) for x in rng.integers(0, M + 1, 16, dtype=U)]
    h = w.digest(msg, 20)
    vs = make_vecsig(w, h, 20)
    v = int(rng.integers(0, M + 1, dtype=U))
    names = ['a6', 'a7', 'a10', 'a11']
    print(f"\n(A) context map, w={WD}: mixed second difference over 2-way splits "
          f"({N:,} random quadruple pairs each)")
    print(f"    {'split':22s} {'zero (add)':>12s} {'zero (xor)':>12s}  mean bits moved")
    for r in (1, 2):
        for S in itertools.combinations(range(4), r):
            if r == 2 and 0 not in S:
                continue
            X = [rng.integers(0, M + 1, N, dtype=U) for _ in range(4)]
            Y = [rng.integers(0, M + 1, N, dtype=U) for _ in range(4)]
            def mix(sel):   # take coords in S from X if sel else from Y
                return [X[i] if ((i in S) == sel) else Y[i] for i in range(4)]
            # S(x,y): x = coords in S, y = the rest
            A = vs(v, *[X[i] if i in S else X[i] for i in range(4)])           # (x , y )
            B = vs(v, *[X[i] if i in S else Y[i] for i in range(4)])           # (x , y')
            C = vs(v, *[Y[i] if i in S else X[i] for i in range(4)])           # (x', y )
            D = vs(v, *[Y[i] if i in S else Y[i] for i in range(4)])           # (x', y')
            zadd = zxor = 0; mb = 0.0
            for k in range(4):
                d = (A[k] - B[k] - C[k] + D[k]) & M
                dx = (A[k] ^ B[k] ^ C[k] ^ D[k]) & M
                zadd += int((d == 0).sum()); zxor += int((dx == 0).sum())
                mb += bits(d, WD).mean()
            lab = '{' + ','.join(names[i] for i in S) + '} | rest'
            print(f"    {lab:22s} {zadd/(4*N):12.2e} {zxor/(4*N):12.2e}  {mb/4:.2f}/{WD}")


def unknownside(WD, seed=2):
    """c3 as a function of the four unknowns, in a real context."""
    w = W(WD); M = w.M
    rng = np.random.default_rng(seed)
    msg = [int(x) for x in rng.integers(0, M + 1, 16, dtype=U)]
    h = w.digest(msg, 20)
    v, a6, a7, a10, a11 = (int(x) for x in rng.integers(0, M + 1, 5, dtype=U))
    sig = w.signature(h, w.context(v, a6, a7, a10, a11), 20)
    KC0, KC1, KC2, vv, K3p = sig
    am1, am2, am3, am4 = w.IV[0], w.IV[1], w.IV[2], w.IV[3]
    em1, em2, em3, em4 = w.IV[4], w.IV[5], w.IV[6], w.IV[7]
    T2iv = w.T2(am1, am2, am3)
    C0c = (-T2iv - em4 - w.S1(em1) - w.Ch(em1, em2, em3) - w.K[0]) & M
    Ce0 = (am4 - T2iv) & M

    def resid(a0, a1, a2, a3):
        E0 = (a0 + Ce0) & M
        W0 = (a0 + C0c) & M
        G = (-(w.S0(a0) + w.Maj(a0, U(am1), U(am2))) - em3 - w.S1(E0)
             - w.Ch(E0, U(em1), U(em2)) - w.K[1]) & M
        W1 = (a1 + G) & M
        E1 = (am3 + a1 - (w.S0(a0) + w.Maj(a0, U(am1), U(am2)))) & M
        F12 = (-(w.S0(a1) + w.Maj(a1, a0, U(am1))) - em2 - w.S1(E1)
               - w.Ch(E1, E0, U(em1)) - w.K[2]) & M
        W2 = (a2 + F12) & M
        e2 = (am2 + a2 - (w.S0(a1) + w.Maj(a1, a0, U(am1)))) & M
        F23 = (-(w.S0(a2) + w.Maj(a2, a1, a0)) - em1 - w.S1(e2)
               - w.Ch(e2, E1, E0) - w.K[3]) & M
        W3 = (a3 + F23) & M
        e3 = (am1 + a3 - (w.S0(a2) + w.Maj(a2, a1, a0))) & M
        W4 = (vv - (w.S0(a3) + w.Maj(a3, a2, a1)) - E0 - w.S1(e3)
              - w.Ch(e3, e2, E1) - w.K[4]) & M
        c0 = (w.s0(W1) + W0 - a1 - KC0) & M
        c1 = (w.s0(W2) + W1 - a2 - KC1) & M
        c2 = (w.s0(W3) + W2 - a3 - KC2) & M
        c3 = (w.s0(W4) + W3 - K3p) & M
        return c0, c1, c2, c3

    names = ['a0', 'a1', 'a2', 'a3']
    print(f"\n(B) constraint residuals, w={WD}: which unknowns each c_j depends on, "
          f"and separability of c3 over 2-way splits ({N:,} pairs)")
    base = [rng.integers(0, M + 1, N, dtype=U) for _ in range(4)]
    r0 = resid(*base)
    for i in range(4):
        pert = list(base); pert[i] = rng.integers(0, M + 1, N, dtype=U)
        r1 = resid(*pert)
        moved = [float((r0[k] != r1[k]).mean()) for k in range(4)]
        print(f"    perturb {names[i]}: P(c_j changes) = " +
              "  ".join(f"c{k}={moved[k]:.3f}" for k in range(4)))
    for r in (1, 2):
        for S in itertools.combinations(range(4), r):
            if r == 2 and 0 not in S:
                continue
            X = [rng.integers(0, M + 1, N, dtype=U) for _ in range(4)]
            Y = [rng.integers(0, M + 1, N, dtype=U) for _ in range(4)]
            A = resid(*[X[i] for i in range(4)])
            B = resid(*[X[i] if i in S else Y[i] for i in range(4)])
            C = resid(*[Y[i] if i in S else X[i] for i in range(4)])
            D = resid(*[Y[i] for i in range(4)])
            d = (A[3] - B[3] - C[3] + D[3]) & M
            # how many low bits are separable
            low = 0
            for b in range(1, WD + 1):
                if int((d & ((1 << b) - 1) != 0).sum()) == 0:
                    low = b
                else:
                    break
            lab = '{' + ','.join(names[i] for i in S) + '} | rest'
            print(f"    c3 split {lab:20s} P(2nd difference = 0) = {float((d==0).mean()):.2e}"
                  f"   separable low bits = {low}")


if __name__ == '__main__':
    contextside(8); contextside(32)
    unknownside(8); unknownside(32)
