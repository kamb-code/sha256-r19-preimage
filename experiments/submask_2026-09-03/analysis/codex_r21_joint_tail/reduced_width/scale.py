"""Scaling of frame 2's consistency condition. Measure the MARGINAL rates for
a4 and a5 (statistically efficient) and their joint rate, across widths where
sigma0 is bijective. Decision rule: if -log2(P) tracks 2w, the reformulation
has only moved the cost."""
import math, numpy as np
from model import Model
from tail_lift import instance

def rates(w, trials, seed=11):
    m = Model(w, R=21); rng = np.random.default_rng(seed)
    h4 = h5 = hb = 0
    for _ in range(trials):
        W, a, e, Wf = instance(m, rng)
        a2 = dict(a)
        for i in range(4): a2[i] = int(rng.integers(0, 1 << m.w))
        e2 = dict(e)
        for r in range(0, 21):
            e2[r] = (a2[r-4] + a2[r] - m.T2(a2[r-1], a2[r-2], a2[r-3])) & m.M
        Wr = lambda r: m.recoverW(a2, e2, r)
        K3p = (Wr(19) - m.s1(Wr(17)) - Wr(12)) & m.M
        K4p = (Wr(20) - m.s1(Wr(18)) - Wr(13)) & m.M
        W4 = int(m.s0_inv((K3p - Wr(3)) & m.M))
        W5 = int(m.s0_inv((K4p - W4) & m.M))
        a4t = (W4 + m.T2(a2[3],a2[2],a2[1]) + e2[0] + m.S1(e2[3]) + m.Ch(e2[3],e2[2],e2[1]) + m.K[4]) & m.M
        a5t = (W5 + m.T2(a4t,a2[3],a2[2]) + e2[1] + m.S1(e2[4]) + m.Ch(e2[4],e2[3],e2[2]) + m.K[5]) & m.M
        x, y = (a4t == a2[4]), (a5t == a2[5])
        h4 += x; h5 += y; hb += (x and y)
    return h4, h5, hb, trials

print(f"{'w':>3} {'trials':>9} {'-log2 P(a4)':>12} {'-log2 P(a5)':>12} {'-log2 P(both)':>14} {'2w':>4} {'indep?':>8}")
for w, n in ((5,300000),(6,300000),(8,300000),(9,300000),(10,300000),(12,200000)):
    h4,h5,hb,t = rates(w,n)
    l4 = -math.log2(h4/t) if h4 else float('inf')
    l5 = -math.log2(h5/t) if h5 else float('inf')
    lb = -math.log2(hb/t) if hb else float('inf')
    ind = (hb/t)/((h4/t)*(h5/t)) if h4 and h5 and hb else float('nan')
    print(f"{w:>3} {t:>9} {l4:>12.2f} {l5:>12.2f} {lb:>14.2f} {2*w:>4} {ind:>8.2f}")
