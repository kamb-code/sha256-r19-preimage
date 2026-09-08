"""The one genuinely 2-dimensional table in the family: index (target, root #).
How much is left in it?  Root multiplicities of u -> sigma0(u) - u.
"""
import sys, numpy as np
sys.path.insert(0, '/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/lattice/mdtable')
from wmodel import W
from math import exp, factorial

print(f"{'w':>3} {'image frac':>11} {'mean roots':>11} {'E[cand] j=1':>12} {'j=3':>8} {'all':>8}")
for wd in (8, 10, 12, 14, 16, 18, 20):
    w = W(wd); M = w.M
    u = np.arange(M + 1, dtype=np.uint64)
    t = (w.s0(u) - u) & M
    cnt = np.bincount(t, minlength=M + 1)
    img = float((cnt > 0).mean())
    e1 = float(np.minimum(cnt, 1).mean())
    e3 = float(np.minimum(cnt, 3).mean())
    ea = float(cnt.mean())
    print(f"{wd:3d} {img:11.6f} {ea/img:11.4f} {e1:12.6f} {e3:8.4f} {ea:8.4f}")
P = [exp(-1) / factorial(k) for k in range(12)]
print(f"\nPoisson(1) prediction: image 1-1/e = {1-exp(-1):.6f}, "
      f"E[min(k,1)] = {1-exp(-1):.4f}, E[min(k,3)] = "
      f"{sum(min(k,3)*P[k] for k in range(12)):.4f}, E[k] = 1.0000")
print("32-bit measured image fraction (paper): 0.633673")
print("32-bit measured E[cand] with the published three tables (paper sec 6.2): 0.9337")
print(f"ceiling of a complete multi-root table: 1.0000 / 0.9337 = "
      f"{1/0.9337:.3f}x = {np.log2(1/0.9337):.3f} bits")
