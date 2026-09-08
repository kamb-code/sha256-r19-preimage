"""S5: diagnose the low output-difference weights seen at 19 rounds.
Hypothesis: not a differential weakness but incomplete avalanche -- a
difference in a LATE message word only reaches the last few rounds.
Test: per input-word, min and mean |dDigest| over all 32 single-bit
differences, at 19/20/21 rounds; plus the full-diffusion reference.
"""
import numpy as np
from core import compress, popcnt32
from s2_witness import WITNESSES, wparse, hparse

for label, (hexh, wtxt, Rw) in WITNESSES[:1] + WITNESSES[2:3]:
    W = wparse(wtxt)
    print(f"\n### {label}")
    for R in (19, 20, 21):
        base = compress(W[None, :], R)[0]
        print(f"  R={R}:", end=" ")
        line = []
        for w in range(16):
            D = np.zeros((32, 16), dtype=np.uint32)
            D[:, w] = (np.uint32(1) << np.arange(32, dtype=np.uint32))
            dig = compress((W[None, :] ^ D).astype(np.uint32), R)
            wt = popcnt32((dig ^ base[None, :]).reshape(-1)).reshape(-1, 8).sum(axis=1)
            line.append(f"W{w}:{wt.min():3d}/{wt.mean():5.1f}")
        print(" ".join(line[:8]))
        print("        " + " ".join(line[8:]))
print("\nformat  Wk: min/mean of |dDigest| over the 32 single-bit differences in Wk")
print("full avalanche reference: mean 128.0, min over 32 draws ~ 116")

# which rounds does each message word reach?
print("\nrounds in which each of W0..W15 is (directly or via expansion) active:")
for R in (19, 20, 21):
    reach = {}
    for w in range(16):
        act = [w]
        dw = [1 if i == w else 0 for i in range(16)]
        for t in range(16, R):
            v = dw[t-2] | dw[t-7] | dw[t-15] | dw[t-16]
            dw.append(v)
            if v:
                act.append(t)
        reach[w] = act
    print(f"  R={R}: " + "  ".join(f"W{w}->{reach[w]}" for w in (13, 14, 15)))
