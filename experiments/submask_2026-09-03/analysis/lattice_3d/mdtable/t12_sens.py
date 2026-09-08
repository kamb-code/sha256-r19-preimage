"""Is the context -> signature map triangular?  (If a11 moved only KC2, a10 only
KC1, a7 only KC0, the fibre could be walked by three independent 32-bit solves.)
Single-bit sensitivity at full width, 32 flips x 200 contexts per word."""
import sys, numpy as np
sys.path.insert(0, '/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/lattice/mdtable')
from wmodel import W
from vecsig import make_vecsig

w = W(32); M = w.M
rng = np.random.default_rng(3)
msg = [int(x) for x in rng.integers(0, 1 << 32, 16, dtype=np.uint64)]
h = w.digest(msg, 20)
vs = make_vecsig(w, h, 20)
v = int(rng.integers(0, 1 << 32, dtype=np.uint64))
n = 200
base = [rng.integers(0, 1 << 32, n, dtype=np.uint64) for _ in range(4)]
S0 = vs(v, *base)
names = ['a6', 'a7', 'a10', 'a11']
coords = ['KC0', 'KC1', 'KC2', 'kappa3']
print("mean Hamming weight of the signature change per single flipped context bit")
print(f"{'word':>6} " + " ".join(f"{c:>9}" for c in coords) + "   (0 = neutral, 16 = full avalanche)")
for i in range(4):
    acc = np.zeros(4)
    for b in range(32):
        p = list(base); p[i] = base[i] ^ np.uint64(1 << b)
        S1 = vs(v, *p)
        for k in range(4):
            d = (S0[k] ^ S1[k]).astype(np.uint64)
            acc[k] += np.unpackbits(d.astype('>u8').view(np.uint8)).reshape(-1, 64)[:, 32:].sum(1).mean()
    acc /= 32
    print(f"{names[i]:>6} " + " ".join(f"{x:9.2f}" for x in acc))
print("\nno zero entry => no word is neutral for any coordinate => the map is not")
print("triangular; the 96-bit fibre condition cannot be split into three 32-bit solves.")
print("(Even if it could, three exhaustive 32-bit solves cost 2^32 per fibre member,")
print(" exactly the break-even -- a tie at 2^45.38, not a gain.)")
