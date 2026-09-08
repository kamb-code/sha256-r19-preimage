# Exact MITM chunk-separation count for R-round SHA-256, fixed IV, known target.
# Base variables W0..W15. Expanded W_j (j>=16) = s1(W_{j-2}) + W_{j-7} + s0(W_{j-15}) + W_{j-16}.
import itertools, sys

def base_deps(R):
    dep = {i: {i} for i in range(16)}
    for j in range(16, R):
        dep[j] = set()
        for k in (j-2, j-7, j-15, j-16):
            dep[j] |= dep[k]
    return dep

def analyse(R, n=256, verbose=True):
    dep = base_deps(R)
    best = None
    rows = []
    for c in range(1, R):
        F = set(); B = set()
        for j in range(0, c):  F |= dep[j]
        for j in range(c, R):  B |= dep[j]
        fo = F - B          # forward-only base words  -> forward neutral
        bo = B - F          # backward-only base words -> backward neutral
        df, db = 32*len(fo), 32*len(bo)
        d = min(df, db, n)
        cost = n - d
        rows.append((c, sorted(fo), sorted(bo), df, db, cost))
        if best is None or cost < best[0]:
            best = (cost, c, sorted(fo), sorted(bo), df, db)
    if verbose:
        print(f"--- R={R} rounds, n={n} ---")
        print("cut  |Fonly| |Bonly|   df   db   log2(MITM cost)=n-min(df,db)")
        for c, fo, bo, df, db, cost in rows:
            print(f"{c:3d}  {len(fo):5d}  {len(bo):6d}  {df:4d} {db:4d}   {cost:6.1f}   F={fo} B={bo}")
        print("BEST:", best)
    return best

for R in (19, 20, 21, 22, 24, 28, 32, 44):
    b = analyse(R, verbose=(R in (20,21,24)))
    print(f"R={R}: best MITM log2 cost = {b[0]}  (cut {b[1]}, df={b[4]}, db={b[5]})")
    print()
