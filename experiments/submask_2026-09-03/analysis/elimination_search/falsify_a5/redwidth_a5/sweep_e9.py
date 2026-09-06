#!/usr/bin/env python3
"""All 2^w constants for a saturated e9 (and e8) at w=8: patterns with a9 solved
from e9 = t (kernel becomes S0(a5) - S0(k - a5) + mild), every tie family."""
import sys; sys.argv = ['sweep.py', '8', '300']
import numpy as np, json, time
from multiprocessing import Pool
import sweep as S
ties = S.tie_options()
jobs = []
for tname, tdefs in ties:
    for t9 in range(256):
        for e8 in (None, 0xff, 0x00):
            for e10 in (None, 0x00, 0xff):
                defs = {4: ('free',), 6: ('free',), 7: ('free',)}; defs.update(tdefs)
                defs[8] = ('free',) if e8 is None else ('sat_r', e8)
                defs[9] = ('sat_r', t9)
                defs[10] = ('free',) if e10 is None else ('sat_r', e10)
                defs[11] = ('free',)
                jobs.append((f"{tname} | e8={'free' if e8 is None else hex(e8)},e9={t9:#x},e10={'free' if e10 is None else hex(e10)},free", defs))
print(len(jobs), "configs", flush=True)
t0 = time.time(); out = []
with Pool(12) as p:
    for r in p.imap_unordered(S.one, jobs, chunksize=16):
        if r is not None: out.append(r)
out.sort(key=lambda r: r['mean'][1])
c1 = np.array([r['mean'][1] for r in out]); c0 = np.array([r['mean'][0] for r in out])
print(f"{len(out)} legal; a5->C1 min {c1.min():.3f} mean {c1.mean():.3f} max {c1.max():.3f}; C0 min {c0.min():.3f}; {time.time()-t0:.0f}s")
print("per e9 constant: min C1 over ties/e8/e10 (lowest 10):")
bye9 = {}
for r in out:
    t = int(r['defs']['9'][1]); bye9[t] = min(bye9.get(t, 9), r['mean'][1])
print(sorted(((round(v, 3), hex(k)) for k, v in bye9.items()))[:10])
print("lowest 10 configs:")
for r in out[:10]: print(f"  C1 {r['mean'][1]:.3f} C0 {r['mean'][0]:.2f} C2 {r['mean'][2]:.2f} C3 {r['mean'][3]:.2f}  {r['name']}")
json.dump(out, open('sweep_e9_w8.json', 'w'))
