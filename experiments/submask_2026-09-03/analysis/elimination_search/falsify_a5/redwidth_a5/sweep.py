#!/usr/bin/env python3
"""Structural sweep: every saturation pattern of a8..a11 x ties among a4,a6,a7 x
constants, measuring the a5 -> C0..C3 edge weights in the w-bit model.

Slots a8..a11: 'free' | sat_r(t): e_i = t via a_i | sat_h(t): e_{i+4} = t via a_i
              | sat_s0(t): e_{i+1} = t via a_i (Sigma0(a_i)+Maj(a_i,.,.) inverted; w<=12 only)
Constants t in {0, -1, two random values (fixed per run)}.
Ties on (a4,a6,a7): none, a7=a6, a7=~a6, a6=a4, a6=~a4, a7=a4, a7=~a4, a6=a7=a4,
  a7=rotr(a6,n) all n, a6=rotr(a4,n) all n, a7=a6+c, a7=a6^c (random c), digest ties
  a7=a12, a7=~a12, a6=a12, a4=a12, a4=const 0/-1, a6=const 0/-1, a7=const 0/-1.
Output: JSON lines with the config and weights.
"""
import sys, json, itertools, time
import numpy as np
from multiprocessing import Pool
from wmodel import Model, R

W = int(sys.argv[1]) if len(sys.argv) > 1 else 8
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
OUT = sys.argv[3] if len(sys.argv) > 3 else f"sweep_w{W}.jsonl"
SEED = 20260906
m = Model(W)
MM = m.M
rngc = np.random.default_rng(SEED + 7)
RC = [int(rngc.integers(0, 1 << W)) for _ in range(2)]        # random constants
CONSTS = [0, MM] + RC


def invert_s0maj(y, b, c):
    """x with S0(x)+Maj(x,b,c) == y, per state (arrays).  Returns x or raises."""
    n = 1 << W
    x = np.arange(n, dtype=np.int64)[None, :]
    v = (m.S0(x) + m.Maj(x, b[:, None], c[:, None])) & MM
    hit = v == y[:, None]
    ok = hit.any(axis=1)
    if not ok.all():
        raise RuntimeError("no root")
    return np.argmax(hit, axis=1).astype(np.int64)


class ModelX(Model):
    """adds sat_s0 (e_{i+1} = t solved through a_i by inversion)."""
    def realise(self, defs, a):
        M = self.M
        pend = {i: d for i, d in defs.items() if i not in a}
        while pend:
            prog = False
            for i, d in list(pend.items()):
                k = d[0]
                if k == 'sat_s0':
                    r = i + 1
                    deps = [r - 4, r, i - 1, i - 2]
                    if not all(j in a for j in deps):
                        continue
                    y = (a[r - 4] + a[r] - d[1]) & M
                    N_ = np.broadcast(y, a[i - 1], a[i - 2]).shape
                    yb = np.broadcast_to(y, N_).astype(np.int64)
                    bb = np.broadcast_to(a[i - 1], N_).astype(np.int64)
                    cb = np.broadcast_to(a[i - 2], N_).astype(np.int64)
                    a[i] = invert_s0maj(yb, bb, cb)
                    del pend[i]; prog = True
                    continue
                sub = {i: d}
                try:
                    Model.realise(self, sub, a)
                except RuntimeError:
                    continue
                if i in a:
                    del pend[i]; prog = True
            if not prog:
                raise RuntimeError(f"cyclic: {pend}")
        return a


mx = ModelX(W)


def tie_options():
    opts = [('none', {})]
    opts.append(('a7=a6', {7: ('eq', 6)}))
    opts.append(('a7=~a6', {7: ('neq', 6)}))
    opts.append(('a6=a4', {6: ('eq', 4)}))
    opts.append(('a6=~a4', {6: ('neq', 4)}))
    opts.append(('a7=a4', {7: ('eq', 4)}))
    opts.append(('a7=~a4', {7: ('neq', 4)}))
    opts.append(('a6=a7=a4', {6: ('eq', 4), 7: ('eq', 4)}))
    opts.append(('a7=a6,a6=~a4', {7: ('eq', 6), 6: ('neq', 4)}))
    for n in range(1, W):
        opts.append((f'a7=rotr(a6,{n})', {7: ('rot', 6, n)}))
        opts.append((f'a6=rotr(a4,{n})', {6: ('rot', 4, n)}))
        opts.append((f'a7=rotr(a4,{n})', {7: ('rot', 4, n)}))
    opts.append((f'a7=a6+{RC[0]:#x}', {7: ('add', 6, RC[0])}))
    opts.append((f'a7=a6^{RC[1]:#x}', {7: ('rotx', 6, 0, RC[1])}))
    opts.append((f'a7=~a6,a6=a4', {7: ('neq', 6), 6: ('eq', 4)}))
    for dw in (12, 13):
        opts.append((f'a7=A{dw}', {7: ('eq', dw)}))
        opts.append((f'a7=~A{dw}', {7: ('neq', dw)}))
        opts.append((f'a6=A{dw}', {6: ('eq', dw)}))
        opts.append((f'a4=A{dw}', {4: ('eq', dw)}))
    for c in (0, MM):
        opts.append((f'a4={c:#x}', {4: ('const', c)}))
        opts.append((f'a6={c:#x}', {6: ('const', c)}))
        opts.append((f'a7={c:#x}', {7: ('const', c)}))
        opts.append((f'a7=a6={c:#x}', {6: ('const', c), 7: ('const', c)}))
        opts.append((f'a7=a6,a4={c:#x}', {7: ('eq', 6), 4: ('const', c)}))
    return opts


def slot_options(i):
    opts = [('free', ('free',))]
    for t in CONSTS:
        opts.append((f'e{i}={t:#x}', ('sat_r', t)))
    if i + 4 <= 15:
        for t in CONSTS:
            opts.append((f'e{i+4}={t:#x}|a{i}', ('sat_h', t)))
    if W <= 12 and i + 1 <= 15:
        for t in (0, MM):
            opts.append((f'e{i+1}={t:#x}|S0a{i}', ('sat_s0', t)))
    return opts


def configs():
    ties = tie_options()
    slots = [slot_options(i) for i in (8, 9, 10, 11)]
    for tname, tdefs in ties:
        for combo in itertools.product(*slots):
            defs = {4: ('free',), 6: ('free',), 7: ('free',)}
            defs.update(tdefs)
            for i, (sname, sdef) in zip((8, 9, 10, 11), combo):
                defs[i] = sdef
            es = [{'sat_r': i, 'sat_h': i + 4, 'sat_s0': i + 1}.get(defs[i][0]) for i in (8, 9, 10, 11)]
            es = [e for e in es if e is not None]
            if len(es) != len(set(es)):
                continue
            name = tname + ' | ' + ','.join(s for s, _ in combo)
            yield name, defs


def one(job):
    name, defs = job
    rng = np.random.default_rng(SEED + (hash(name) & 0xFFFF))
    try:
        r = mx.edge_weights(defs, 5, rng, N=N)
    except RuntimeError as ex:
        return None
    return dict(name=name, defs={str(k): list(v) for k, v in defs.items()},
                mean=[float(x) for x in r['mean']], zero=[float(x) for x in r['zero_frac']],
                min_state=[float(x) for x in r['per_state_mean'].min(axis=1)])


if __name__ == '__main__':
    t0 = time.time()
    jobs = list(configs())
    print(f"w={W}: {len(jobs)} configurations, N={N} states", flush=True)
    n_ok = 0
    best = None
    with open(OUT, 'w') as f, Pool(26) as p:
        for k, r in enumerate(p.imap_unordered(one, jobs, chunksize=32)):
            if r is None:
                continue
            n_ok += 1
            f.write(json.dumps(r) + '\n')
            if best is None or r['mean'][1] < best['mean'][1]:
                best = r
            if k % 20000 == 0:
                print(f"  {k}/{len(jobs)} legal {n_ok} best C1 so far {best['mean'][1]:.3f} [{best['name']}] {time.time()-t0:.0f}s", flush=True)
    print(f"done: {n_ok} legal of {len(jobs)}; best C1 = {best['mean'][1]:.3f} at {best['name']}; {time.time()-t0:.0f}s")
