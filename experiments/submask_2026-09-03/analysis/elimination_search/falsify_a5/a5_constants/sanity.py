import sys, struct, time, numpy as np
sys.path.insert(0, "/home/administrator/sha/publish/code")
from submask_family import M, K, IV, forward, digest, recover_W, S0, S1, s0, Ch, Maj
from edge_bulk import targets, measure, U, IVA, IVE
rng = np.random.default_rng(1)
# (a) genuine messages: T_j == s0(W_{j+1}), and W recomputed == true W
bad = 0
for t in range(50):
    msg = bytes(rng.integers(0, 256, 55, dtype=np.uint8).tolist())
    pad = msg + b"\x80" + b"\x00" * 0 + struct.pack(">Q", 55 * 8)
    Wt = [struct.unpack(">I", pad[4*i:4*i+4])[0] for i in range(16)]
    a, e, Wf = forward(Wt, 20)
    A = {r: np.array([a[r]], dtype=U) for r in range(-4, 20)}
    T, E, W = targets(A)
    for r in range(20):
        if int(W[r][0]) != Wf[r] or int(E[r][0]) != e[r]: bad += 1
    for j in range(4):
        if int(T[j][0]) != s0(Wf[j+1]): bad += 1
print("genuine-message check: T_j == s0(W_{j+1}) and W,e exact:", "OK" if bad == 0 else f"{bad} MISMATCHES")
# (b) closest-miss frame of FINDINGS.md (a7=a6, e8=-1): expect C0 0.97 C1 12.11 C2 1.37 ; a4->C0 ~12.2-12.5
for cond in (dict(a7='eq6', e8=M), dict(a7='eq6', e8=M, e11=0), dict(), dict(a6='eq4', a7='eq6', e8=M, e10=M, e11=0)):
    t0 = time.time(); r = measure(cond, 7); el = time.time() - t0
    print(cond, "legal", r['legal'], " a5->C0..C3:", [round(r[f'C{j}_mean'], 2) for j in range(4)],
          " moved:", [round(r[f'C{j}_moved'], 3) for j in range(4)], " a4->C0:", round(r['a4C0_mean'], 2),
          " a4->C0(resolved a8):", round(r.get('a4C0_resolved_mean', -1), 2), f" {el*1000:.0f} ms")
# illegal example: e8 fixed without a7 = a6 (references a5)
r = measure(dict(e8=M), 7); print("illegal probe e8=-1 without a7=a6: legal =", r['legal'])
r = measure(dict(a7='neq6', e8=M), 7); print("illegal probe e8=-1 with a7=~a6: legal =", r['legal'])
