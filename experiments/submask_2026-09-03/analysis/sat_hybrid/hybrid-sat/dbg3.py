import numpy as np, struct
from pysat.solvers import Cadical195
from cnf_sha256 import *
rng = np.random.default_rng(1)
R=2
Wt = [int(v) for v in rng.integers(0, 1<<32, 16, dtype=np.uint64)]
h = digest(Wt, R).hex()
cnf = CNF()
W = [cnf.word() for _ in range(16)]
a = {-1: cnf.const_word(IV[0]), -2: cnf.const_word(IV[1]), -3: cnf.const_word(IV[2]), -4: cnf.const_word(IV[3])}
e = {-1: cnf.const_word(IV[4]), -2: cnf.const_word(IV[5]), -3: cnf.const_word(IV[6]), -4: cnf.const_word(IV[7])}
for r in range(R):
    T1 = cnf.add_many([e[r - 4], cnf.S1(e[r - 1]), cnf.chw(e[r - 1], e[r - 2], e[r - 3]), cnf.const_word(K[r]), W[r]])
    T2 = cnf.addw(cnf.S0(a[r - 1]), cnf.majw(a[r - 1], a[r - 2], a[r - 3]))
    a[r] = cnf.addw(T1, T2)
    e[r] = cnf.addw(a[r - 4], T1)
for i in range(16): cnf.fix_word(W[i], Wt[i])
s = Cadical195(bootstrap_with=cnf.clauses); print(s.solve())
info=dict(W=W,a=a,e=e)
Wm, st = decode(s.get_model(), info)
at, et, _ = forward(Wt, R)
for r in range(R): print(r, hex(st[('a',r)]), hex(at[r]), hex(st[('e',r)]), hex(et[r]))
print(h, [hex(x) for x in struct.unpack('>8I', bytes.fromhex(h))])
