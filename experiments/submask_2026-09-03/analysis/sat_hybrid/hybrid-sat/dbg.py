import numpy as np
from pysat.solvers import Cadical195
from cnf_sha256 import *
rng = np.random.default_rng(1)
# unit tests of gates
c = CNF()
x = c.word(); y = c.word()
z = c.addw(x, y)
X, Y = 0xdeadbeef, 0x12345678
c.fix_word(x, X); c.fix_word(y, Y)
s = Cadical195(bootstrap_with=c.clauses); print("add sat", s.solve())
ms=set(l for l in s.get_model() if l>0)
val=lambda w: sum(((1 if (b is True) else 0 if (b is False) else (1 if (b>0 and b in ms) or (b<0 and -b not in ms) else 0))<<i) for i,b in enumerate(w))
print(hex(val(z)), hex((X+Y)&M))
for name, fn, ref in [("S0", c.S0, lambda v: (((v>>2)|(v<<30))^((v>>13)|(v<<19))^((v>>22)|(v<<10)))&M)]:
    c = CNF(); x = c.word(); z = fn(x); c.fix_word(x, X)
    s = Cadical195(bootstrap_with=c.clauses); s.solve(); ms=set(l for l in s.get_model() if l>0)
    print(name, hex(val(z)), hex(ref(X)))
c = CNF(); x=c.word(); y=c.word(); w=c.word(); z=c.majw(x,y,w); c.fix_word(x,X); c.fix_word(y,Y); c.fix_word(w,0x0f0f0f0f)
s = Cadical195(bootstrap_with=c.clauses); s.solve(); ms=set(l for l in s.get_model() if l>0)
print("maj", hex(val(z)), hex((X&Y)^(X&0x0f0f0f0f)^(Y&0x0f0f0f0f)))
c = CNF(); x=c.word(); y=c.word(); w=c.word(); z=c.chw(x,y,w); c.fix_word(x,X); c.fix_word(y,Y); c.fix_word(w,0x0f0f0f0f)
s = Cadical195(bootstrap_with=c.clauses); s.solve(); ms=set(l for l in s.get_model() if l>0)
print("ch", hex(val(z)), hex((X&Y)^((~X)&0x0f0f0f0f&M)))
# xor3 with constant
c = CNF(); x=c.word(); z=c.xor3w(x, c.const_word(Y), c.const_word(0x0f0f0f0f)); c.fix_word(x,X)
s = Cadical195(bootstrap_with=c.clauses); s.solve(); ms=set(l for l in s.get_model() if l>0)
print("xor3c", hex(val(z)), hex(X^Y^0x0f0f0f0f))
# 1 round
for R in (1,2,3):
    Wt = [int(v) for v in rng.integers(0, 1<<32, 16, dtype=np.uint64)]
    cnf, info = build(R, digest(Wt, R).hex(), "none")
    for i in range(16): cnf.fix_word(info["W"][i], Wt[i])
    s = Cadical195(bootstrap_with=cnf.clauses); print("R",R, s.solve())
