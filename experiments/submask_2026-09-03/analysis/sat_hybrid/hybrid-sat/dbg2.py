from pysat.solvers import Cadical195
from cnf_sha256 import *
c = CNF(); a=c.var(); b=c.var(); d=c.var(); z=c.xor3(a,b,d)
print(c.clauses)
for va in (0,1):
  for vb in (0,1):
    for vd in (0,1):
      s=Cadical195(bootstrap_with=c.clauses); s.solve(assumptions=[a if va else -a, b if vb else -b, d if vd else -d]); m=s.get_model(); print(va,vb,vd, '->', int(z in m), 'expected', va^vb^vd)
X=0xdeadbeef
c = CNF(); x = c.word(); z = c.S0(x); print(len(c.clauses), c.nv, z[:4])
