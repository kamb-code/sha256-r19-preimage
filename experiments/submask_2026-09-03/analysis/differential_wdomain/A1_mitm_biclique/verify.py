import numpy as np, random
M=0xFFFFFFFF
def rotr(x,n): return ((x>>n)|(x<<(32-n)))&M
def s0(x): return rotr(x,7)^rotr(x,18)^(x>>3)
def s1(x): return rotr(x,17)^rotr(x,19)^(x>>10)
def expand(W,R):
    W=list(W)
    for j in range(16,R): W.append((s1(W[j-2])+W[j-7]+s0(W[j-15])+W[j-16])&M)
    return W
random.seed(1)
for R in (19,20,21,22,23,24):
    dep=[set() for _ in range(R)]
    for i in range(16): dep[i]={i}
    for j in range(16,R):
        dep[j]=set().union(*[dep[k] for k in (j-2,j-7,j-15,j-16)])
    expanded=set().union(*[dep[j] for j in range(16,R)]) if R>16 else set()
    free=sorted(set(range(16))-expanded)
    print(f"R={R}: W16..W{R-1} depend on {len(expanded)}/16 base words; NOT touched: {free} "
          f"-> max forward-neutral df = {32*len(free)} bits -> MITM floor 2^{256-32*len(free)}")
# numeric confirmation for R=21
R=21; bad=0
for t in range(3000):
    W=[random.getrandbits(32) for _ in range(16)]
    E=expand(W,R)[16:]
    W2=list(W)
    for i in (6,7,8): W2[i]=random.getrandbits(32)
    E2=expand(W2,R)[16:]
    if E!=E2: bad+=1
print("R=21 numeric: perturbing W6,W7,W8 changed W16..W20 in",bad,"of 3000 trials")
# and confirm every OTHER word does change them
for i in range(16):
    ch=0
    for t in range(200):
        W=[random.getrandbits(32) for _ in range(16)]; E=expand(W,R)[16:]
        W2=list(W); W2[i]^=1<<random.randrange(32)
        if expand(W2,R)[16:]!=E: ch+=1
    print(f"  W{i}: single-bit flip changes W16..W20 in {ch}/200 trials")
