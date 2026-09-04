import numpy as np, sys
D="/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/"
M=np.uint32(0xFFFFFFFF)
def rotr(x,n): return (x>>np.uint32(n))|(x<<np.uint32(32-n))
def s0(x): return rotr(x,7)^rotr(x,18)^(x>>np.uint32(3))
for f in ("corpus_r19.npz","corpus_r20.npz"):
    z=np.load(D+f); n=len(z['a0'])
    print("==",f,n,"rows")
    for w in ("W1","W2","W3"):
        x=z[w].astype(np.uint32)
        print("  ",w,"== 0xFFFFFFFF:",int((x==M).sum()),
              " == 0x80000000:",int((x==np.uint32(0x80000000))).__str__() if False else int((x==np.uint32(0x80000000)).sum()),
              " v==0x20000000:",int((((s0(x)-x)&M)==np.uint32(0x20000000)).sum()),
              " v==0x91002000:",int((((s0(x)-x)&M)==np.uint32(0x91002000)).sum()),
              " max W:",hex(int(x.max())))
    # how many distinct lookup values used at all, and expected hits on one specific index
    tot=3*n
    print("   total lookups represented:",tot," expected rows hitting one fixed index if uniform:",tot/2**32)
