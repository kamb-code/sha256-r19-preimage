"""From-scratch SHA-256, constants derived from primes. No imports from repo."""
import hashlib, struct, sys
MASK = (1<<32)-1
def _primes(n):
    ps=[];x=2
    while len(ps)<n:
        if all(x%p for p in ps if p*p<=x): ps.append(x)
        x+=1
    return ps
def _iroot(n,k):
    lo,hi=1,1<<((n.bit_length()//k)+2)
    while lo<hi:
        mid=(lo+hi+1)//2
        if mid**k<=n: lo=mid
        else: hi=mid-1
    return lo
_P=_primes(64)
IVr=[_iroot(p<<64,2)&MASK for p in _P[:8]]
Kr=[_iroot(p<<96,3)&MASK for p in _P]
def rotr(x,n): return ((x>>n)|(x<<(32-n)))&MASK
def Sig0(x): return rotr(x,2)^rotr(x,13)^rotr(x,22)
def Sig1(x): return rotr(x,6)^rotr(x,11)^rotr(x,25)
def sig0(x): return rotr(x,7)^rotr(x,18)^(x>>3)
def sig1(x): return rotr(x,17)^rotr(x,19)^(x>>10)
def Ch(x,y,z): return (x&y)^(~x&z)&MASK
def Maj(x,y,z): return (x&y)^(x&z)^(y&z)
def compress(H, W16, rounds=64):
    """classic a..h formulation, independent of the a_r/e_r recurrence."""
    W=list(W16)
    for t in range(16,rounds):
        W.append((sig1(W[t-2])+W[t-7]+sig0(W[t-15])+W[t-16])&MASK)
    a,b,c,d,e,f,g,h=H
    for t in range(rounds):
        T1=(h+Sig1(e)+Ch(e,f,g)+Kr[t]+W[t])&MASK
        T2=(Sig0(a)+Maj(a,b,c))&MASK
        h=g; g=f; f=e; e=(d+T1)&MASK
        d=c; c=b; b=a; a=(T1+T2)&MASK
    return [(x+y)&MASK for x,y in zip(H,[a,b,c,d,e,f,g,h])]
def digest_words(W16, rounds=64):
    return b"".join(struct.pack(">I",w) for w in compress(IVr,W16,rounds))
if __name__=="__main__":
    import os
    assert IVr==[0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19], IVr
    assert Kr[0]==0x428a2f98 and Kr[63]==0xc67178f2
    ok=0
    for _ in range(300):
        m=os.urandom(55)
        p=m+b"\x80"+b"\x00"*(56-1-55)+struct.pack(">Q",55*8)
        W=[struct.unpack(">I",p[4*i:4*i+4])[0] for i in range(16)]
        assert digest_words(W,64)==hashlib.sha256(m).digest(); ok+=1
    print("ref_sha: constants derived from primes; matches hashlib on",ok,"messages")
