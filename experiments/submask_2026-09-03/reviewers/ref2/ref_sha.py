# Fully independent SHA-256, constants derived from prime roots. No imports from repo.
import hashlib, struct
def _primes(n):
    ps=[];x=2
    while len(ps)<n:
        if all(x%p for p in ps if p*p<=x): ps.append(x)
        x+=1
    return ps
def _iroot(n,k):
    lo,hi=0,1<<((n.bit_length()//k)+2)
    while lo<hi:
        mid=(lo+hi+1)//2
        if mid**k<=n: lo=mid
        else: hi=mid-1
    return lo
P=_primes(64)
IV=[_iroot(p<<64,2)&0xFFFFFFFF for p in P[:8]]
KK=[_iroot(p<<96,3)&0xFFFFFFFF for p in P[:64]]
M=0xFFFFFFFF
def rr(x,n): return ((x>>n)|(x<<(32-n)))&M
def BS0(x): return rr(x,2)^rr(x,13)^rr(x,22)
def BS1(x): return rr(x,6)^rr(x,11)^rr(x,25)
def ss0(x): return rr(x,7)^rr(x,18)^(x>>3)
def ss1(x): return rr(x,17)^rr(x,19)^(x>>10)
def CH(x,y,z): return ((x&y)^((~x&M)&z))&M
def MJ(x,y,z): return ((x&y)^(x&z)^(y&z))&M
def compress(cv, W16, R=64):
    W=list(W16)
    for t in range(16,R): W.append((ss1(W[t-2])+W[t-7]+ss0(W[t-15])+W[t-16])&M)
    a,b,c,d,e,f,g,h=cv
    for t in range(R):
        t1=(h+BS1(e)+CH(e,f,g)+KK[t]+W[t])&M
        t2=(BS0(a)+MJ(a,b,c))&M
        h,g,f,e,d,c,b,a=g,f,e,(d+t1)&M,c,b,a,(t1+t2)&M
    return [(x+y)&M for x,y in zip(cv,[a,b,c,d,e,f,g,h])]
def sha256(msg):
    L=len(msg)*8; m=msg+b"\x80"+b"\x00"*((55-len(msg))%64)+struct.pack(">Q",L)
    cv=list(IV)
    for i in range(0,len(m),64):
        cv=compress(cv,[struct.unpack(">I",m[i+4*j:i+4*j+4])[0] for j in range(16)])
    return b"".join(struct.pack(">I",x) for x in cv)
if __name__=="__main__":
    print("IV ok:", IV==[0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19])
    import os
    ok=all(sha256(os.urandom(i%200))==hashlib.sha256.__call__(b"") .digest() if False else True for i in range(1))
    bad=0
    for i in range(300):
        m=os.urandom(i)
        if sha256(m)!=hashlib.sha256(m).digest(): bad+=1
    print("hashlib mismatches over 300 msgs:",bad)
