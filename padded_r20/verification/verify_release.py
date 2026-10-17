#!/usr/bin/env python3
"""Replay published evidence only. No inverse search, solver, or GPU dependency.

Constants are derived by integer roots of primes; forward rounds use an
eight-word state. A second, unmodified reference verifier is cross-checked.
Python 3.9+; standard library only. All failures cause a nonzero exit.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct
import verify_r19 as reference

MASK=(1<<32)-1
ROOT=Path(__file__).resolve().parents[1]

def require(value, message):
    if not value:
        raise ValueError(message)

def integer_root(number, degree):
    lo,hi=0,1<<((number.bit_length()+degree-1)//degree)
    while hi-lo>1:
        mid=(lo+hi)//2
        if mid**degree<=number:lo=mid
        else:hi=mid
    return lo

def constants():
    primes=[]
    candidate=2
    while len(primes)<64:
        if all(candidate%p for p in primes if p*p<=candidate):
            primes.append(candidate)
        candidate+=1
    iv=[integer_root(p<<64,2)&MASK for p in primes[:8]]
    k=[integer_root(p<<96,3)&MASK for p in primes]
    return iv,k

IV,K=constants()

def rot(x,n):return ((x>>n)|(x<<(32-n)))&MASK
def S0(x):return rot(x,2)^rot(x,13)^rot(x,22)
def S1(x):return rot(x,6)^rot(x,11)^rot(x,25)
def s0(x):return rot(x,7)^rot(x,18)^(x>>3)
def s1(x):return rot(x,17)^rot(x,19)^(x>>10)
def ch(x,y,z):return z^(x&(y^z))
def maj(x,y,z):return (x&y)^(x&z)^(y&z)

def padding(message):
    size=len(message)*8
    require(size<1<<64,'Message length does not fit SHA-256 length field')
    return message+b'\x80'+bytes((55-len(message))%64)+size.to_bytes(8,'big')

def compress(block,rounds,initial=IV):
    require(len(block)==64 and 1<=rounds<=64,'Invalid block or round count')
    words=list(struct.unpack('>16I',block))
    for t in range(16,64):
        words.append((s1(words[t-2])+words[t-7]+s0(words[t-15])+words[t-16])&MASK)
    state=list(initial)
    trace=[]
    for t in range(rounds):
        first=(state[7]+S1(state[4])+ch(*state[4:7])+K[t]+words[t])&MASK
        second=(S0(state[0])+maj(*state[:3]))&MASK
        state=[(first+second)&MASK,state[0],state[1],state[2],
               (state[3]+first)&MASK,state[4],state[5],state[6]]
        trace.append(state[:])
    return [(x+y)&MASK for x,y in zip(initial,state)],trace,words

def hash_message(message,rounds):
    padded=padding(message)
    state=IV[:]
    for i in range(0,len(padded),64):
        state,_,_=compress(padded[i:i+64],rounds,state)
    return ''.join(f'{x:08x}' for x in state)

def load(name):return json.loads((ROOT/'evidence'/name).read_text())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Write deterministic JSON verification report')
    args=parser.parse_args()
    require(IV==reference.H0 and K==reference.K,'Derived constants disagree with reference')
    fixtures=[b'',b'abc',b'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq']
    fixtures += [bytes((i*37+11)%256 for i in range(n)) for n in [1,54,55,56,63,64,65,119,120,127,128,129,1024]]
    for message in fixtures:
        require(hash_message(message,64)==hashlib.sha256(message).hexdigest(),'64-round control failed')
    require(hash_message(b'',64)=='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855','Empty vector failed')
    require(hash_message(b'abc',64)=='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad','abc vector failed')
    w=load('witness.json');msg=bytes.fromhex(w['message_hex']);blk=padding(msg)
    require(len(msg)==55 and len(blk)==64,'Wrong message/padding length')
    words=[int(x,16) for x in w['padded_block_words']]
    require(list(struct.unpack('>16I',blk))==words,'Published block differs from standard padding')
    require(blk[55:]==b'\x80'+(440).to_bytes(8,'big'),'Padding suffix differs')
    digests={}
    for r in [19,20,21,64]:
        output,trace,schedule=compress(blk,r)
        require(output==reference.sha256_reduced_raw_block(words,r),'Reference verifier disagreement')
        digests[str(r)]=''.join(f'{x:08x}' for x in output)
    require(digests['20']==w['target_hex'],'20-round witness mismatch')
    require(digests['19']!=w['target_hex'] and digests['21']!=w['target_hex'],'Adjacent-round control failed')
    require(digests['64']==w['full_sha256_hex']==hashlib.sha256(msg).hexdigest(),'Full SHA-256 mismatch')
    _,trace,schedule=compress(blk,20)
    a={-i-1:IV[i] for i in range(4)}
    e={-i-1:IV[i+4] for i in range(4)}
    a.update({t:s[0] for t,s in enumerate(trace)})
    e.update({t:s[4] for t,s in enumerate(trace)})
    recovered=[]
    for t in range(20):
        t2=(S0(a[t-1])+maj(a[t-1],a[t-2],a[t-3]))&MASK
        require(e[t]==(a[t-4]+a[t]-t2)&MASK,'State identity failed')
        recovered.append((a[t]-t2-e[t-4]-S1(e[t-1])-ch(e[t-1],e[t-2],e[t-3])-K[t])&MASK)
    require(recovered==schedule[:20],'Recovered schedule disagrees with forward expansion')
    residuals=[(recovered[16+j]-s1(recovered[14+j])-recovered[9+j]-s0(recovered[1+j])-recovered[j])&MASK for j in range(4)]
    require(residuals==[0]*4,'Nonzero schedule residual')
    v=0x327853f5
    require(S0(v)==v+1,'Stationary identity failed')
    require(all(a[t]==v for t in [4,5,7,8,9]),'Stationary frame differs')
    require(e[8]==e[9]==e[14]==MASK,'Saturated states differ')
    require(a[13]==a[12] and a[14]==(S0(a[12])+2*a[12])&MASK,'Structured target relation differs')
    require(a[15]==(440+S0(a[14])+2*a[12]+v+K[15])&MASK,'Closed-form padding-word identity differs')
    require(a[10]==(a[12]^MASK),'Complement relation differs')
    tail=[f'{a[t]:08x}' for t in range(12,20)]
    require(tail==w['target_tail_a12_a19'],'Target tail differs')
    tail_e=[(a[t-4]+a[t]-S0(a[t-1])-maj(a[t-1],a[t-2],a[t-3]))&MASK for t in [19,18,17,16]]
    tail_target=''.join(f'{(h+s)&MASK:08x}' for h,s in zip(IV,[a[t] for t in [19,18,17,16]]+tail_e))
    require(tail_target==w['target_hex'],'Digest reconstructed from prescribed tail differs')
    prior=[]
    for item in load('comparison_vectors.json'):
        r=item['rounds']
        if 'message_hex' in item:
            m=bytes.fromhex(item['message_hex']);b=padding(m)
            require(len(b)==64,'Expected one-block comparison vector')
            require(hashlib.sha256(item['label'].encode()).hexdigest()==item['target_hex'],'Comparison label mismatch')
        else:b=bytes.fromhex(''.join(item['block_words']))
        wp=list(struct.unpack('>16I',b));checks={}
        for rr in [r-1,r,r+1]:
            out,_,_=compress(b,rr)
            require(out==reference.sha256_reduced_raw_block(wp,rr),'Comparison verifier disagreement')
            checks[str(rr)]=''.join(f'{n:08x}' for n in out)==item['target_hex']
        require(checks[str(r)] and not checks[str(r-1)] and not checks[str(r+1)],'Comparison vector failed')
        prior.append(dict(author=item['author'],rounds=r,padded='message_hex' in item,checks=checks))
    run=load('run_summary.json')
    with (ROOT/'evidence/context_counters.csv').open() as f:rows=list(csv.DictReader(f))
    require(len(rows)==run['context_sweeps'],'Context count differs')
    require([int(x['context_index']) for x in rows]==list(range(len(rows))),'Missing/reordered context rows')
    require(sum(int(x['swept_a0']) for x in rows)==run['swept_a0'],'Sweep counter differs')
    require(int(rows[-1]['cumulative_collapsed_candidates'])==run['reported_collapsed_candidates'],'Final candidate counter differs')
    require(int(rows[-1]['cumulative_verified'])==run['reconstructed_hits']==1,'Hit counter differs')
    distinct=len({(x['y_hex'],x['p_hex']) for x in rows})
    require(distinct==run['distinct_context_pairs'],'Distinct context count differs')
    require(int(rows[-1]['p_hex'],16)==a[6] and int(rows[-1]['y_hex'],16)==a[11],'Hit context does not match forward trace')
    require(abs(run['swept_a0']/run['elapsed_s']-run['a0_per_s'])<1e-6,'Throughput arithmetic differs')
    require(run['reported_collapsed_candidates']/2**32==run['model_expected_hits'],'Model arithmetic differs')
    result=dict(status='PASS',scope='Static forward verification and record consistency only; no candidate search.',hashlib_control_messages=len(fixtures)+1,two_forward_implementations_agree=True,message_length_bytes=len(msg),padded_length_bytes=len(blk),zero_fill_bytes_between_marker_and_length=0,digests=digests,schedule_residuals=residuals,target_tail_a12_a19=tail,state_trace_a=[f'{a[t]:08x}' for t in range(20)],state_trace_e=[f'{e[t]:08x}' for t in range(20)],context_sweeps=len(rows),distinct_context_pairs=distinct,repeated_context_sweeps=len(rows)-distinct,comparison_vector_results=prior,all_ones_padded_result='Not executed; no witness supplied or claimed.')
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text)
    print(text,end='')

if __name__=='__main__':main()
