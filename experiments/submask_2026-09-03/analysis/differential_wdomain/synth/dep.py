# transitive base-word dependency of expanded words; and the MITM cut scan
def deps(R):
    d=[{i} for i in range(16)]
    for t in range(16,R):
        s=d[t-2]|d[t-7]|d[t-15]|d[t-16]
        d.append(s)
    return d
for R in range(19,25):
    d=deps(max(R,17))
    used=set()
    for t in range(16,R): used|=d[t]
    untouched=sorted(set(range(16))-used)
    # cut scan: forward chunk rounds 0..c-1 uses W_0..W_{c-1}; backward uses W_c..W_{R-1}
    best=None
    for c in range(1,R):
        fwdwords=set(range(min(c,16)))
        bwd=set()
        for t in range(c,R):
            bwd|= d[t] if t>=16 else {t}
        df=32*len(fwdwords-bwd)
        # backward neutral: base words in backward chunk not used by forward chunk
        fwd=set()
        for t in range(0,c): fwd|= d[t] if t>=16 else {t}
        db=32*len((bwd|set())-fwd) if False else 32*len(set(range(16))-fwd-(set()))
        # simpler: backward-only words = base words not appearing in forward chunk
        dbw=set(range(16))-fwd
        db=32*len(dbw)
        m=min(df,db)
        if best is None or m>best[0]: best=(m,c,sorted(fwdwords-bwd),sorted(dbw))
    print(R, "untouched",untouched, "best d=",best[0],"cut",best[1],"fwdneutral",best[2],"cost 2^%d"%(256-best[0]))
