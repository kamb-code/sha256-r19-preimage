import json, math
from scipy import stats
S="/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad"
c=[json.load(open(f"{S}/{f}"))["c3_lowk_zero"] for f in ("r20_status.json","r20_a100a_status.json","r20_a100b_status.json")]
P=[sum(x) for x in zip(*c)]
n0=P[0]
print("pooled n0",n0)
for k in range(16,29):
    exp=n0*2.0**-k; obs=P[k]
    print(f"k={k:2d} obs={obs:8d} exp={exp:10.1f} ratio={obs/exp:.3f} P(X<=obs)={stats.poisson.cdf(obs,exp):.3f}")
# conditional halvings from 20
print("steps 20..27 aggregate", sum(P[k+1] for k in range(20,28)), "of", sum(P[k] for k in range(20,28)))
# GLRT over 16..28
dev=0
for k in range(16,28):
    n,m=P[k],P[k+1]; th=m/n
    dev+=2*(m*math.log(th/0.5)+(n-m)*math.log((1-th)/0.5))
print("GLRT 16..28 dev",dev,"p",stats.chi2.sf(dev,12))
