#!/usr/bin/env python3
"""Statistics lens: halving test, Poisson claims, dedup bound. Read-only on data."""
import json, math
import numpy as np
from scipy import stats

SP = "/tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad"
EXP = "/home/administrator/sha/publish/experiments/submask_2026-09-03"

pods = {
    "h100_snapshot": json.load(open(f"{EXP}/gpu_run_snapshot/h100_status.json")),
    "a100a_final": json.load(open(f"{SP}/r20_a100a_status.json")),
    "a100b_final": json.load(open(f"{SP}/r20_a100b_status.json")),
}

print("=" * 78)
print("(a) NESTED HALVING TEST on c3_lowk_zero counters")
print("=" * 78)
counts = {}
for name, s in pods.items():
    c = np.array(s["c3_lowk_zero"], dtype=np.int64)
    assert c[0] == s["sub"], (name, c[0], s["sub"])
    assert np.all(np.diff(c) <= 0), "counts must be nested/non-increasing"
    counts[name] = c
    print(f"{name}: sub={s['sub']:,} a0={s['a0']:.3e} kmax={len(c)-1} "
          f"exp_preimages={s['expected_preimages']:.3f}")
pooled = sum(counts.values())
counts["POOLED"] = pooled


def step_table(c, k_lo=20, k_hi=28, label=""):
    print(f"\n-- {label}: conditional halving steps k -> k+1, n_(k+1) | n_k ~ Bin(n_k, 1/2)")
    print(f"{'k':>3} {'n_k':>8} {'n_k+1':>8} {'exp':>8} {'z':>6} {'2-sided p (exact)':>18}")
    zs, ps, dev = [], [], 0.0
    for k in range(k_lo, k_hi):
        n, m = int(c[k]), int(c[k + 1])
        if n == 0:
            continue
        z = (m - n / 2) / math.sqrt(n / 4)
        p = stats.binomtest(m, n, 0.5).pvalue
        th = m / n
        if 0 < th < 1:
            dev += 2 * (m * math.log(th / 0.5) + (n - m) * math.log((1 - th) / 0.5))
        zs.append(z); ps.append(p)
        print(f"{k:>3} {n:>8} {m:>8} {n/2:>8.1f} {z:>+6.2f} {p:>18.4f}")
    zs = np.array(zs); ps = np.array(ps)
    S_next = sum(int(c[k + 1]) for k in range(k_lo, k_hi))
    S_n = sum(int(c[k]) for k in range(k_lo, k_hi))
    zagg = (S_next - S_n / 2) / math.sqrt(S_n / 4)
    print(f"  aggregate (memory-file statistic) sum n_(k+1) = {S_next} of sum n_k = {S_n} "
          f"(exp {S_n/2:.1f}), z = {zagg:+.2f}, p = {2*stats.norm.sf(abs(zagg)):.3f}")
    print(f"  note: this aggregate weights step k by n_k -> dominated by the shallowest step "
          f"(k={k_lo}: {int(c[k_lo])} of {S_n}, {100*int(c[k_lo])/S_n:.0f}% of the weight)")
    m_ = len(zs)
    print(f"  max|z| = {abs(zs).max():.2f} at step k={k_lo + int(abs(zs).argmax())}; "
          f"min exact p = {ps.min():.4f}; Bonferroni x{m_}: {min(1, m_*ps.min()):.3f}; "
          f"Sidak: {1-(1-ps.min())**m_:.3f}")
    chi2 = float((zs ** 2).sum())
    print(f"  global tests over {m_} independent steps: sum z^2 = {chi2:.2f} "
          f"(chi2_{m_} p = {stats.chi2.sf(chi2, m_):.3f}); "
          f"GLRT deviance = {dev:.2f} (chi2_{m_} p = {stats.chi2.sf(dev, m_):.3f}); "
          f"Fisher = {-2*np.log(ps).sum():.2f} (chi2_{2*m_} p = {stats.chi2.sf(-2*np.log(ps).sum(), 2*m_):.3f})")
    # Depth-weighted (cost-relevant) statistic: product of theta_k over the steps
    # = n_khi / n_klo, compared with 2^-(khi-klo).  Binomial tail exact.
    n0, nK = int(c[k_lo]), int(c[k_hi])
    p0 = 2.0 ** -(k_hi - k_lo)
    bt = stats.binomtest(nK, n0, p0)
    ci = bt.proportion_ci(0.95, method="exact")
    print(f"  DEPTH-WEIGHTED: n_{k_hi}/n_{k_lo} = {nK}/{n0} = {nK/n0:.3e} vs 2^-{k_hi-k_lo} = {p0:.3e} "
          f"(expected {n0*p0:.1f}); ratio {nK/(n0*p0):.2f}; "
          f"one-sided P(X<={nK}) = {stats.binom.cdf(nK, n0, p0):.3f}; two-sided {bt.pvalue:.3f}")
    print(f"     95% exact CI on the ratio (prod theta)/(2^-{k_hi-k_lo}): "
          f"[{ci.low/p0:.2f}, {ci.high/p0:.2f}]  -> cost multiplier over bits {k_lo}..{k_hi-1} "
          f"in [{p0/ci.high:.2f}x, {p0/ci.low:.2f}x]")
    # From the top: n_khi / n_0 vs 2^-khi
    n_all = int(c[0])
    pa = 2.0 ** -k_hi
    bt2 = stats.binomtest(nK, n_all, pa)
    ci2 = bt2.proportion_ci(0.95, method="exact")
    print(f"  FROM TOP: n_{k_hi}/n_0 = {nK}/{n_all} vs 2^-{k_hi}: expected {n_all*pa:.1f}, "
          f"ratio {nK/(n_all*pa):.2f}, one-sided P(X<={nK}) = {stats.binom.cdf(nK, n_all, pa):.3f}; "
          f"95% CI ratio [{ci2.low/pa:.2f}, {ci2.high/pa:.2f}]")
    return zs


for name in ["h100_snapshot", "a100a_final", "a100b_final", "POOLED"]:
    step_table(counts[name], 20, 28, name)
print()
step_table(counts["POOLED"], 16, 28, "POOLED (steps 16..28)")
step_table(counts["POOLED"], 0, 28, "POOLED (all steps 0..28)")

# What the memory-file numbers were (H100 at 2.31e9 candidates, step 25->26 z=-2.58)
c = counts["h100_snapshot"]
z2526 = (c[26] - c[25] / 2) / math.sqrt(c[25] / 4)
print(f"\nH100 snapshot step 25->26: {c[25]} -> {c[26]}, z={z2526:+.2f}, exact two-sided p = "
      f"{stats.binomtest(int(c[26]), int(c[25]), 0.5).pvalue:.4f}")

# Sequential-analysis remark: a mixture-SPRT / always-valid p-value for the
# depth-weighted count vs. Poisson expectation, with a Gamma(1,1)-mixture prior on the rate ratio
def always_valid_p(x, lam):
    # mixture likelihood ratio  int Poisson(x;r*lam) dGamma(r;a=1,b=1) / Poisson(x;lam)
    # = Gamma(x+1)/(lam+1)^(x+1) * 1/(lam^x e^-lam / x!) ... closed form:
    # LR = x! * (1/(1+lam))^(x+1) * e^{lam} / lam^x  ... compute in logs
    lr = math.lgamma(x + 1) - (x + 1) * math.log(1 + lam) + lam - x * math.log(lam)
    return min(1.0, math.exp(-lr))
nK, n0 = int(pooled[28]), int(pooled[20])
lam = n0 * 2.0 ** -8
print(f"Always-valid (Gamma-mixture) p for pooled n_28 = {nK} vs lam = {lam:.1f}: {always_valid_p(nK, lam):.3f}")

print("\n" + "=" * 78)
print("(b) POISSON MODEL CHECKS")
print("=" * 78)
for name, s in pods.items():
    sub, a0, el = s["sub"], s["a0"], s["elapsed_s"]
    exp = sub * 2.0 ** -32
    print(f"{name}: sub/a0 = {sub/a0:.4e} (theory 0.934*1.0045e-4 = {0.934*1.0045e-4:.4e}); "
          f"expected = {exp:.4f}; rate = {exp/el*3600:.4f}/h; hours so far {el/3600:.2f}")
tot_exp = sum(s["sub"] for s in pods.values()) * 2.0 ** -32
print(f"old fleet total expected (h100 snapshot + final a100a/b) = {tot_exp:.3f}; "
      f"P(no hit) = exp(-{tot_exp:.3f}) = {math.exp(-tot_exp):.3f}")
# cost spent: h100 $3.49/h * 19.6h + 2 A100 $1.59/h * ~8.7h
h100_h = pods["h100_snapshot"]["elapsed_s"] / 3600
a_h = (pods["a100a_final"]["elapsed_s"] + pods["a100b_final"]["elapsed_s"]) / 3600
print(f"old-fleet spend estimate: H100 {h100_h:.1f}h*$3.49 + A100 {a_h:.1f}h*$1.59 = "
      f"${h100_h*3.49 + a_h*1.59:.0f} (memory says ~$95; snapshot H100 time is a lower bound)")

# New fleet
prod = {"t1": (0.006002072477713227, 325.3342881202698),
        "t2": (0.005439839093014598, 301.9164628982544)}
rates = {k: v[0] / v[1] * 3600 for k, v in prod.items()}
rates["dev"] = 7.10e8 * 3600 * 0.934 * 1.0045e-4 / 2 ** 32   # from r20_dev_run.log 7.10e8 a0/s
bench = 9.30e8 * 3600 * 0.934 * 1.0045e-4 / 2 ** 32
print(f"per-pod rates from status files: {rates}; bench-rate (9.30e8 a0/s) = {bench:.4f}/h")
lam_fleet = sum(rates.values())
print(f"fleet rate (measured production) = {lam_fleet:.3f}/h -> mean {1/lam_fleet:.2f} h, "
      f"expected cost ${4.77/lam_fleet:.1f}; stated 0.216/h -> mean 4.6 h, $22")
for q in (0.5, 0.9, 0.95):
    t = -math.log(1 - q) / lam_fleet
    print(f"   P(hit by {t:5.1f} h) = {q:.2f}  (cost ${4.77*t:.0f})")
print(f"P(hit within 60h cap, H100 alone at 0.0352/h) = {1-math.exp(-0.0352*60):.3f}")

print("\n" + "=" * 78)
print("(c) DEDUP BOUND on the lo-pass lift, data_campaign_screening.json")
print("=" * 78)
d = json.load(open("/home/administrator/sha/publish/data_campaign_screening.json"))
rows = d["rows"]
arms = {"screened": [r for r in rows if r["arm"] == "screened"],
        "control": [r for r in rows if r["arm"] == "control"]}
agg = {}
for a, rs in arms.items():
    lo = sum(r["lo"] for r in rs); fp = sum(r["fixed_points"] for r in rs)
    tr = sum(r["fp_trajectories"] for r in rs); rep = sum(r["fp_replay"] for r in rs)
    dup = tr - fp
    dup_cap = sum(min(r["fp_trajectories"] - r["fixed_points"], r["lo"]) for r in rs)
    n = len(rs)
    agg[a] = dict(lo=lo, fp=fp, tr=tr, dup=dup, dup_cap=dup_cap, n=n, rep=rep)
    print(f"{a:9s}: contexts {n}, trajectories {tr:,}, unique fp {fp:,}, duplicates {dup:,} "
          f"({100*dup/tr:.3f}%), lo {lo:,} (lo/traj {lo/tr:.2e}), replay {rep:,} (x{rep/tr:.2f}); "
          f"sum_rows min(dup,lo) = {dup_cap:,}")
s, cc = agg["screened"], agg["control"]
lift = (s["lo"] / s["n"]) / (cc["lo"] / cc["n"])
print(f"trajectory-counted lo lift = {lift:.3f}")
lo_s_min = s["lo"] - s["dup_cap"]; lo_c_max = cc["lo"]
lo_s_max = s["lo"]; lo_c_min = cc["lo"] - cc["dup_cap"]
print(f"hard bounds (each duplicate's lo status = its original's; per-row cap min(dup,lo)):")
print(f"   screened dedup lo in [{lo_s_min}, {lo_s_max}], control dedup lo in [{lo_c_min}, {lo_c_max}]")
print(f"   dedup lift in [{lo_s_min/lo_c_max:.3f}, {lo_s_max/max(lo_c_min,1):.3f}]")
print(f"   naive (no per-row cap) lower bound: {(s['lo']-s['dup'])/cc['lo']:.3f}")
# expected correction if duplicates pass at the ordinary rate
exp_s = s["dup"] * s["lo"] / s["tr"]; exp_c = cc["dup"] * cc["lo"] / cc["tr"]
print(f"if duplicates pass lo at the arm's ordinary rate: remove {exp_s:.1f} screened, {exp_c:.1f} control "
      f"-> lift {(s['lo']-exp_s)/(cc['lo']-exp_c):.3f}")
# which rows carry duplicates?
dr = sorted(rows, key=lambda r: -(r["fp_trajectories"] - r["fixed_points"]))[:8]
print("rows with most duplicates:")
for r in dr:
    print(f"   t{r['target']:02d} c{r['context']} {r['arm']:8s} dup={r['fp_trajectories']-r['fixed_points']:5d} "
          f"fp={r['fixed_points']:8,} lo={r['lo']:4d}")
ndup_rows = sum(1 for r in rows if r["fp_trajectories"] > r["fixed_points"])
print(f"rows with any duplicate: {ndup_rows} of {len(rows)}")

# target-blocked bootstrap of the lo lift (trajectory-counted) and per-target sign test
rng = np.random.default_rng(1)
T = sorted(set(r["target"] for r in rows))
S = np.array([sum(r["lo"] for r in arms["screened"] if r["target"] == t) for t in T], float)
C = np.array([sum(r["lo"] for r in arms["control"] if r["target"] == t) for t in T], float)
nS = np.array([sum(1 for r in arms["screened"] if r["target"] == t) for t in T], float)
nC = np.array([sum(1 for r in arms["control"] if r["target"] == t) for t in T], float)
B = 20000
idx = rng.integers(0, len(T), size=(B, len(T)))
bl = (S[idx].sum(1) / nS[idx].sum(1)) / (C[idx].sum(1) / nC[idx].sum(1))
print(f"target-blocked bootstrap (B={B}) 95% CI on pooled lo lift: "
      f"[{np.percentile(bl,2.5):.2f}, {np.percentile(bl,97.5):.2f}] (paper: [2.13, 5.82])")
wins = int((S > C).sum()); losses = int((S < C).sum()); ties = int((S == C).sum())
print(f"per-target sign test: screened > control on {wins}, < on {losses}, ties {ties}; "
      f"one-sided p = {stats.binomtest(wins, wins+losses, 0.5, alternative='greater').pvalue:.2e}")
