lines = []
# R=19: the key question.  Structured variants get 5 seeds x 2 solvers at the 30-min cap;
# the unstructured control (a certain timeout per Zaikin) gets 2 seeds x 2 solvers.
for seed in range(1, 6):
    for S in ("family", "context", "collapse"):
        for sol in ("kissat", "cadical195"):
            lines.append(f"19 {S} {seed} {sol} 1800")
for seed in (1, 2):
    for sol in ("kissat", "cadical195"):
        lines.append(f"19 none {seed} {sol} 1800")
# R=20: context fixed, a0 free (about one solution expected per context), 3 seeds x 2 solvers
for seed in (1, 2, 3):
    for sol in ("kissat", "cadical195"):
        lines.append(f"20 context {seed} {sol} 1800")
# Zaikin's all-ones target at R=18, none vs family, both solvers
for S in ("none", "family"):
    for sol in ("kissat", "cadical195"):
        lines.append(f"18 {S} 1 {sol} 1500 --ones")
# tighter saturation (a4=a5=a6, e8..e10 = -1) at R=18, 2 seeds, kissat
for seed in (1, 2):
    lines.append(f"18 family {seed} kissat 1500 --tight 1")
open("jobs2.txt", "w").write("\n".join(lines) + "\n")
print(len(lines))
