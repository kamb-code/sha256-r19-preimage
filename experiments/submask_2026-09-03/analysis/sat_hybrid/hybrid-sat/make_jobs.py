import sys
lines = []
for R, cap in ((17, 600), (18, 1500)):
    for seed in range(1, 6):
        for S in ("none", "family", "collapse", "context"):
            for sol in ("kissat", "cadical195"):
                lines.append(f"{R} {S} {seed} {sol} {cap}")
for seed in range(1, 6):
    for sol in ("kissat", "cadical195"):
        lines.append(f"19 context {seed} {sol} 600 --fix-a0")
open("jobs1.txt", "w").write("\n".join(lines) + "\n")
print(len(lines))
