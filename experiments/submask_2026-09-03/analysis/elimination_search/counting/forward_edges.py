import sys
sys.path.insert(0, ".")
from symdep_r20 import build, atoms_with, sym
from L5_50_symdep import occ
# published family frame (a4 = a5 = v context, e8 = e9 = -1), unknowns a0..a3
C, W, unk, legal = build(set(), {5: 4, 6: 6, 7: 7}, {8: 0xFFFFFFFF, 9: 0xFFFFFFFF, 10: None, 11: None})
for k in (1, 2, 3):
    print(f"a{k}-carrying top-level atoms of C{k} (forward edge a{k} -> C{k}; the self term s0(W{k+1}) aside):")
    for c, s in atoms_with(C[k], f'a{k}'):
        print(f"   {'+' if c > 0 else '-'} {s[:150]}")
    # inside s0(W_{k+1}): which one-input functions of a_k does W_{k+1} contain?
    for at, c in C[k]:
        if at[0] == 's0':
            inner = at[1]
            print(f"   inside s0(W{k+1}): W{k+1} carries a{k} through:")
            for c2, s2 in atoms_with(inner, f'a{k}'):
                print(f"        {'+' if c2 > 0 else '-'} {s2[:120]}")
    print()
# the shifted frame: a1 frozen (guess), a4 unknown absorbed by C3, a5 = a6 = v, e9 = e10 = -1
print("shifted frame check (a5 = a6 context-equal, e9 = e10 = -1; a4 unknown): a4-atoms of C1, C2")
C, W, unk, legal = build({4}, {5: 5, 6: 5, 7: 7}, {8: None, 9: 0xFFFFFFFF, 10: 0xFFFFFFFF, 11: None})
print("  legal:", legal)
for j in (1, 2):
    print(f"  C{j}:")
    for c, s in atoms_with(C[j], 'a4'):
        print(f"     {'+' if c > 0 else '-'} {s[:150]}")
    print(f"  C{j} a3-atoms:")
    for c, s in atoms_with(C[j], 'a3'):
        print(f"     {'+' if c > 0 else '-'} {s[:150]}")
