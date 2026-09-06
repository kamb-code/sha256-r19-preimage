#!/usr/bin/env python3
"""Same enumeration as enum_ctx.py, but records per message word (the words of
C0, C1, C2) whether w is absent / linear / heavy in it, so that the words no
context condition can clear are identified.  Usage: python3 enum_words.py a10 [ncore]"""
import sys, itertools, collections, multiprocessing as mp, time, pickle
import symeng as E
from enum_ctx import menu

M = E.M
WORDS = (9, 10, 11, 14, 15, 16, 17, 18)


def evaluate(args):
    wi, first_opt = args
    w = f'a{wi}'
    words = [i for i in range(5, 12) if i != wi]
    per_word = {r: collections.Counter() for r in WORDS}
    fewest = []          # configs with the fewest w-carrying words per constraint
    n_legal = 0
    for rest in itertools.product(*[menu(i, wi) for i in words[1:]]):
        defs = {words[0]: first_opt}
        defs.update(zip(words[1:], rest))
        try:
            S = E.build(defs)
        except E.Illegal:
            continue
        n_legal += 1
        cl = {r: E.classify(S['W'][r], w) for r in WORDS}
        for r in WORDS: per_word[r][cl[r]] += 1
        carrying = {j: [r for r in E.CWORDS[j] if r in cl and cl[r] != 'none'] for j in range(3)}
        key = tuple(len(carrying[j]) for j in range(3))
        fewest.append((sum(key), key, tuple(sorted(cl.items())), dict(defs)))
        if len(fewest) > 200:
            fewest.sort(key=lambda x: x[0]); del fewest[60:]
    return n_legal, per_word, fewest


if __name__ == "__main__":
    wi = int(sys.argv[1].lstrip('a'))
    ncore = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    words = [i for i in range(5, 12) if i != wi]
    first = menu(words[0], wi)
    t0 = time.time()
    n_legal = 0; per_word = {r: collections.Counter() for r in WORDS}; fewest = []
    with mp.Pool(min(ncore, len(first))) as p:
        for nl, pw, fw in p.imap_unordered(evaluate, [(wi, o) for o in first]):
            n_legal += nl; fewest += fw
            for r in WORDS: per_word[r].update(pw[r])
    print(f"w = a{wi}: {n_legal:,} legal configurations, {time.time()-t0:.0f}s")
    print("per message word of C0..C2: how often w is absent / linear / heavy")
    for r in WORDS:
        c = per_word[r]
        print(f"  W{r:<3} none={c.get('none',0):>9,}  lin={c.get('lin',0):>9,}  heavy={c.get('heavy',0):>9,}")
    fewest.sort(key=lambda x: x[0])
    print("\nconfigurations with the fewest w-carrying words in (C0, C1, C2):")
    shown = set()
    for tot, key, cl, defs in fewest:
        if (key, cl) in shown: continue
        shown.add((key, cl))
        print(f"  counts {key}: " + ", ".join(f"W{r}:{c}" for r, c in cl if c != 'none'))
        print(f"      defs = {defs}")
        if len(shown) >= 8: break
    pickle.dump(dict(per_word=per_word, fewest=fewest, n_legal=n_legal), open(f"words_a{wi}.pkl", "wb"))
