"""{Q_0, Q_1} = 0 on random BMN words (formal words, N = 2, 3), and the pair rules printed for inspection."""
import random, sys
from collections import defaultdict
from fortuity.full.letters import name
from fortuity.full.words import Q_multi, Q_expr, sectors_at_level
from fortuity.full.wordenum import words_of_sector
from fortuity.full.q1 import pair_rule, Q1_multi, Q1_expr
for S1, S2 in [((0,1,2), ()), ((0,1,2), (0,1,2)), ((1,2), (0,)), ((1,2), (0,2)), ((0,1,2), (0,)), ((1,2), (0,1,2))]:
    print(name((0,0,S1)), name((0,0,S2)), '->', [(c, name(a), name(b)) for c, a, b in pair_rule(S1, S2)])
random.seed(3)
for N in (2, 3):
    nt = nbad = 0
    for n in (8, 9, 10, 12):
        for s in sectors_at_level(n):
            ws = [w for w in words_of_sector(s, 4) if all(L[0] == 0 and L[1] == 0 for x in w for L in x)]
            for w in random.sample(ws, min(4, len(ws))):
                a = Q_expr(Q1_multi(w, N)); b = Q1_expr(Q_multi(w), N)
                tot = defaultdict(int)
                for m, c in a.items(): tot[m] += c
                for m, c in b.items(): tot[m] += c
                tot = {m: c for m, c in tot.items() if c}
                nt += 1
                if tot:
                    nbad += 1
                    if nbad <= 2:
                        print('FAIL N=%d' % N, w, 'residual terms', len(tot), list(tot.items())[:3])
    print(f'N={N}: {{Q0,Q1}}=0 on {nt} words, failures {nbad}')
