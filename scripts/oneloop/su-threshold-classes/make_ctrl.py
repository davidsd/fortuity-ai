"""Control vectors Q_1 Q_0 u (u = a few SU(3)_R hw vectors of the sector one step up) for fullpar.py:
Q_1 Q_0 u = -Q_0 Q_1 u is Q_0-exact, so adding these rows must not raise the rank."""
import sys, pickle, os
from fastcontents import prime_cache
from fortuity.bmn.modp import P
from fortuity.full.sector import words_of
from fortuity.full.hw import hw_vectors
from fortuity.full.words import Q_multi
from fortuity.full.q1 import Q1_multi
CASES = {'su2_j36': (2, (9, 9, 12, 12, 12), 9, 3), 'su3_j24': (3, (3, 3, 8, 8, 8), 9, 6),
         'su3_j30': (3, (5, 5, 14, 8, 8), 10, 6), 'su2_j24': (2, (5, 5, 8, 8, 8), 7, 3)}
case = sys.argv[1]
if case in CASES:
    N, SEC, Y, ML = CASES[case]
else:      # generic 'su<N>:<2J>:<q1>,<q2>,<q3>' (as in oneloop_test.py)
    g, twoJ, q = case.split(':')
    N, q = int(g[2:]), tuple(int(x) for x in q.split(','))
    SEC, Y, ML = (int(twoJ), int(twoJ)) + tuple(2 * x for x in q), sum(q) - int(twoJ), {2: 3, 3: 6}[N]
D = f'classes/{case.replace(":", "_").replace(",", "-")}'
UP = (SEC[0] + 1, SEC[1] + 1) + SEC[2:]
prime_cache(UP, Y - 1)
ws = words_of(UP, Y - 1, ML)
blocks = hw_vectors(ws)
us = []
for widx, H in blocks:
    for r in range(H.shape[0]):
        us.append({ws[int(widx[c])]: int(H[r, c]) for c in range(H.shape[1]) if H[r, c]})
        if len(us) == 3:
            break
    if len(us) == 3:
        break
def app(op, v):
    out = {}
    for w, c in v.items():
        for m, c2 in op(w).items():
            out[m] = (out.get(m, 0) + c * c2) % P
    return {m: c for m, c in out.items() if c}
ctrl = [app(lambda w: Q1_multi(w, N), app(Q_multi, u)) for u in us]
os.makedirs(D, exist_ok=True)
pickle.dump(ctrl, open(f'{D}/ctrl.pkl', 'wb'))
print(case, 'UP words', len(ws), 'controls', [len(c) for c in ctrl])
