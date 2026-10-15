"""T-043(c): is the SO(9) representative (R-006, data/so_sp/so9_nongraviton_rep_1533.json) Q_0-closed on so(11), so(13)?
Random evaluation of Q_0 O over F_p on so(n)-valued letters (fortuity.full.lifting.values), two seeds; O_18 as a control.
Run from code/."""
import os
from fortuity.full.lifting import load_rep, values, O18, _DATA
from fortuity.full.words import Q_expr
O9, info = load_rep(os.path.join(_DATA, 'so9_nongraviton_rep_1533.json'))
q = Q_expr(O9)
print('O_9 terms', len(O9), '; Q_0 O_9 terms', len(q))
for N in (4, 5, 6):
    for seed in (5, 6):
        print(f'SO({2*N+1}) seed {seed}: Q_0 O_9 = 0:', not values(q, N, 'SO_odd', seed=seed).any())
q = Q_expr(O18())
for N in (3, 4):
    print(f'control O_18 on SO({2*N+1}): Q_0 O = 0:', not values(q, N, 'SO_odd').any())
