import time
from collections import defaultdict
from fortuity.bmn.words import add_term, w_gen
from fortuity.bmn.g2_checks import so7_class
from fortuity.full.lifting import O18, to_full, values
from fortuity.full.words import Q_expr, sector_of
from fortuity.full.q1 import Q1_expr
t = time.time()
O = O18()
Ob = so7_class()
w = w_gen((0,), 1)
H = defaultdict(int)
for m1, c1 in Ob.items():
    for m2, c2 in w.items():
        add_term(H, tuple(m1) + tuple(m2), c1 * c2)
H = to_full({m: c for m, c in H.items() if c})
print('O18', len(O), 'hairy', len(H), set(sector_of(m) for m in H), flush=True)
for name, X in [('O18', O), ('hairy', H)]:
    print(name, 'Q0 closed on so(7):', not values(Q_expr(X), 3, 'SO_odd').any(), flush=True)
    q1 = Q1_expr(X, 7, 'SO_odd')
    print(name, 'Q1:', len(q1), 'terms; nonzero on so(7):', bool(values(q1, 3, 'SO_odd').any()),
          '| nonzero on Cartan of so(7):', bool(values(q1, 3, 'SO_odd', cartan=True).any()), f'({time.time()-t:.0f} s)', flush=True)
from fortuity.full.lifting import rho2
for name, X in [('O18', O), ('hairy', H)]:
    r = rho2(Q1_expr(X, 7, 'SO_odd'))
    print(name, 'rho2(Q1):', len(r), 'terms; nonzero on so(5):', bool(r) and bool(values(r, 2, 'SO_odd').any()), flush=True)
