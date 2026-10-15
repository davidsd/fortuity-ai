import os as _os, sys as _sys
REPO = (_os.environ.get('REPO') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *['..'] * 5))).rstrip('/') + '/'
_sys.path.insert(0, REPO + 'code'); _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
"""Spot check (exact, N=3) of Q(P r) = P R on a few trace-sensitive non-hw components, plus sensitivity (raw fails)."""
import sys, time
pass  # symeval is found next to this script
import symeval as S
from fortuity.bmn import dmk
t0 = time.time()
A3 = S.Alg(3)
cases = {'r14_13': [(1, 1, 0, 0), (0, 0, 1, 2), (2, 2, 2, 1)], 'r12_22': [(0, 1, 1, 0), (1, 1, 1, 2)],
         'r16_23': [(0, 1, 1, 0, 2), (2, 2, 2, 0, 1)]}
for name, comps in cases.items():
    r, R = dmk.tensor(name), dmk.tensor(dmk.R_OF_r[name])
    rr, RR = dmk.tensor(name, project=False), dmk.tensor(dmk.R_OF_r[name], project=False)
    for I in comps:
        d = S.padd(A3.Q(A3.expr(r(*I))), A3.expr(R(*I)), -1)
        draw = S.padd(A3.Q(A3.expr(rr(*I))), A3.expr(RR(*I)), -1)
        print('%-7s %s: projected Q r - R: %d monomials; as printed: %d monomials  [%.1fs]'
              % (name, I, len(d), len(draw), time.time() - t0))
