import os as _os, sys as _sys
REPO = (_os.environ.get('REPO') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *['..'] * 5))).rstrip('/') + '/'
_sys.path.insert(0, REPO + 'code'); _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
"""Exact check, at N = 3, of Q r = R on EVERY component of the traceless-projected r/R tensors used in (2.24),
and of the raw (as printed) tensors' trace parts (formal)."""
import sys
import time
from fractions import Fraction as Fr
from itertools import product

pass  # symeval is found next to this script
import symeval as S
from fortuity.bmn import dmk

t0 = time.time()
A3 = S.Alg(3)
NIDX = {'r12_03': 3, 'r12_11': 2, 'r12_22': 4, 'r14_13': 4, 'r16_01': 1, 'r16_23': 5}
for name, n in NIDX.items():
    Rn = dmk.R_OF_r[name]
    key = name[1:]
    Draw = {}
    for J in product(range(3), repeat=n):
        Draw[J] = S.padd(A3.Q(A3.expr(dmk.tensor(name, project=False)(*J))),
                         A3.expr(dmk.tensor(Rn, project=False)(*J)), -1)
    nraw_bad = sum(1 for J in Draw if Draw[J])
    if key in dmk.MIXED:
        Pm = dmk.projector(*dmk.MIXED[key])
        nbad = 0
        for I in product(range(3), repeat=n):
            acc = {}
            for c, J in Pm[I]:
                acc = S.padd(acc, Draw[J], c)
            nbad += bool(acc)
    else:
        nbad = nraw_bad
    print('%-7s components %3d: Q r - R != 0 at N=3 on %3d projected components (raw as printed: %3d)  [%.1fs]'
          % (name, 3 ** n, nbad, nraw_bad, time.time() - t0))

# formal traces of the printed mixed tensors (contract first upper with first lower index)
for name in ('r12_11', 'r12_22', 'r14_13', 'r16_12', 'r16_23', 'R12_11', 'R12_22', 'R14_13', 'R16_12', 'R16_23'):
    p, q = dmk.MIXED[name[1:]]
    f = dmk.tensor(name, project=False)
    nz = 0
    for K in product(range(3), repeat=p + q - 2):
        up, lo = K[:p - 1], K[p - 1:]
        tr = dmk.add(*[(1, f(*((a,) + up + (a,) + lo))) for a in range(3)])
        nz += bool(tr)
    print('printed %-7s: formal trace nonzero on %d of %d components' % (name, nz, 3 ** (p + q - 2)))
print('time %.1fs' % (time.time() - t0))
