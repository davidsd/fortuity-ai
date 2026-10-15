"""T-029: charge-conjugation parity of the SU(3) R-007 class, via the block restriction of T-028.

rho_3 := coefficient of x_1^3 in SU(3) > SU(2) x U(1).  C commutes with the block embedding up to x -> -x, and C is
inner on su(2) (trivial on SU(2) invariants), so as functions on su(2):  rho_3(C O) = - rho_3(O).
Numerical check of this identity on the T-021 representative (tests the fermion signs of conj_expr), then the
conclusion: rho_3 kills SU(3) gravitons + exact in (5;7,4,4) and is nonzero on the class (T-028), and the
non-graviton quotient there is one-dimensional (R-007), so C acts on the class as -1 mod gravitons.
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts/bmn/block-restriction/code'))
from fortuity.bmn.blockrestrict import restrict_u1, load_expr_json
from parity import conj_expr, letters
from fortuity.bmn.evaluate import Points
from fortuity.bmn.words import nferm
from fortuity.bmn.modp import P

t0 = time.time()
rep = load_expr_json(os.path.join(REPO, 'scripts/bmn/su3-level30-representative/runs/rep_nmax5_seed11_expr.json.gz'))
crep = conj_expr(rep)
print(f'rep: {len(rep)} terms, Y = {letters(rep)};  C(rep): {len(crep)} terms  ({time.time()-t0:.0f}s)', flush=True)
a = restrict_u1(rep, 2, (3, 0, 0))
b = restrict_u1(crep, 2, (3, 0, 0))
degs = sorted({sum(nferm(w) for w in m) for m in a} | {sum(nferm(w) for w in m) for m in b})
pts = Points(2, 6, max(degs), degs, seed=23, group='SU')
va, vb = pts.expr_vector(a) % P, pts.expr_vector(b) % P
print(f'rho_3(rep): {len(a)} terms, nonzero entries {np.count_nonzero(va)};  rho_3(C rep): {len(b)} terms')
ok = np.array_equal((va + vb) % P, np.zeros_like(va))
print(('PASS' if ok else 'FAIL') + ': rho_3(C rep) = - rho_3(rep) on su(2)  =>  the R-007 class is C-odd (mod gravitons); '
      f'with Y = {letters(rep)} (even) its graviton parity is P = C(-1)^Y = -1')
