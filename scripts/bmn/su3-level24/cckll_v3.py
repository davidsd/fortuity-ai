"""Eq. (4.8) of arXiv:2312.16443 as corrected in its version 3 (15 October 2026).

Version 3 changes the last term of (r_14^{(0,2)})_{ij} in its appendix A,
    -1/3 eps_{a1a2(i} [ 5 tr(psi_{j)} phi^a1 phi^a2) tr(psi_a3 phi^a3) + 2 tr(psi_{j)} phi^(a1 phi^a3)) tr(psi_a3 phi^a2)
                        - 2 tr(psi_{j)} phi^a2) tr(psi_a3 phi^(a1 phi^a3)) ]                               (v1, v2)
 -> -5/3 eps_{a1a2(i} [ tr(psi_{j)} phi^a1 phi^a2) tr(psi_a3 phi^a3) + tr(psi_{j)} phi^a1 phi^a3) tr(psi_a3 phi^a2)
                        + tr(psi_{j)} phi^a3 phi^a1) tr(psi_a3 phi^a2) ]                                    (v3)
and with it the last line of (4.8), +12 u v eps[...] -> +60 u v eps[...]; the other lines of (4.8) are unchanged up to
notation.  This script builds the v3 operator O = -6 O_6^{(0,3)} two ways, from the printed expansion (build_O.py with
the last line replaced) and from the definition with the corrected r_14 (build_O6.py with r_14 replaced), checks that
the two agree term by term, and writes O_cckll_v3.json, which decompose_multi.py then classifies.
usage (from this directory, with PYTHONPATH=<repo>/code):
    python3 cckll_v3.py
    python3 decompose_multi.py 67108859 20261016 60 runs/multi_v3_p1.json O_cckll_v3.json O_cckll.json R24.json
"""
from fractions import Fraction
from collections import defaultdict
import build_O
from build_O import eval_term, E, addto, P_, S_
from gens_util import canon_expr
from build_O6 import TENSORS, tensor, build, save
from fortuity.bmn.words import expr_sector, Q_expr

# ---- printed expansion of v3: the v2 transcription with its last line replaced ----
pre6 = [('u2', ['a', 'i']), ('v2', ['j', 'a']), ('eps', ['b1', 'b2', 'i'])]
sym6 = [[(2, 2), (3, 0)]]
assert [t[0] for t in build_O.TERMS[-3:]] == [60, 24, -24], 'last line of the v2 transcription not where expected'
V3_LAST = [(Fraction(60), pre6 + [('tr', [S_('j'), P_('b1'), P_('b2')]), ('tr', [S_('c'), P_('c')])], sym6),
           (Fraction(60), pre6 + [('tr', [S_('j'), P_('b1'), P_('c')]), ('tr', [S_('c'), P_('b2')])], sym6),
           (Fraction(60), pre6 + [('tr', [S_('j'), P_('c'), P_('b1')]), ('tr', [S_('c'), P_('b2')])], sym6)]
printed = E()
for coeff, factors, s in build_O.TERMS[:-3] + V3_LAST:
    addto(printed, eval_term(coeff, factors, s))
printed = canon_expr(printed)

# ---- definition with the corrected r_14 ----
free, r14a = TENSORS['r14a']
assert len(r14a) == 17 and [t[0] for t in r14a[-3:]] == [Fraction(-5, 3), Fraction(-2, 3), Fraction(2, 3)]
sym_ij = [[(0, 2), (1, 0)]]
r14a_v3 = r14a[:-3] + [
    (Fraction(-5, 3), [('eps', ['a1', 'a2', 'i']), ('tr', [S_('j'), P_('a1'), P_('a2')]), ('tr', [S_('a3'), P_('a3')])], sym_ij),
    (Fraction(-5, 3), [('eps', ['a1', 'a2', 'i']), ('tr', [S_('j'), P_('a1'), P_('a3')]), ('tr', [S_('a3'), P_('a2')])], sym_ij),
    (Fraction(-5, 3), [('eps', ['a1', 'a2', 'i']), ('tr', [S_('j'), P_('a3'), P_('a1')]), ('tr', [S_('a3'), P_('a2')])], sym_ij)]
tensor('r14a', free, r14a_v3)
O6 = build([(24, [('v2', ['j', 'a']), ('v3', ['k', 'a', 'i']), ('r10P', ['i', 'j', 'k'])], [[(0, 0), (1, 0)]]),
            (6, [('u2', ['a', 'i']), ('v2', ['j', 'a']), ('r14a', ['i', 'j'])], [[(0, 1), (1, 0)]]),
            (-1, [('eps', ['a1', 'a2', 'i']), ('u2', ['a1', 'k']), ('v2', ['a2', 'j']), ('r14bP', ['i', 'j', 'k'])], [[(0, 2), (2, 1)]])])
defined = canon_expr({m: -6 * v for m, v in O6.items()})

diff = defaultdict(Fraction)
for m, v in printed.items():
    diff[m] += v
for m, v in defined.items():
    diff[m] -= v
diff = {m: v for m, v in diff.items() if v}
print('v3 operator: %d canonical terms (printed), %d (from the definition), sector %s'
      % (len(printed), len(defined), expr_sector(printed)))
print('printed - defined: %d nonzero terms -> %s' % (len(diff), 'IDENTICAL' if not diff else 'DIFFERENT'))
print('formally closed:', not Q_expr({m: v for m, v in printed.items()}))
ei, scale = save(printed, 'O_cckll_v3.json', 'eq. (4.8) of arXiv:2312.16443v3, O = -6 O_6^{(0,3)} with the corrected r_14')
print('saved O_cckll_v3.json, scale', scale)
