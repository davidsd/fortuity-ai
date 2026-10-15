"""Lambda = (r_12)^i_j (r_12)^j_i with O_1^{(0,3)} = -(1/2) Q_code(Lambda): triple check and explicit expansion in the
permutation-sum basis of arXiv:2312.16443 (their eq. 4.11 style)."""
import json, itertools, numpy as np
from fractions import Fraction
from collections import defaultdict
from build_O import P_, S_
from build_O6 import tensor, build, to_int
from gens_util import canon_expr, ratrec
from fortuity.bmn.words import Q_expr, expr_sector, canon_word, canon_multi
from fortuity.bmn.evaluate import Points
from fortuity.bmn.symdirect import PERMS, perm_sign, relabel_letter
from fortuity.bmn.modp import P as P1
P2 = 67108837

tensor('r12', ['i', 'j'], [(6, [('tr', [P_('i'), P_('a'), S_('a'), S_('j')])], [[(0, 0), (0, 1)], [(0, 2), (0, 3)]]),
                           (Fraction(-5, 2), [('tr', [P_('i'), S_('a'), P_('a'), S_('j')])], []),
                           (Fraction(5, 2), [('tr', [P_('a'), S_('a'), P_('i'), S_('j')])], [])])
tensor('r12P', ['i', 'j'], [(1, [('r12', ['i', 'j'])], []), (Fraction(-1, 3), [('delta', ['i', 'j']), ('r12', ['a', 'a'])], [])])
tensor('R12', ['i', 'j'], [(4, [('eps', ['j', 'a1', 'a2']), ('u2', ['a1', 'b']), ('v3', ['i', 'a2', 'b'])], []),
                           (3, [('eps', ['j', 'a1', 'a2']), ('u3', ['i', 'a1', 'b']), ('v2', ['a2', 'b'])], [])])
tensor('R12P', ['i', 'j'], [(1, [('R12', ['i', 'j'])], []), (Fraction(-1, 3), [('delta', ['i', 'j']), ('R12', ['a', 'a'])], [])])

Lam = build([(1, [('r12P', ['i', 'j']), ('r12P', ['j', 'i'])], [])])
O1 = build([(-1, [('R12P', ['j', 'i']), ('r12P', ['i', 'j'])], [])])
Lam_i, sL = to_int(Lam); O1_i, sO = to_int(O1)
Lc, Oc = canon_expr(Lam_i), canon_expr(O1_i)
print('Lambda: canonical terms', len(Lc), '| sector', expr_sector(Lc), '| stored = %s * defined' % sL, '| max|coeff|', max(abs(v) for v in Lc.values()))
print('O_1   : canonical terms', len(Oc), '| sector', expr_sector(Oc), '| stored = %s * defined' % sO)
# ---- triple check: D = O_1 + (1/2) Q_code(Lambda) ----
QL = canon_expr(Q_expr(Lam_i))                      # = sL * Q(Lambda_defined)
D = defaultdict(Fraction)
for m, c in Oc.items(): D[m] += Fraction(c, int(sO))
for m, c in QL.items(): D[m] += Fraction(c, int(sL)) / 2
D = {m: v for m, v in D.items() if v}
print('D = O_1 + (1/2) Q(Lambda): formal canonical terms =', len(D), '(nonzero: the identity uses N=3 trace relations)' if D else '(vanishes formally)')
Di, sD = to_int(D) if D else ({}, 1)
def zero_at(expr, N, npts, p, seed):
    degs = sorted({sum(1 for w in m for l in w if 3 <= l <= 5) for m in expr})
    pts = Points(N, npts, max(degs) + 2, degs, seed=seed, p=p)
    return not np.any(pts.expr_vector(expr))
for N, p, seed in ((3, P1, 11), (3, P2, 12), (4, P1, 13), (5, P1, 14)):
    print('   D on random traceless %dx%d matrices, p=%d, 40 points: %s' % (N, N, p, 'ZERO' if zero_at(Di, N, 40, p, seed) else 'nonzero'))
# Weyl behaviour
def weyl_img(expr, s):
    out = defaultdict(int)
    for m, c in expr.items():
        sign = 1; ws = []
        for w in m:
            sw, cw = canon_word(tuple(relabel_letter(l, s) for l in w))
            if sw == 0: sign = 0; break
            sign *= sw; ws.append(cw)
        if sign == 0: continue
        s2, cm = canon_multi(ws)
        if s2 == 0: continue
        out[cm] += sign * s2 * c
    return {m: v for m, v in out.items() if v}
for name, e in (('Lambda', Lc), ('O_1', Oc)):
    rels = []
    for s in PERMS:
        img = weyl_img(e, s)
        rels.append('+' if img == e else ('-' if img == {m: -v for m, v in e.items()} else '?'))
    print('   %s under the 6 relabellings (signs of perms %s): %s' % (name, [perm_sign(s) for s in PERMS], rels))
# ---- orbit (permutation-sum) decomposition of Lambda ----
def orbit_sum(w):
    """canonical expression of  sum over the DISTINCT relabellings sigma(w) (as words, then canonicalised)"""
    imgs = set()
    for s in PERMS:
        imgs.add(tuple(tuple(relabel_letter(l, s) for l in ww) for ww in w))
    out = defaultdict(int)
    for m in imgs:
        sign = 1; ws = []
        for ww in m:
            sw, cw = canon_word(ww)
            if sw == 0: sign = 0; break
            sign *= sw; ws.append(cw)
        if sign == 0: continue
        s2, cm = canon_multi(ws)
        if s2 == 0: continue
        out[cm] += sign * s2
    return {m: v for m, v in out.items() if v}, len(imgs)
remaining = dict(Lc); orbits = []
while remaining:
    w = min(remaining)                       # orbit representative: lexicographically smallest canonical multi
    S, nimg = orbit_sum(w)
    assert w in S, (w, S)
    c = Fraction(remaining[w], S[w])
    for m, v in S.items():
        assert remaining.get(m, 0) == c * v, ('orbit inconsistency', w, m)
        del remaining[m]
    orbits.append((c, w, nimg))
print('Lambda = sum over %d permutation-sum basis elements; coefficients (times %s):' % (len(orbits), sL), sorted({str(c) for c, _, _ in orbits}))
# reconstruct and verify
recon = defaultdict(Fraction)
for c, w, _ in orbits:
    S, _ = orbit_sum(w)
    for m, v in S.items(): recon[m] += c * v
assert {m: v for m, v in recon.items() if v} == {m: Fraction(v) for m, v in Lc.items()}
print('reconstruction of Lambda from the permutation sums: exact')
# ---- LaTeX ----
LET = ['\\phi^1', '\\phi^2', '\\phi^3', '\\psi_1', '\\psi_2', '\\psi_3', 'f']
def tex_word(m):
    return ''.join('\\tr(' + '\\,'.join(LET[l] for l in w) + ')' for w in m)
den = 1
from math import lcm
for c, _, _ in orbits: den = lcm(den, (c / sL).denominator)
lines = []
for c, w, nimg in sorted(orbits, key=lambda t: t[1]):
    cc = c / sL * den                          # coefficient of Lambda_defined times den
    assert cc.denominator == 1
    cc = int(cc)
    sgn = '-' if cc < 0 else '+'; mag = '' if abs(cc) == 1 else str(abs(cc))
    lines.append('%s %s\\,\\big[%s\\big]_{%d}' % (sgn, mag, tex_word(w), nimg))
open('Lambda_tex.txt', 'w').write((' \\\\\n&\\quad ').join([' '.join(lines[i:i + 2]) for i in range(0, len(lines), 2)]))
open('Lambda_tex_den.txt', 'w').write(str(den))
json.dump({'note': 'Lambda = (r12)^i_j (r12)^j_i, r12 = (r_12^{(0,2)}) of arXiv:2312.16443 eq. (A.6) traceless-projected; O_1^{(0,3)} = -(1/2) Q_code(Lambda) on 3x3 matrices',
           'canonical_terms_times_%s' % sL: [[c, [list(w) for w in m]] for m, c in sorted(Lc.items())],
           'orbit_decomposition_times_%d' % den: [[str(c / sL * den), [list(w) for w in wrep], nimg] for c, wrep, nimg in orbits]}, open('Lambda_O1.json', 'w'))
print('LaTeX: %d permutation-sum terms, overall denominator %d, written to Lambda_tex.txt' % (len(lines), den))
print(open('Lambda_tex.txt').read()[:700])
