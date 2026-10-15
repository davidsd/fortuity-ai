"""Transcribe the SU(3) threshold representative O = -6 O_6^{(0,3)} of Choi, Choi, Kim, Lee, Lee
(arXiv:2312.16443, eq. labelled 'Q-coho' in the TeX source, section 4.1 iii) into the multi-trace
expression format of code/fortuity/bmn, and run the basic checks.

Conventions of the paper (BMN sector): letters phi^m, psi_m, f; Q psi_m = -(i/2) eps_{mnp}[phi^n,phi^p],
Q f = -i[phi^m, psi_m].  The code uses Q psi_m = eps_{mnp} phi_n phi_p, Q f = sum_m [phi_m, psi_m], i.e.
Q_paper = -i Q_code: same kernel, same image, same cohomology.  O contains no f, so no w-normalisation
question arises in the transcription.

Paper generators used in O:  u^{ij} = tr(phi^(i phi^j)),  v^i_j = tr(phi^i psi_j) - (1/3) delta^i_j tr(phi^a psi_a),
v^{ij}_k = tr(phi^(i phi^j) psi_k) - (1/4) delta^i_k tr(phi^(j phi^a) psi_a) - (1/4) delta^j_k tr(phi^(i phi^a) psi_a).
Round brackets = symmetrisation with 1/n!.  Index values 1,2,3 -> 0,1,2 (X,Y,Z).  eps_{012} = +1.
"""
import sys, json, itertools
from fractions import Fraction
from collections import defaultdict
from fortuity.bmn.words import PHI, PSI, add_term, mul_expr, expr_sector, Q_expr, contents_of_sector
from fortuity.bmn.symdirect import symmetrize_expr
from fortuity.bmn.evaluate import Points
import numpy as np

# ---------------- elementary expressions (dict multitrace -> Fraction) ----------------

def E():
    return defaultdict(Fraction)

def single(word, c=Fraction(1)):
    out = E(); add_term(out, (tuple(word),), c); return out

def scale(e, c):
    return {m: v * c for m, v in e.items()} if c else {}

def addto(acc, e, c=Fraction(1)):
    for m, v in e.items():
        acc[m] += v * c
        if acc[m] == 0:
            del acc[m]

def eps(a, b, c):
    if len({a, b, c}) < 3:
        return 0
    return 1 if (a, b, c) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1

def u2(i, j):
    out = E()
    for p in itertools.permutations((i, j)):
        addto(out, single([PHI[p[0]], PHI[p[1]]]), Fraction(1, 2))
    return out

def u3(i, j, k):
    out = E()
    for p in itertools.permutations((i, j, k)):
        addto(out, single([PHI[p[0]], PHI[p[1]], PHI[p[2]]]), Fraction(1, 6))
    return out

def v2(i, j):
    out = E()
    addto(out, single([PHI[i], PSI[j]]))
    if i == j:
        for a in range(3):
            addto(out, single([PHI[a], PSI[a]]), Fraction(-1, 3))
    return out

def S3(i, j, k):
    """tr(phi^(i phi^j) psi_k)"""
    out = E()
    addto(out, single([PHI[i], PHI[j], PSI[k]]), Fraction(1, 2))
    addto(out, single([PHI[j], PHI[i], PSI[k]]), Fraction(1, 2))
    return out

def v3(i, j, k):
    out = E()
    addto(out, S3(i, j, k))
    if i == k:
        for a in range(3):
            addto(out, S3(j, a, a), Fraction(-1, 4))
    if j == k:
        for a in range(3):
            addto(out, S3(i, a, a), Fraction(-1, 4))
    return out

# ---------------- term engine ----------------
# factor = (kind, slots); kind in u2,u3,v2,v3,eps,tr ; slots = list of labels, or for 'tr' list of (letterkind,label)
# sym = list of groups; group = list of (factor_index, slot_index) whose labels are permuted and averaged.

def labels_of(factors):
    L = set()
    for kind, slots in factors:
        if kind == 'fix':
            L.add(slots[0]); continue
        for s in slots:
            L.add(s[1] if kind == 'tr' else s)
    return sorted(L)

def apply_perm(factors, group, perm):
    fac = [(k, list(s)) for k, s in factors]
    def get(fi, si):
        s = fac[fi][1][si]
        return s[1] if fac[fi][0] == 'tr' else s
    def put(fi, si, lab):
        if fac[fi][0] == 'tr':
            fac[fi][1][si] = (fac[fi][1][si][0], lab)
        else:
            fac[fi][1][si] = lab
    labs = [get(fi, si) for fi, si in group]
    for (fi, si), lab in zip(group, [labs[p] for p in perm]):
        put(fi, si, lab)
    return [(k, s) for k, s in fac]

def sym_templates(factors, sym):
    temps = [(Fraction(1), factors)]
    for group in sym:
        new = []
        n = len(group)
        for w, fac in temps:
            for perm in itertools.permutations(range(n)):
                new.append((w / Fraction(len(list(itertools.permutations(range(n))))), apply_perm(fac, group, perm)))
        temps = new
    return temps

def eval_term(coeff, factors, sym):
    total = E()
    labs = labels_of(factors)
    for w, fac in sym_templates(factors, sym):
        for vals in itertools.product(range(3), repeat=len(labs)):
            val = dict(zip(labs, vals))
            scalar = Fraction(coeff) * w
            exprs = []
            for kind, slots in fac:
                if kind == 'eps':
                    scalar *= eps(*[val[s] for s in slots])
                    if scalar == 0:
                        break
                elif kind == 'delta':
                    if val[slots[0]] != val[slots[1]]:
                        scalar = 0
                        break
                elif kind == 'fix':
                    if val[slots[0]] != slots[1]:
                        scalar = 0
                        break
                elif kind == 'tr':
                    word = [(PHI if lk == 'p' else PSI)[val[lab]] for lk, lab in slots]
                    exprs.append(single(word))
                elif kind == 'u2':
                    exprs.append(u2(*[val[s] for s in slots]))
                elif kind == 'u3':
                    exprs.append(u3(*[val[s] for s in slots]))
                elif kind == 'v2':
                    exprs.append(v2(*[val[s] for s in slots]))
                elif kind == 'v3':
                    exprs.append(v3(*[val[s] for s in slots]))
                else:
                    raise ValueError(kind)
            if scalar == 0:
                continue
            prod = exprs[0]
            for e in exprs[1:]:
                prod = mul_expr(prod, e)
            addto(total, prod, scalar)
    return total

# ---------------- the operator, term by term from the paper ----------------
P_ = lambda lab: ('p', lab)
S_ = lambda lab: ('s', lab)
TERMS = []
def T(coeff, factors, sym):
    TERMS.append((Fraction(coeff), factors, sym))

# 288 v^j_a v^{ka}_i eps_{c1c2(j} tr(phi^{c1} phi^{c2} phi^i psi_{k)})
T(288, [('v2', ['j', 'a']), ('v3', ['k', 'a', 'i']), ('eps', ['c1', 'c2', 'j']),
        ('tr', [P_('c1'), P_('c2'), P_('i'), S_('k')])], [[(2, 2), (3, 3)]])
# -72 v^a_b v^{bk}_a eps_{c1c2(k} tr(phi^{c1} phi^{c2} phi^d psi_{d)})
T(-72, [('v2', ['a', 'b']), ('v3', ['b', 'k', 'a']), ('eps', ['c1', 'c2', 'k']),
        ('tr', [P_('c1'), P_('c2'), P_('d'), S_('d')])], [[(2, 2), (3, 3)]])
# +36 eps_{a1a2i} u^{a1k} v^{a2}_j [ 2 tr(phi^{(i} phi^c phi^{j)} psi_{(c} psi_{k)}) + 2 tr(phi^{(i|} phi^c phi^{|j)} psi_{(c} psi_{k)})
#                                  + 9 tr(phi^{(i} phi^j psi_{(c} phi^{c)} psi_{k)}) - 6 tr(phi^{(i} phi^{j)} psi_{(c} phi^c psi_{k)}) ]
pre3 = [('eps', ['a1', 'a2', 'i']), ('u2', ['a1', 'k']), ('v2', ['a2', 'j'])]
T(36 * 2, pre3 + [('tr', [P_('i'), P_('c'), P_('j'), S_('c'), S_('k')])], [[(3, 0), (3, 1), (3, 2)], [(3, 3), (3, 4)]])
T(36 * 2, pre3 + [('tr', [P_('i'), P_('c'), P_('j'), S_('c'), S_('k')])], [[(3, 0), (3, 2)], [(3, 3), (3, 4)]])
T(36 * 9, pre3 + [('tr', [P_('i'), P_('j'), S_('c'), P_('c'), S_('k')])], [[(3, 0), (3, 1), (3, 3)], [(3, 2), (3, 4)]])
T(36 * -6, pre3 + [('tr', [P_('i'), P_('j'), S_('c'), P_('c'), S_('k')])], [[(3, 0), (3, 1)], [(3, 2), (3, 4)]])
# -9 eps_{a1a2j} u^{a1b} v^{a2}_b [ same bracket with (i,j,c,k) -> (j,d,c,d) ]
pre4 = [('eps', ['a1', 'a2', 'j']), ('u2', ['a1', 'b']), ('v2', ['a2', 'b'])]
T(-9 * 2, pre4 + [('tr', [P_('j'), P_('c'), P_('d'), S_('c'), S_('d')])], [[(3, 0), (3, 1), (3, 2)], [(3, 3), (3, 4)]])
T(-9 * 2, pre4 + [('tr', [P_('j'), P_('c'), P_('d'), S_('c'), S_('d')])], [[(3, 0), (3, 2)], [(3, 3), (3, 4)]])
T(-9 * 9, pre4 + [('tr', [P_('j'), P_('d'), S_('c'), P_('c'), S_('d')])], [[(3, 0), (3, 1), (3, 3)], [(3, 2), (3, 4)]])
T(-9 * -6, pre4 + [('tr', [P_('j'), P_('d'), S_('c'), P_('c'), S_('d')])], [[(3, 0), (3, 1)], [(3, 2), (3, 4)]])
# -20 u^{ai} v^j_a eps_{b1b2b3} [ 2 tr(psi_{(i} psi_{j)} phi^{b1} phi^{b2} phi^{b3}) + tr(psi_{(i} phi^{b1} psi_{j)} phi^{b2} phi^{b3}) ]
pre5 = [('u2', ['a', 'i']), ('v2', ['j', 'a']), ('eps', ['b1', 'b2', 'b3'])]
T(-20 * 2, pre5 + [('tr', [S_('i'), S_('j'), P_('b1'), P_('b2'), P_('b3')])], [[(3, 0), (3, 1)]])
T(-20 * 1, pre5 + [('tr', [S_('i'), P_('b1'), S_('j'), P_('b2'), P_('b3')])], [[(3, 0), (3, 2)]])
# -36 u^{ai} v^j_a eps_{b1b2(i} [ ... psi_{j)} ... ]  four groups of three single traces
pre6 = [('u2', ['a', 'i']), ('v2', ['j', 'a']), ('eps', ['b1', 'b2', 'i'])]
sym6 = [[(2, 2), (3, 0)]]
for slots in ([S_('j'), S_('c'), P_('b1'), P_('b2'), P_('c')], [S_('j'), S_('c'), P_('b1'), P_('c'), P_('b2')], [S_('j'), S_('c'), P_('c'), P_('b1'), P_('b2')],
              [S_('j'), P_('b1'), S_('c'), P_('b2'), P_('c')], [S_('j'), P_('b1'), S_('c'), P_('c'), P_('b2')], [S_('j'), P_('c'), S_('c'), P_('b1'), P_('b2')],
              [S_('j'), P_('b1'), P_('b2'), S_('c'), P_('c')], [S_('j'), P_('b1'), P_('c'), S_('c'), P_('b2')], [S_('j'), P_('c'), P_('b1'), S_('c'), P_('b2')],
              [S_('j'), P_('b1'), P_('b2'), P_('c'), S_('c')], [S_('j'), P_('b1'), P_('c'), P_('b2'), S_('c')], [S_('j'), P_('c'), P_('b1'), P_('b2'), S_('c')]):
    T(-36, pre6 + [('tr', slots)], sym6)
# +12 u^{ai} v^j_a eps_{b1b2(i} [ 5 tr(psi_{j)} phi^{b1} phi^{b2}) tr(psi_c phi^c) + 2 tr(psi_{j)} phi^{(b1} phi^{c)}) tr(psi_c phi^{b2})
#                                - 2 tr(psi_{j)} phi^{b2}) tr(psi_c phi^{(b1} phi^{c)}) ]
T(12 * 5, pre6 + [('tr', [S_('j'), P_('b1'), P_('b2')]), ('tr', [S_('c'), P_('c')])], [[(2, 2), (3, 0)]])
T(12 * 2, pre6 + [('tr', [S_('j'), P_('b1'), P_('c')]), ('tr', [S_('c'), P_('b2')])], [[(2, 2), (3, 0)], [(3, 1), (3, 2)]])
T(12 * -2, pre6 + [('tr', [S_('j'), P_('b2')]), ('tr', [S_('c'), P_('b1'), P_('c')])], [[(2, 2), (3, 0)], [(4, 1), (4, 2)]])

if __name__ == '__main__':
    O = E()
    for coeff, factors, sym in TERMS:
        addto(O, eval_term(coeff, factors, sym))
    print('terms in O (rational coefficients):', len(O))
    from math import lcm
    den = 1
    for v in O.values():
        den = lcm(den, v.denominator)
    Oint = {m: int(v * den) for m, v in O.items()}
    g = 0
    from math import gcd
    for v in Oint.values():
        g = gcd(g, abs(v))
    Oint = {m: v // g for m, v in Oint.items()}
    print('common denominator cleared:', den, '; common factor removed:', g, '; O_int = (%d/%d) O_paper' % (den, g))
    sec = expr_sector(Oint)
    print('sector (2J; q1,q2,q3) =', sec, ' expected (3, 4, 4, 4)')
    maxc = max(abs(v) for v in Oint.values())
    print('number of multi-trace terms:', len(Oint), '; max |coeff| =', maxc)
    # checks
    QO = Q_expr(Oint)
    print('formal Q(O) (no trace relations): %d multi-trace terms -> %s' % (len(QO), 'NOT formally closed' if QO else 'formally closed'))
    low = (2, 4, 4, 4)
    degs_low = sorted({sum(c[3:6]) for c in contents_of_sector(low)})
    degs = sorted({sum(c[3:6]) for c in contents_of_sector(sec)})
    for N in (3, 4, 5):
        pl = Points(N, 12, max(degs_low) + 2, degs_low, seed=101 + N)
        v = pl.expr_vector(QO) if QO else np.zeros(1)
        print('  Q(O) at random traceless %dx%d points: %s' % (N, N, 'zero  (Q-closed at N=%d)' % N if not np.any(v) else 'NONZERO (%d of %d components)' % (int(np.count_nonzero(v)), v.size)))
    pg = Points(3, 12, max(degs) + 2, degs, seed=7)
    pd = Points(3, 12, max(degs) + 2, degs, seed=8, diagonal=True)
    vg, vd = pg.expr_vector(Oint), pd.expr_vector(Oint)
    print('O at generic 3x3 points nonzero:', bool(np.any(vg)), '; O on diagonal 3x3 points zero:', not np.any(vd))
    sym = symmetrize_expr(Oint)
    diff = defaultdict(int)
    for m, c in sym.items(): diff[m] += c
    for m, c in Oint.items(): diff[m] -= 6 * c
    print('Weyl(S3)-invariant in the code sense (sum over 6 Weyl images == 6 O):', not any(diff.values()))
    json.dump({'source': 'arXiv:2312.16443 eq. (Q-coho), O = -6 O_6^{(0,3)}, transcribed 2026-09-26',
               'normalisation': 'O_int = (%d/%d) * O_paper' % (den, g), 'sector': list(sec),
               'letters': '0,1,2 = phi_1,2,3 ; 3,4,5 = psi_1,2,3 ; 6 = f',
               'terms': [[c, [list(w) for w in m]] for m, c in sorted(Oint.items())]},
              open(sys.argv[1] if len(sys.argv) > 1 else 'O_cckll.json', 'w'))
    print('saved')
