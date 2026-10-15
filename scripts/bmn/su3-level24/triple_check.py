"""Independent validation of the conventions used for O = -6 O_6^{(0,3)} against the paper's own algebra
(arXiv:2312.16443, appendix): (1) the relation of relations 24 s_2 - 6 s_11 + s_12 = 0 holds FORMALLY with the
printed R's; (2) Q_code r_X = c_X R_X on 3x3 matrices for every r used (fixes signs/normalisations);
(3) closure of O at N=3 is destroyed by rescaling any one of the ten coefficient groups; (4) graviton rows have
rank 28 on diagonal matrices (kernel of the diagonal restriction is one-dimensional); (5) builds the paper's
manifestly exact O_1^{(0,3)} = (1/2) i Q(r_12 r_12) as a positive control for the exactness pipeline."""
import json, numpy as np
from fractions import Fraction
from collections import defaultdict
from build_O import eval_term, E, addto, P_, S_, TERMS
from build_O6 import tensor, build, to_int, TENSORS, save
from gens_util import canon_expr, ratrec, load_O
from fortuity.bmn.words import Q_expr, expr_sector, contents_of_sector
from fortuity.bmn.evaluate import Points
from fortuity.bmn.symdirect import SymDirect, symmetrize_expr
from fortuity.bmn.modp import P, RREFBasis

# ---- the fundamental trace relations R (appendix tr-rel), with the appendix's traceless projections ----
tensor('R10', ['i', 'j', 'k'], [(1, [('eps', ['a1', 'a2', 'j']), ('eps', ['k', 'b1', 'b2']), ('u2', ['a1', 'b1']), ('u3', ['i', 'a2', 'b2'])], [[(0, 2), (1, 0)]])])
tensor('R10P', ['i', 'j', 'k'], [(1, [('R10', ['i', 'j', 'k'])], []),
                                 (Fraction(-1, 4), [('delta', ['i', 'j']), ('R10', ['b', 'k', 'b'])], []),
                                 (Fraction(-1, 4), [('delta', ['i', 'k']), ('R10', ['b', 'j', 'b'])], [])])
tensor('R14a', ['i', 'j'], [(1, [('eps', ['a1', 'a2', 'i']), ('eps', ['b1', 'b2', 'b3']), ('u2', ['a1', 'b1']), ('u2', ['a2', 'b2']), ('v2', ['b3', 'j'])], [[(0, 2), (4, 1)]]),
                            (-2, [('eps', ['a1', 'a2', 'i']), ('eps', ['j', 'b1', 'b2']), ('u3', ['a1', 'b1', 'c']), ('v3', ['a2', 'b2', 'c'])], [[(0, 2), (1, 0)]])])
tensor('R14b', ['i', 'j', 'k'], [(3, [('eps', ['k', 'a1', 'a2']), ('u2', ['a1', 'b']), ('u2', ['i', 'j']), ('v2', ['a2', 'b'])], [[(1, 0), (2, 0), (2, 1)]]),
                                 (4, [('eps', ['k', 'a1', 'a2']), ('u2', ['a1', 'b']), ('u2', ['a2', 'i']), ('v2', ['j', 'b'])], [[(2, 1), (3, 0)]]),
                                 (24, [('eps', ['k', 'a1', 'a2']), ('u3', ['a1', 'b', 'i']), ('v3', ['j', 'a2', 'b'])], [[(1, 2), (2, 0)]])])
tensor('R14bP', ['i', 'j', 'k'], [(1, [('R14b', ['i', 'j', 'k'])], []),
                                  (Fraction(-1, 4), [('delta', ['i', 'k']), ('R14b', ['j', 'b', 'b'])], []),
                                  (Fraction(-1, 4), [('delta', ['j', 'k']), ('R14b', ['i', 'b', 'b'])], [])])
# r_12^{(0,2)} and R_12^{(0,1)} in [1,1] (for the positive control)
tensor('r12', ['i', 'j'], [(6, [('tr', [P_('i'), P_('a'), S_('a'), S_('j')])], [[(0, 0), (0, 1)], [(0, 2), (0, 3)]]),
                           (Fraction(-5, 2), [('tr', [P_('i'), S_('a'), P_('a'), S_('j')])], []),
                           (Fraction(5, 2), [('tr', [P_('a'), S_('a'), P_('i'), S_('j')])], [])])
tensor('r12P', ['i', 'j'], [(1, [('r12', ['i', 'j'])], []), (Fraction(-1, 3), [('delta', ['i', 'j']), ('r12', ['a', 'a'])], [])])
tensor('R12', ['i', 'j'], [(4, [('eps', ['j', 'a1', 'a2']), ('u2', ['a1', 'b']), ('v3', ['i', 'a2', 'b'])], []),
                           (3, [('eps', ['j', 'a1', 'a2']), ('u3', ['i', 'a1', 'b']), ('v2', ['a2', 'b'])], [])])
tensor('R12P', ['i', 'j'], [(1, [('R12', ['i', 'j'])], []), (Fraction(-1, 3), [('delta', ['i', 'j']), ('R12', ['a', 'a'])], [])])

def degrees_of(e):
    return sorted({sum(1 for w in m for l in w if 3 <= l <= 5) for m in e})

def ratio(e1, e2, N=3, seed=5, npts=10):
    """c with e1 = c e2 on N x N random matrices (mod P), consistency over all components; returns (c_rational, ok)."""
    a, sa = to_int(e1); b, sb = to_int(e2)
    degs = sorted(set(degrees_of(a)) | set(degrees_of(b)))
    pts = Points(N, npts, max(degs) + 2, degs, seed=seed)
    va = (pts.expr_vector(a) * pow(int(sa.numerator), -1, P) * int(sa.denominator)) % P   # stored = s * defined -> defined = stored / s
    vb = (pts.expr_vector(b) * pow(int(sb.numerator), -1, P) * int(sb.denominator)) % P
    nz = np.nonzero(vb)[0]
    if nz.size == 0:
        return None, not np.any(va)
    c = (int(va[nz[0]]) * pow(int(vb[nz[0]]), P - 2, P)) % P
    return ratrec(c, P), bool(np.array_equal((c * vb) % P, va))

print('=== (1) relation of relations, formally (canonical expression must be empty) ===')
S = build([(24, [('v2', ['j', 'a']), ('v3', ['k', 'a', 'i']), ('R10P', ['i', 'j', 'k'])], [[(0, 0), (1, 0)]]),
           (-6, [('u2', ['a', 'i']), ('v2', ['j', 'a']), ('R14a', ['i', 'j'])], [[(0, 1), (1, 0)]]),
           (1, [('eps', ['a1', 'a2', 'i']), ('u2', ['a1', 'k']), ('v2', ['a2', 'j']), ('R14bP', ['i', 'j', 'k'])], [[(0, 2), (2, 1)]])])
Sc = canon_expr(S)
print('   24 s_2 - 6 s_11 + s_12 : canonical terms =', len(Sc), '->', 'VANISHES IDENTICALLY' if not Sc else 'NONZERO')
for name, coeffs in (('s_2 alone', (1, 0, 0)), ('s_11 alone', (0, 1, 0)), ('s_12 alone', (0, 0, 1))):
    parts = [(coeffs[0], [('v2', ['j', 'a']), ('v3', ['k', 'a', 'i']), ('R10P', ['i', 'j', 'k'])], [[(0, 0), (1, 0)]]),
             (coeffs[1], [('u2', ['a', 'i']), ('v2', ['j', 'a']), ('R14a', ['i', 'j'])], [[(0, 1), (1, 0)]]),
             (coeffs[2], [('eps', ['a1', 'a2', 'i']), ('u2', ['a1', 'k']), ('v2', ['a2', 'j']), ('R14bP', ['i', 'j', 'k'])], [[(0, 2), (2, 1)]])]
    print('   %s: canonical terms = %d (nonzero, so the relation is not trivial termwise)' % (name, len(canon_expr(build([t for t in parts if t[0]])))))

print('=== (2) Q_code r_X = c_X R_X on random 3x3 matrices (projected objects) ===')
for rname, Rname, idxs in (('r10P', 'R10P', [(0, 1, 2), (0, 0, 1), (1, 2, 2), (2, 0, 0)]),
                           ('r14a', 'R14a', [(0, 1), (0, 0), (2, 2), (1, 2)]),
                           ('r14bP', 'R14bP', [(0, 1, 2), (0, 0, 1), (1, 2, 2), (1, 1, 1)]),
                           ('r12P', 'R12P', [(0, 1), (0, 0), (2, 1), (1, 1)])):
    for idx in idxs:
        labs = list('ijk')[:len(idx)]
        r = build([(1, [(rname, labs)], [])])
        # substitute concrete index values: rebuild with fixed labels via delta factors
        r = build([(1, [(rname, ['i', 'j', 'k'][:len(idx)])] + [('delta', [l, 'v%d' % v]) for l, v in zip(labs, idx)], [])]) if False else None
        # simpler: evaluate the tensor at fixed index values by wrapping the engine with fixed assignment
        from build_O6 import expand
        def at(tname, vals):
            tot = E()
            free, temps = TENSORS[tname]
            for c2, f2, s2 in expand(Fraction(1), [(tname, list(free))], []):
                # fix the free labels to the values by adding delta factors against dummy constant labels is awkward;
                # instead evaluate eval_term restricted: iterate assignments and keep those matching
                from build_O import labels_of, sym_templates, u2, u3, v2, v3, eps, single, mul_expr
                import itertools
                labs2 = labels_of(f2)
                for w, fac in sym_templates(f2, s2):
                    for valsall in itertools.product(range(3), repeat=len(labs2)):
                        val = dict(zip(labs2, valsall))
                        if any(val[fl] != v for fl, v in zip(free, vals)):
                            continue
                        scalar = c2 * w
                        exprs = []
                        ok = True
                        for kind, slots in fac:
                            if kind == 'eps':
                                scalar *= eps(*[val[s] for s in slots])
                            elif kind == 'delta':
                                if val[slots[0]] != val[slots[1]]: scalar = 0
                            elif kind == 'tr':
                                exprs.append(single([(0 if lk == 'p' else 3) + val[lab] for lk, lab in slots]))
                            elif kind == 'u2': exprs.append(u2(*[val[s] for s in slots]))
                            elif kind == 'u3': exprs.append(u3(*[val[s] for s in slots]))
                            elif kind == 'v2': exprs.append(v2(*[val[s] for s in slots]))
                            elif kind == 'v3': exprs.append(v3(*[val[s] for s in slots]))
                            if scalar == 0: ok = False; break
                        if not ok: continue
                        prod = exprs[0]
                        for e_ in exprs[1:]: prod = mul_expr(prod, e_)
                        addto(tot, prod, scalar)
            return tot
        r = at(rname, idx); R = at(Rname, idx)
        if not r and not R:
            print('   %s%s: both zero' % (rname, idx)); continue
        ri, sr = to_int(r)
        q = Q_expr(ri)
        qf = {m: Fraction(c) / sr for m, c in q.items()}
        c, ok = ratio(qf, R)
        print('   Q_code(%s)%s = c * %s%s  with c = %s, consistent on all components: %s' % (rname, idx, Rname, idx, c, ok))

print('=== (3) closure sensitivity: rescale one coefficient group of the printed expansion by 2 ===')
groups = {'T1 (288)': [0], 'T2 (-72)': [1], 'T3 (36[...])': [2, 3, 4, 5], 'T4 (-9[...])': [6, 7, 8, 9], 'T5 (-20[...])': [10, 11],
          'T6 (-36 line 1)': [12, 13, 14], 'T7 (-36 line 2)': [15, 16, 17], 'T8 (-36 line 3)': [18, 19, 20], 'T9 (-36 line 4)': [21, 22, 23], 'T10 (12[...])': [24, 25, 26]}
assert len(TERMS) == 27
termvals = [eval_term(c, f, s) for c, f, s in TERMS]
low_degs = [0, 2]
ptsl = Points(3, 10, 4, low_degs, seed=99)
def closed(expr):
    ei, _ = to_int(expr)
    q = Q_expr(ei)
    return (not q) or not np.any(ptsl.expr_vector(q))
base = E()
for t in termvals: addto(base, t)
print('   unperturbed O: Q-closed at N=3:', closed(base))
for g, idx in groups.items():
    e = E()
    for i, t in enumerate(termvals): addto(e, t, Fraction(2 if i in idx else 1))
    print('   %-18s doubled -> closed at N=3: %s' % (g, closed(e)))

print('=== (4) graviton rows on DIAGONAL 3x3 matrices, Weyl-symmetric subspace of (3;4,4,4) ===')
D = SymDirect(N=3, gen_nmax=3, seed=31337, verbose=False); D.guard.log = False
sec = (3, 4, 4, 4)
k, degs, cpp = D._sector_k(sec)
pd = Points(3, 64, k, degs, seed=4242, diagonal=True)
G, nm = D.graviton_rows(sec, pd)
rd = RREFBasis(pd.ncols); rd.add_batched(G)
print('   symmetrised graviton monomials:', nm, '| rank on diagonal matrices:', rd.rank, '(paper: 28; full-matrix classes mod exact: 29)')
O, meta = load_O('O_cckll.json')
dN = json.load(open('../../../data/su3/graviton_class_vanishing_on_diagonal_j24.json'))
from gens_util import poly_from_terms
GN = symmetrize_expr(poly_from_terms([(t['coeff'], t['monomial']) for t in dN['operator']]))
print('   O on diagonal: zero =', not np.any(pd.expr_vector(O)), '| Sym(R-002 polynomial) on diagonal: zero =', not np.any(pd.expr_vector(GN)))

print('=== (5) positive control: O_1^{(0,3)} = -R12^j_i r12^i_j  vs  (1/2) Q_code(r12^i_j r12^j_i) ===')
O1 = build([(-1, [('R12P', ['j', 'i']), ('r12P', ['i', 'j'])], [])])
Lam = build([(1, [('r12P', ['i', 'j']), ('r12P', ['j', 'i'])], [])])
Lam_i, sL = to_int(Lam)
QL = {m: Fraction(c) / sL for m, c in Q_expr(Lam_i).items()}
O1c = canon_expr(to_int(O1)[0])
print('   O_1 built: canonical terms', len(O1c), 'sector', expr_sector(O1c), '| Lambda sector', expr_sector(Lam_i))
c, ok = ratio(O1, QL)
print('   O_1 = c * Q_code(Lambda) with c =', c, '| consistent:', ok, '  (paper: O_1 = (1/2) i Q_paper(Lambda) = (1/2) Q_code(Lambda))')
print('   O_1 Q-closed at N=3:', closed(O1))
save(O1, 'O1_control.json', "O_1^{(0,3)} of arXiv:2312.16443 (stated there to be exact), built as -R12^j_i r12^i_j")
save(QL, 'QLambda_control.json', "Q_code(r12^i_j r12^j_i): exact by construction")
print('saved O1_control.json, QLambda_control.json')
