"""Independent construction of O_6^{(0,3)} of arXiv:2312.16443 from its DEFINITION (eq. O3 and appendix tr-r):
   O_6 = 24 v^{(j}_a v^{k)a}_i (r_10)^i_{jk} + 6 u^{a(i} v^{j)}_a (r_14)_{ij} - eps_{a1a2(i} u^{a1k} v^{a2}_{j)} (r_14)^{ij}_k
with the appendix convention that mixed-index objects are traceless-projected:
   [1,2]: O^i_{jk} -> O^i_{jk} - 1/2 delta^i_(j O^a_k)a ;  [2,1]: O^{ij}_k -> O^{ij}_k - 1/2 delta^(i_k O^j)a_a  (round brackets: 1/n!).
Also isolates the two trace-projection pieces, which are separately Q-closed at N=3 (the trace parts of R_10 and R_14
vanish identically as meson polynomials), so that closure cannot fix their coefficients.
Writes: O6_from_r.json (= -6 O_6, same normalisation as O_cckll.json), piece_T2.json, piece_T4.json.
"""
import sys, json, itertools
from fractions import Fraction
from collections import defaultdict
from build_O import eval_term, E, addto, P_, S_
from gens_util import canon_expr
from fortuity.bmn.words import expr_sector

# ---- tensors: name -> (free labels, list of (coeff, factors, sym)) ; internal labels are renamed on expansion ----
TENSORS = {}
def tensor(name, free, templates):
    TENSORS[name] = (free, templates)

# (r_10)^i_{jk} = -2 eps_{a1a2(j} tr(phi^{a1} phi^{a2} phi^i psi_{k)})
tensor('r10', ['i', 'j', 'k'], [(-2, [('eps', ['a1', 'a2', 'j']), ('tr', [P_('a1'), P_('a2'), P_('i'), S_('k')])], [[(0, 2), (1, 3)]])])
# projected [1,2]: r10P^i_{jk} = r10^i_{jk} - 1/4 (delta^i_j r10^b_{kb} + delta^i_k r10^b_{jb})
tensor('r10P', ['i', 'j', 'k'], [(1, [('r10', ['i', 'j', 'k'])], []),
                                 (Fraction(-1, 4), [('delta', ['i', 'j']), ('r10', ['b', 'k', 'b'])], []),
                                 (Fraction(-1, 4), [('delta', ['i', 'k']), ('r10', ['b', 'j', 'b'])], [])])
# (r_14)_{ij}  ([0,2], no projection)
r14a = [(Fraction(5, 9) * 2, [('eps', ['a1', 'a2', 'a3']), ('tr', [S_('i'), S_('j'), P_('a1'), P_('a2'), P_('a3')])], [[(1, 0), (1, 1)]]),
        (Fraction(5, 9) * 1, [('eps', ['a1', 'a2', 'a3']), ('tr', [S_('i'), P_('a1'), S_('j'), P_('a2'), P_('a3')])], [[(1, 0), (1, 2)]])]
for slots in ([S_('j'), S_('a3'), P_('a1'), P_('a2'), P_('a3')], [S_('j'), S_('a3'), P_('a1'), P_('a3'), P_('a2')], [S_('j'), S_('a3'), P_('a3'), P_('a1'), P_('a2')],
              [S_('j'), P_('a1'), S_('a3'), P_('a2'), P_('a3')], [S_('j'), P_('a1'), S_('a3'), P_('a3'), P_('a2')], [S_('j'), P_('a3'), S_('a3'), P_('a1'), P_('a2')],
              [S_('j'), P_('a1'), P_('a2'), S_('a3'), P_('a3')], [S_('j'), P_('a1'), P_('a3'), S_('a3'), P_('a2')], [S_('j'), P_('a3'), P_('a1'), S_('a3'), P_('a2')],
              [S_('j'), P_('a1'), P_('a2'), P_('a3'), S_('a3')], [S_('j'), P_('a1'), P_('a3'), P_('a2'), S_('a3')], [S_('j'), P_('a3'), P_('a1'), P_('a2'), S_('a3')]):
    r14a.append((1, [('eps', ['a1', 'a2', 'i']), ('tr', slots)], [[(0, 2), (1, 0)]]))
r14a += [(Fraction(-1, 3) * 5, [('eps', ['a1', 'a2', 'i']), ('tr', [S_('j'), P_('a1'), P_('a2')]), ('tr', [S_('a3'), P_('a3')])], [[(0, 2), (1, 0)]]),
         (Fraction(-1, 3) * 2, [('eps', ['a1', 'a2', 'i']), ('tr', [S_('j'), P_('a1'), P_('a3')]), ('tr', [S_('a3'), P_('a2')])], [[(0, 2), (1, 0)], [(1, 1), (1, 2)]]),
         (Fraction(-1, 3) * -2, [('eps', ['a1', 'a2', 'i']), ('tr', [S_('j'), P_('a2')]), ('tr', [S_('a3'), P_('a1'), P_('a3')])], [[(0, 2), (1, 0)], [(2, 1), (2, 2)]])]
tensor('r14a', ['i', 'j'], r14a)
# (r_14)^{ij}_k  and its [2,1] projection
r14b = [(12, [('tr', [P_('i'), P_('a'), P_('j'), S_('a'), S_('k')])], [[(0, 0), (0, 1), (0, 2)], [(0, 3), (0, 4)]]),
        (12, [('tr', [P_('i'), P_('a'), P_('j'), S_('a'), S_('k')])], [[(0, 0), (0, 2)], [(0, 3), (0, 4)]]),
        (54, [('tr', [P_('i'), P_('j'), S_('a'), P_('a'), S_('k')])], [[(0, 0), (0, 1), (0, 3)], [(0, 2), (0, 4)]]),
        (-36, [('tr', [P_('i'), P_('j'), S_('a'), P_('a'), S_('k')])], [[(0, 0), (0, 1)], [(0, 2), (0, 4)]])]
tensor('r14b', ['i', 'j', 'k'], r14b)
tensor('r14bP', ['i', 'j', 'k'], [(1, [('r14b', ['i', 'j', 'k'])], []),
                                  (Fraction(-1, 4), [('delta', ['i', 'k']), ('r14b', ['j', 'b', 'b'])], []),
                                  (Fraction(-1, 4), [('delta', ['j', 'k']), ('r14b', ['i', 'b', 'b'])], [])])

_cnt = [0]
def expand(coeff, factors, sym):
    """Replace tensor factors by their templates (renaming internal labels); returns list of plain terms."""
    for fi, (kind, slots) in enumerate(factors):
        if kind in TENSORS:
            free, temps = TENSORS[kind]
            out = []
            for tc, tf, ts in temps:
                _cnt[0] += 1
                sub = dict(zip(free, slots))
                def ren(lab):
                    return sub[lab] if lab in sub else '%s#%d' % (lab, _cnt[0])
                newf = []
                for k2, s2 in tf:
                    if k2 == 'tr':
                        newf.append((k2, [(lk, ren(l)) for lk, l in s2]))
                    else:
                        newf.append((k2, [ren(l) for l in s2]))
                before = factors[:fi]; after = factors[fi + 1:]
                nf = before + newf + after
                # remap sym positions: outer groups referring to factor index > fi shift by len(newf)-1;
                # outer groups referring to the tensor factor itself are not allowed (we never do that)
                def remap_outer(pos):
                    f_, s_ = pos
                    assert f_ != fi, 'outer symmetrisation over a tensor slot not supported'
                    return (f_ + len(newf) - 1, s_) if f_ > fi else (f_, s_)
                nsym = [[remap_outer(p) for p in g] for g in sym] + [[(f_ + fi, s_) for f_, s_ in g] for g in ts]
                out += expand(coeff * tc, nf, nsym)
            return out
    return [(coeff, factors, sym)]

def build(terms):
    tot = E()
    for c, f, s in terms:
        for c2, f2, s2 in expand(Fraction(c), f, s):
            addto(tot, eval_term(c2, f2, s2))
    return tot

def to_int(expr):
    from math import lcm, gcd
    den = 1
    for v in expr.values(): den = lcm(den, v.denominator)
    e = {m: int(v * den) for m, v in expr.items()}
    g = 0
    for v in e.values(): g = gcd(g, abs(v))
    return {m: v // g for m, v in e.items()}, Fraction(den, g)

def save(expr, path, note):
    ei, scale = to_int(expr)
    json.dump({'note': note, 'scale': 'stored = (%s) * defined' % scale, 'sector': list(expr_sector(ei)),
               'terms': [[c, [list(w) for w in m]] for m, c in sorted(ei.items())]}, open(path, 'w'))
    return ei, scale

if __name__ == '__main__':
    # O_6 from the definition
    O6 = build([(24, [('v2', ['j', 'a']), ('v3', ['k', 'a', 'i']), ('r10P', ['i', 'j', 'k'])], [[(0, 0), (1, 0)]]),
                (6, [('u2', ['a', 'i']), ('v2', ['j', 'a']), ('r14a', ['i', 'j'])], [[(0, 1), (1, 0)]]),
                (-1, [('eps', ['a1', 'a2', 'i']), ('u2', ['a1', 'k']), ('v2', ['a2', 'j']), ('r14bP', ['i', 'j', 'k'])], [[(0, 2), (2, 1)]])])
    O6c = canon_expr(O6)
    print('O_6 built from r definitions: canonical terms', len(O6c), 'sector', expr_sector(O6c))
    # compare with the transcription of the printed expansion (O_cckll.json stores 12 * O_printed, O_printed = -6 O_6)
    from gens_util import load_O
    Ot, meta = load_O('O_cckll.json'); Otc = canon_expr(Ot)
    diff = defaultdict(Fraction)
    for m, v in O6c.items(): diff[m] += Fraction(-6 * 12) * v
    for m, v in Otc.items(): diff[m] -= v
    diff = {m: v for m, v in diff.items() if v}
    print('12 * (-6 O_6_built) - O_transcribed: %d nonzero canonical terms -> %s' % (len(diff), 'IDENTICAL' if not diff else 'DIFFERENT'))
    if diff:
        for m, v in list(diff.items())[:8]: print('   ', v, m)
    save(build([(-6, [('v2', ['j', 'a']), ('v3', ['k', 'a', 'i']), ('r10P', ['i', 'j', 'k'])], [[(0, 0), (1, 0)]])]) if False else {m: -6 * v for m, v in O6.items()}, 'O6_from_r.json', '-6 O_6^{(0,3)} built from the r definitions with traceless projections')
    # closure-neutral pieces (trace-projection parts), normalised as they appear in the printed expansion
    T2 = build([(1, [('v2', ['a', 'b']), ('v3', ['b', 'k', 'a']), ('eps', ['c1', 'c2', 'k']), ('tr', [P_('c1'), P_('c2'), P_('d'), S_('d')])], [[(2, 2), (3, 3)]])])
    pre4 = [('eps', ['a1', 'a2', 'j']), ('u2', ['a1', 'b']), ('v2', ['a2', 'b'])]
    T4 = build([(2, pre4 + [('tr', [P_('j'), P_('c'), P_('d'), S_('c'), S_('d')])], [[(3, 0), (3, 1), (3, 2)], [(3, 3), (3, 4)]]),
                (2, pre4 + [('tr', [P_('j'), P_('c'), P_('d'), S_('c'), S_('d')])], [[(3, 0), (3, 2)], [(3, 3), (3, 4)]]),
                (9, pre4 + [('tr', [P_('j'), P_('d'), S_('c'), P_('c'), S_('d')])], [[(3, 0), (3, 1), (3, 3)], [(3, 2), (3, 4)]]),
                (-6, pre4 + [('tr', [P_('j'), P_('d'), S_('c'), P_('c'), S_('d')])], [[(3, 0), (3, 1)], [(3, 2), (3, 4)]])])
    for name, e in (('piece_T2', T2), ('piece_T4', T4)):
        ei, sc = save(e, name + '.json', name + ': trace-projection piece of the printed expansion (unit coefficient)')
        print(name, ': canonical terms', len(canon_expr(ei)), 'sector', expr_sector(ei), 'scale', sc)
