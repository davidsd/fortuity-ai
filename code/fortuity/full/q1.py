"""One-loop supercharge Q_1 of Budzik-Kulp (arXiv:2512.07771, eq. Q1N4 and App. B) on full-letter multitrace words,
for SU(N) and (group='SO_odd') SO(n), in the superfield conventions of letters.py.  The so(n) case adds, to every
trace rejoining, the second term of the so(n) completeness relation (a segment reversed and transposed); it was written
for T-019 and validated in code/tests/test_q1_groups.py (T-039(a)).

Superfield form:  Q_1(C^A(theta) C^B(theta')) = K f_{ACD} f_{BCE} (theta_1-theta_1')(theta_2-theta_2')(theta_3-theta_3')
                                                  D^tri_{w,z}(C^D(theta), C^E(theta'))
with D^tri_{0,0}(f, g) = 1/2 eps^{ab} d_a f d_b g (fields at coincident points; derivative-dressed pairs would need the
z, w expansion of D^tri, not implemented: Q_1 is provided for pairs of derivative-free letters only, which is all that is
needed to act on BMN-sector words).  The overall constant K is irrelevant for lifting questions and is set to 1.
Gauge structure: f_{ACD} f_{BCE} X^D Y^E = sum_C ([T_C, X])^A ([T_C, Y])^B, and the completeness relation
sum_C (T_C)_{ab} (T_C)_{cd} = kappa (delta_{ad} delta_{bc} - delta_{ab} delta_{cd} / N) reconnects the traces (the 1/N
part cancels in the double commutator); an empty trace gives a factor N.
Leibniz rule over pairs: Q_1(f_1 ... f_n) = sum_{i<j} (-1)^{eps_ij} Q_1(f_i f_j) (the rest in order).
The signs are fixed mechanically by Grassmann reordering (letters of parity |S|+1 mod 2, thetas odd) and checked by
{Q_0, Q_1} = 0 (test_q1.py)."""
from functools import lru_cache
from itertools import product
from collections import defaultdict
from .letters import is_ferm, THETA
from .words import canon_word, canon_multi

D1, D2 = (1, 0), (0, 1)


def _perm_sign(items, parity):
    """Sign of bringing a list of (id) items with given parities into sorted-by-id order (odd items anticommute).
    'items' is the current order; parity(id) -> 0/1."""
    sign = 1
    arr = list(items)
    # bubble sort counting odd-odd swaps
    for i in range(len(arr)):
        for j in range(len(arr) - 1 - i):
            if arr[j] > arr[j + 1]:
                if parity(arr[j]) and parity(arr[j + 1]):
                    sign = -sign
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return sign


def _reorder_sign(current, target, parity):
    """Sign of reordering the Grassmann monomial 'current' (list of distinct symbols) into 'target' order."""
    pos = {s: i for i, s in enumerate(target)}
    return _perm_sign([pos[s] for s in current], lambda i: parity(target[i]))


@lru_cache(maxsize=None)
def pair_rule(S1, S2):
    """Q_1(Psi_{S1} Psi_{S2}) for derivative-free letters (0,0,S1), (0,0,S2): list of (coeff, LD, LE) meaning
    coeff * sum_C [T_C, LD] (x) [T_C, LE] with the Grassmann order LD before LE, LD, LE derivative letters."""
    S1, S2 = tuple(S1), tuple(S2)
    # LHS: coefficient of theta^{S1} theta'^{S2} in Q_1[Psi(theta) Psi(theta')]
    lhs_par = lambda s: 1 if s[0] in ('t', 'p') else (len(S1) + 1) % 2 if s == 'F1' else (len(S2) + 1) % 2
    cur = [('t', i) for i in S1] + ['F1'] + [('p', i) for i in S2] + ['F2']
    tgt = [('t', i) for i in S1] + [('p', i) for i in S2] + ['F1', 'F2']
    sigma_L = _reorder_sign(cur, tgt, lhs_par) * (-1) ** (len(S1) + len(S2))   # Q_1 (odd) passes the thetas
    out = defaultdict(int)
    for choice in product((0, 1), repeat=3):        # 0: theta_i, 1: -theta'_i
        pref = [('p', i) if c else ('t', i) for i, c in zip(THETA, choice)]
        sgn = (-1) ** sum(choice)
        T = tuple(i for i, c in zip(THETA, choice) if not c)
        Tc = tuple(i for i, c in zip(THETA, choice) if c)
        if not set(T) <= set(S1) or not set(Tc) <= set(S2):
            continue
        SD = tuple(i for i in S1 if i not in T)
        SE = tuple(i for i in S2 if i not in Tc)
        par = lambda s: 1 if s[0] in ('t', 'p') else (len(SD) + 1) % 2 if s == 'D' else (len(SE) + 1) % 2
        cur = pref + [('t', i) for i in SD] + ['D'] + [('p', i) for i in SE] + ['E']
        if len(set(cur)) < len(cur):
            continue
        tgt = [('t', i) for i in S1] + [('p', i) for i in S2] + ['D', 'E']
        sigma_R = _reorder_sign(cur, tgt, par)
        c = sigma_L * sgn * sigma_R
        # D^tri_{0,0}: 1/2 (d1 X d2 Y - d2 X d1 Y); the 1/2 is absorbed in K
        out[((1, 0, SD), (0, 1, SE))] += c
        out[((0, 1, SD), (1, 0, SE))] -= c
    return tuple((c, LD, LE) for (LD, LE), c in out.items() if c)


def Q1_multi(multi, N, group='SU'):
    """Q_1 of a canonical multitrace of derivative-free letters -> dict {multitrace: coeff}.
    SO/Sp: group = 'SO_odd' uses the so(n) completeness relation (see _fierz); N is then the matrix size n = 2N+1.
    (usp(2N) needs the Omega-twisted relation; not implemented.)"""
    if group not in ('SU', 'SO_odd'):
        raise NotImplementedError(group)
    # instances: global index in written order
    letters = [L for w in multi for L in w]
    loc = [(a, u) for a, w in enumerate(multi) for u in range(len(w))]
    n = len(letters)
    par = [1 if is_ferm(L) else 0 for L in letters]
    out = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            Li, Lj = letters[i], letters[j]
            if Li[0] or Li[1] or Lj[0] or Lj[1]:
                raise NotImplementedError("Q_1 on derivative-dressed letters is not implemented")
            rule = pair_rule(Li[2], Lj[2])
            if not rule:
                continue
            # sign of bringing instances i, j to the front (i first)
            eps = par[i] * sum(par[k] for k in range(i)) + par[j] * sum(par[k] for k in range(j) if k != i)
            s_front = -1 if eps % 2 else 1
            rest = [k for k in range(n) if k != i and k != j]
            for c, LD, LE in rule:
                for term_multi, term_sign in _fierz(multi, loc, i, j, LD, LE, rest, N, par, group):
                    add = c * s_front * term_sign
                    if isinstance(term_multi, tuple):
                        out[term_multi] += add
                    else:
                        m, coeff = term_multi
                        out[m] += add * coeff
    return {m: c for m, c in out.items() if c}


def _fierz(multi, loc, i, j, LD, LE, rest, N, par, group='SU'):
    """Terms of sum_C [T_C, LD]_slot i (x) [T_C, LE]_slot j inside the multitrace; yields (result, sign) where result is
    a canonical multitrace (or (multi, N-factor) pairs handled by the caller).  Reference Grassmann order: LD, LE, rest."""
    (p, u), (q, v) = loc[i], loc[j]
    new_par = {i: (len(LD[2]) + 1) % 2, j: (len(LE[2]) + 1) % 2}
    parity = lambda k: new_par.get(k, par[k])
    letter = lambda k: LD if k == i else LE if k == j else None
    # necklaces as lists of instance ids
    ids, k = [], 0
    for w in multi:
        ids.append(list(range(k, k + len(w))))
        k += len(w)
    all_letters = [L for w in multi for L in w]
    ref = [i, j] + rest
    res = []
    variants = []
    for cx, cy in product((0, 1), repeat=2):         # 0: T_C before the letter, 1: after (sign -1 each)
        sgn = (-1) ** (cx + cy)
        g1, g2 = u + cx, v + cy
        necks = [list(x) for x in ids]
        factor = 1
        if p != q:
            A, B = necks[p][:g1], necks[p][g1:]
            D, E = necks[q][:g2], necks[q][g2:]
            new = A + E + D + B
            variants.append((sgn, 1, [x for a, x in enumerate(necks) if a not in (p, q)] + [new]))
            if group == 'SO_odd':
                # SO/Sp: so(n) completeness sum_C (T_C)_ab (T_C)_cd ~ delta_ad delta_bc - delta_ac delta_bd.  The second
                # term gives tr(A T B) tr(D T E) -> -tr(A D^T E^T B); with antisymmetric letters X^T = -X each transposed
                # letter gives -1 and the entries are reordered (the Grassmann sign of the reversal is in s_perm below).
                new2 = A + D[::-1] + E[::-1] + B
                variants.append((-sgn * (-1) ** (len(D) + len(E)), 1,
                                 [x for a, x in enumerate(necks) if a not in (p, q)] + [new2]))
        else:
            lo, hi = min(g1, g2), max(g1, g2)
            A, Bm, C = necks[p][:lo], necks[p][lo:hi], necks[p][hi:]
            rest_necks = [x for a, x in enumerate(necks) if a != p]
            nk, factor = list(rest_necks), 1
            if Bm:
                nk.append(Bm)
            else:
                factor = N
            if A + C:
                nk.append(A + C)
            else:
                factor *= N
            variants.append((sgn, factor, nk))
            if group == 'SO_odd':
                # SO/Sp: tr(A T B T C) -> second term -tr(A B^T C) (no empty-trace factor: it is a single trace)
                nk2 = list(rest_necks)
                w2 = A + Bm[::-1] + C
                if w2:
                    nk2.append(w2)
                    variants.append((-sgn * (-1) ** len(Bm), 1, nk2))
                else:
                    variants.append((-sgn * (-1) ** len(Bm) * N, 1, nk2))
    for sgn, factor, necks in variants:
        if any(len(x) == 1 for x in necks):
            continue                                   # tr of a single traceless letter
        # Grassmann sign: written order of the new necklaces vs reference order
        written = [k for x in necks for k in x]
        s_perm = _reorder_sign(written, ref, parity)
        # canonicalise
        total = sgn * factor * s_perm
        cw_list = []
        ok = True
        for x in necks:
            w = tuple(letter(k) if k in (i, j) else all_letters[k] for k in x)
            s, cw = canon_word(w)
            if not s:
                ok = False
                break
            total *= s
            cw_list.append(cw)
        if not ok:
            continue
        sm, cm = canon_multi(cw_list)
        if not sm:
            continue
        res.append((cm, total * sm))
    return res


def Q1_expr(expr, N, group='SU'):
    out = defaultdict(int)
    for m, c in expr.items():
        for m2, c2 in Q1_multi(m, N, group).items():
            out[m2] += c * c2
    return {m: c for m, c in out.items() if c}
