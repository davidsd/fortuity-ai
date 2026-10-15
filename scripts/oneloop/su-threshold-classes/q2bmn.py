"""Two-loop supercharge Q_2 on derivative-free (BMN) letters, up to one overall constant.

Derivation (summary/fortuity-summary.pdf).  In the superfield Feynman rules of the holomorphic twist (propagator
delta^3(theta - theta') x Bochner-Martinelli kernel, cubic vertex), the l-loop correction to Q is a sum of Laman graphs
with l + 2 vertices, one of them the operator (BudzikKulp2025Loop).  For l = 2 the only Laman graph is K_4 minus an
edge.  With the operator at a 2-valent vertex the three interaction vertices form a triangle, whose three fermionic
delta functions give delta^3(0) = 0.  So only the operator at a 3-valent vertex contributes: three letters go in, the
middle interaction vertex has no external leg and the two outer ones have one each,

    Q_2(C^A(th1) C^B(th2) C^C(th3)) = K_2 f_{APX} f_{BPQ} f_{CQY} delta^3(th1 - th2) delta^3(th2 - th3)
                                          D2(C^X(th1), C^Y(th3)).

For underived inputs the kinematic integral is an SU(2)_R-invariant tensor with four derivatives, and the only one is
    D2(f, g) = eps^{ac} eps^{bd} d_a d_b f d_c d_d g = d1^2 f d2^2 g - 2 d1 d2 f d1 d2 g + d2^2 f d1^2 g.
(A term with two letters in and one out would need an invariant with four derivatives on one field: there is none.)
The constant K_2 is fixed by Q_1^2 + {Q_0, Q_2} = 0 (test_q2.py), which is also the check of the whole form:
K_2 = -1/6 in the normalisation of q1d.py / fortuity.full.q1, where Q_1 = f f delta^3 eps^{ab} d_a C d_b C.
The three letters enter symmetrically except for the choice of the middle one: Q_2 on a word is the sum over
unordered triples of letters and over the three choices of the middle letter.

Gauge structure: f f f X^X Y^Y = sum_{P,Q} [T_P, X]_slot1 [T_P, T_Q]_slot2 [T_Q, Y]_slot3 (slot 2 loses its letter).
All generators sit in commutators, so the U(N) completeness relation can be used for SU(N); an empty trace gives N and
a trace of one (traceless) letter gives 0.  Signs: mechanical Grassmann reordering, as in fortuity.full.q1.

U_multi(multi, N) returns the operator with K_2 = 1; on letters (a, b, S) = d1^a d2^b C_S / (a! b!):
    D2 -> 4 (2,0)(0,2) - 2 (1,1)(1,1) + 4 (0,2)(2,0).
"""
from functools import lru_cache
from itertools import product, combinations
from collections import defaultdict
try:
    from fortuity.full.q1 import _reorder_sign
    from fortuity.full.letters import is_ferm, THETA
    from fortuity.full.words import canon_word, canon_multi
except ImportError:            # older clones: the original fullsector modules
    from fortuity.full.q1 import _reorder_sign
    from fortuity.full.letters import is_ferm, THETA
    from fortuity.full.words import canon_word, canon_multi
from fortuity.bmn.modp import P

DERIV = ((4, (2, 0), (0, 2)), (-2, (1, 1), (1, 1)), (4, (0, 2), (2, 0)))


@lru_cache(maxsize=None)
def triple_rule(S1, S2, S3):
    """Theta structure of Q_2(Psi_{S1} Psi_{S2} Psi_{S3}) with S2 at the middle vertex: tuple of (sign, SD, SE), the
    output being D2(Psi_{SD}, Psi_{SE}) with the Grassmann order D before E."""
    S1, S2, S3 = tuple(S1), tuple(S2), tuple(S3)
    fpar = {'F1': (len(S1) + 1) % 2, 'F2': (len(S2) + 1) % 2, 'F3': (len(S3) + 1) % 2}
    lpar = lambda s: fpar[s] if s in fpar else 1
    cur = ([('t', i) for i in S1] + ['F1'] + [('p', i) for i in S2] + ['F2'] + [('q', i) for i in S3] + ['F3'])
    tgt = [('t', i) for i in S1] + [('p', i) for i in S2] + [('q', i) for i in S3] + ['F1', 'F2', 'F3']
    sigma_L = _reorder_sign(cur, tgt, lpar) * (-1) ** (len(S1) + len(S2) + len(S3))    # Q_2 (odd) passes the thetas
    out = defaultdict(int)
    for ch1 in product((0, 1), repeat=3):            # delta(th1 - th2): 0 -> th1_i, 1 -> -th2_i
        for ch2 in product((0, 1), repeat=3):        # delta(th2 - th3): 0 -> th2_i, 1 -> -th3_i
            pref = ([('p', i) if c else ('t', i) for i, c in zip(THETA, ch1)]
                    + [('q', i) if c else ('p', i) for i, c in zip(THETA, ch2)])
            if len(set(pref)) < len(pref):
                continue
            sgn = (-1) ** (sum(ch1) + sum(ch2))
            T1 = {i for i, c in zip(THETA, ch1) if not c}
            P2 = {i for i, c in zip(THETA, ch1) if c} | {i for i, c in zip(THETA, ch2) if not c}
            Q3 = {i for i, c in zip(THETA, ch2) if c}
            if P2 != set(S2) or not T1 <= set(S1) or not Q3 <= set(S3):
                continue
            SD = tuple(i for i in S1 if i not in T1)
            SE = tuple(i for i in S3 if i not in Q3)
            rpar = lambda s: (len(SD) + 1) % 2 if s == 'D' else (len(SE) + 1) % 2 if s == 'E' else 1
            cur = pref + [('t', i) for i in SD] + ['D'] + [('q', i) for i in SE] + ['E']
            tgt = [('t', i) for i in S1] + [('p', i) for i in S2] + [('q', i) for i in S3] + ['D', 'E']
            out[(SD, SE)] += sigma_L * sgn * _reorder_sign(cur, tgt, rpar)
    return tuple((c, SD, SE) for (SD, SE), c in out.items() if c)


def _contract(necks, tok):
    """U(N) completeness on the two occurrences of the generator token tok:
    tr(T A T B) -> tr(A) tr(B),  tr(T A) tr(T B) -> tr(A B)."""
    (p, u), (q, v) = [(a, i) for a, x in enumerate(necks) for i, t in enumerate(x) if t == tok]
    if p == q:
        x = necks[p]
        return [y for a, y in enumerate(necks) if a != p] + [x[u + 1:v], x[v + 1:] + x[:u]]
    x, y = necks[p], necks[q]
    return [z for a, z in enumerate(necks) if a not in (p, q)] + [x[u + 1:] + x[:u] + y[v + 1:] + y[:v]]


def U_multi(multi, N, both_orders=False):
    """Q_2 / K_2 of a canonical multitrace of derivative-free letters -> {multitrace: coeff mod p}.
    both_orders: also add the assignment with the two outer letters exchanged (must double the result)."""
    letters = [L for w in multi for L in w]
    if any(L[0] or L[1] for L in letters):
        raise NotImplementedError("Q_2 is only known on derivative-free letters")
    n = len(letters)
    par = [1 if is_ferm(L) else 0 for L in letters]
    ids, k = [], 0
    for w in multi:
        ids.append(list(range(k, k + len(w))))
        k += len(w)
    out = defaultdict(int)
    for tri in combinations(range(n), 3):
        rest = [t for t in range(n) if t not in tri]
        for m in tri:
            lo, hi = [t for t in tri if t != m]
            for a, c in (((lo, hi), (hi, lo)) if both_orders else ((lo, hi),)):
                rule = triple_rule(letters[a][2], letters[m][2], letters[c][2])
                if not rule:
                    continue
                s_front = _reorder_sign(list(range(n)), [a, m, c] + rest, lambda t: par[t])
                ref = [a, c] + rest
                for c0, SD, SE in rule:
                    newpar = {a: (len(SD) + 1) % 2, c: (len(SE) + 1) % 2}
                    parity = lambda t: newpar.get(t, par[t])
                    for ca, cm, cc in product((0, 1), repeat=3):       # 0: generator before, 1: after (sign -1)
                        necks = []
                        for x in ids:
                            y = []
                            for t in x:
                                if t == a:
                                    y += ['P', a] if ca == 0 else [a, 'P']
                                elif t == m:
                                    y += ['P', 'Q'] if cm == 0 else ['Q', 'P']
                                elif t == c:
                                    y += ['Q', c] if cc == 0 else [c, 'Q']
                                else:
                                    y.append(t)
                            necks.append(y)
                        necks = _contract(_contract(necks, 'P'), 'Q')
                        if any(len(x) == 1 for x in necks):
                            continue
                        factor = N ** sum(1 for x in necks if not x)
                        necks = [x for x in necks if x]
                        written = [t for x in necks for t in x]
                        base = c0 * s_front * (-1) ** (ca + cm + cc) * factor * _reorder_sign(written, ref, parity)
                        for cd, dD, dE in DERIV:
                            LD, LE = dD + (SD,), dE + (SE,)
                            total, cws = base * cd, []
                            for x in necks:
                                s, cw = canon_word(tuple(LD if t == a else LE if t == c else letters[t] for t in x))
                                if not s:
                                    total = 0
                                    break
                                total *= s
                                cws.append(cw)
                            if not total:
                                continue
                            sm, cmul = canon_multi(cws)
                            if sm:
                                out[cmul] = (out[cmul] + total * sm) % P
    return {m: c for m, c in out.items() if c}
