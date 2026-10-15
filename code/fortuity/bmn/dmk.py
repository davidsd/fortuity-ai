"""The brane-fused SU(3) operator (O^3_30)^{ijk} of de Mello Koch, Kim, Kim, Lee, Lee (arXiv:2412.08695), rebuilt from
the formulas printed there, and the SU(2) threshold operator O^2_24 (their eq. (2.6)).

Construction (dMK sec. 2.1.1): u_n, v_n are the single-trace gravitons (A.1); the R_a are fundamental trace relations,
graviton polynomials that vanish at N = 3 up to Q-exact terms, R_a = Q r_a (A.2, A.3). A relation of relations
sum_a c_a f_a R_a = 0 (their (2.20), 22 terms s_a = f_a R_a of (A.4)) gives the Q-closed operator sum_a c_a f_a r_a,
which is (2.24).

Three readings of the paper are needed (each is settled by a check in code/tests/test_dmk.py, not assumed):
  1. The R and r tensors of (A.2), (A.3) with both upper and lower indices are their traceless parts. As printed,
     Q r = R fails on the trace components and (2.20) does not hold formally; after the traceless projection
     Q r = R holds at N = 3 with constant exactly 1 in our convention (Q psi_m = eps_{mnp} phi_n phi_p) and (2.20) holds
     identically (zero formal terms).
  2. Grassmann sign: Q(f r) = (-1)^{|f|} f Q r. Five terms of (2.24) (s1, s3, s4, s5, s9) have a fermion-odd graviton
     prefactor v v v; they need an extra minus sign for the operator to be Q-closed. These terms do not contribute to
     the x^3 coefficient of the SU(2) x U(1) restriction.
  3. The 8/5 term (s22) prints r^(0,2)_16, which does not exist; the [0,1] partner of R^(0,2)_16 is r^(0,3)_16 (A.3).

Letters (fortuity.bmn.words): phi^a -> a, psi_a -> 3 + a (dMK index 1,2,3 -> 0,1,2). Coefficients are exact Fractions;
(..) symmetrisations are normalised (1/n!). The [3,0] highest-weight component is ijk = (0, 0, 0).
Transcribed 2026-10-01/02 by Claude agents (review C8 of R-016).
"""
from fractions import Fraction as Fr
from collections import defaultdict
from itertools import permutations, product
from functools import lru_cache

from .words import canon_word, add_term, mul_expr

R3 = range(3)


# ---------------------------------------------------------------- expression helpers
def eps(a, b, c):
    if len({a, b, c}) < 3:
        return 0
    return 1 if (a, b, c) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1


def _ph(a):
    return {a: 1}


def _ps(a):
    return {3 + a: 1}


def tr(*slots):
    """Single trace of a product of letters (each slot a {letter: coeff} dict)."""
    out = defaultdict(int)
    for combo in product(*[list(s.items()) for s in slots]):
        c = Fr(1)
        w = []
        for l, v in combo:
            c *= v
            w.append(l)
        if c == 0:
            continue
        s, cw = canon_word(tuple(w))
        if s:
            add_term(out, (cw,), s * c)
    return out


def add(*terms):
    """add((c, expr), ...) with exact coefficients."""
    out = defaultdict(int)
    for c, e in terms:
        if c == 0:
            continue
        for m, v in e.items():
            out[m] += c * v
            if out[m] == 0:
                del out[m]
    return out


def mul(*es):
    out = {(): Fr(1)}
    for e in es:
        out = mul_expr(out, e)
    return out


def _sym(f, idx):
    """Normalised symmetrisation over the index tuple idx (f takes a tuple)."""
    perms = list(permutations(idx))
    return add(*[(Fr(1, len(perms)), f(p)) for p in perms])


# ---------------------------------------------------------------- gravitons (A.1)
@lru_cache(maxsize=None)
def u2(i, j):
    return dict(add((Fr(1, 2), tr(_ph(i), _ph(j))), (Fr(1, 2), tr(_ph(j), _ph(i)))))


@lru_cache(maxsize=None)
def u3(i, j, k):
    return dict(add(*[(Fr(1, 6), tr(_ph(a), _ph(b), _ph(c))) for a, b, c in permutations((i, j, k))]))


@lru_cache(maxsize=None)
def v2(i, j):
    """v^i_j (traceless)."""
    t = [(Fr(1), tr(_ph(i), _ps(j)))]
    if i == j:
        t += [(Fr(-1, 3), tr(_ph(a), _ps(a))) for a in R3]
    return dict(add(*t))


def _trpp_s(i, j, k):
    return add((Fr(1, 2), tr(_ph(i), _ph(j), _ps(k))), (Fr(1, 2), tr(_ph(j), _ph(i), _ps(k))))


@lru_cache(maxsize=None)
def v3(i, j, k):
    """v^{ij}_k (traceless)."""
    t = [(Fr(1), _trpp_s(i, j, k))]
    if i == k:
        t += [(Fr(-1, 4), _trpp_s(j, a, a)) for a in R3]
    if j == k:
        t += [(Fr(-1, 4), _trpp_s(i, a, a)) for a in R3]
    return dict(add(*t))


# ---------------------------------------------------------------- r's (A.3), as printed
@lru_cache(maxsize=None)
def r12_03(i, j, k):
    """(r^(0,2)_12)_{ijk} = 1/2 eps_{a1 a2 (i} tr(phi^a1 psi_j phi^a2 psi_k))."""
    def f(p):
        i_, j_, k_ = p
        return add(*[(Fr(1, 2) * eps(a1, a2, i_), tr(_ph(a1), _ps(j_), _ph(a2), _ps(k_)))
                     for a1 in R3 for a2 in R3 if eps(a1, a2, i_)])
    return dict(_sym(f, (i, j, k)))


@lru_cache(maxsize=None)
def r12_11(i, j):
    """(r^(0,2)_12)^i_j = 6 tr(phi^(i phi^a) psi_(a psi_j)) - 5 tr(phi^[i psi_a phi^a] psi_j)."""
    t = []
    for a in R3:
        for (x, y) in ((i, a), (a, i)):
            for (z, w) in ((a, j), (j, a)):
                t.append((Fr(6, 4), tr(_ph(x), _ph(y), _ps(z), _ps(w))))
        t.append((Fr(-5, 2), tr(_ph(i), _ps(a), _ph(a), _ps(j))))
        t.append((Fr(5, 2), tr(_ph(a), _ps(a), _ph(i), _ps(j))))
    return dict(add(*t))


@lru_cache(maxsize=None)
def r12_22(i, j, k, l):
    """(r^(0,2)_12)^{ij}_{kl} = tr(phi^(i phi^j) psi_(k psi_l))."""
    t = []
    for (x, y) in ((i, j), (j, i)):
        for (z, w) in ((k, l), (l, k)):
            t.append((Fr(1, 4), tr(_ph(x), _ph(y), _ps(z), _ps(w))))
    return dict(add(*t))


@lru_cache(maxsize=None)
def r14_13(i, j, k, l):
    """(r^(0,3)_14)^i_{jkl} = -1/2 tr(phi^i psi_(j psi_k psi_l))."""
    return dict(add(*[(Fr(-1, 12), tr(_ph(i), _ps(a), _ps(b), _ps(c))) for a, b, c in permutations((j, k, l))]))


@lru_cache(maxsize=None)
def r16_01(i):
    """(r^(0,3)_16)_i."""
    t = []
    for b1 in R3:
        for b2 in R3:
            t.append((Fr(39, 4), tr(_ps(i), _ps(b1), _ps(b2), _ph(b1), _ph(b2))))
            t.append((Fr(39, 4), tr(_ps(i), _ph(b1), _ph(b2), _ps(b1), _ps(b2))))
            t.append((Fr(2), tr(_ps(i), _ps(b1), _ph(b1), _ps(b2), _ph(b2))))
            t.append((Fr(-61, 4), tr(_ps(i), _ps(b1), _ph(b2), _ps(b2), _ph(b1))))
            t.append((Fr(97, 4), tr(_ps(i), _ph(b1), _ps(b1), _ps(b2), _ph(b2))))
            t.append((Fr(-41, 4), tr(_ps(i), _ph(b2), _ps(b1), _ps(b2), _ph(b1))))
            t.append((Fr(-5), tr(_ps(i), _ps(b1), _ph(b1), _ph(b2), _ps(b2))))
            t.append((Fr(-25, 2), tr(_ps(i), _ps(b1), _ph(b2), _ph(b1), _ps(b2))))
            t.append((Fr(2), tr(_ps(i), _ph(b1), _ps(b1), _ph(b2), _ps(b2))))
            t.append((Fr(-61, 4), tr(_ps(i), _ph(b2), _ps(b1), _ph(b1), _ps(b2))))
            t.append((Fr(-11, 4), mul(tr(_ph(b1), _ph(b2)), tr(_ps(i), _ps(b1), _ps(b2)))))
            t.append((Fr(-27, 2), mul(tr(_ps(b1), _ps(b2)), tr(_ps(i), _ph(b1), _ph(b2)))))
            t.append((Fr(29, 4), mul(tr(_ph(b2), _ps(b2)), tr(_ps(i), _ps(b1), _ph(b1)))))
            t.append((Fr(-29, 4), mul(tr(_ph(b2), _ps(b2)), tr(_ps(i), _ph(b1), _ps(b1)))))
    return dict(add(*t))


@lru_cache(maxsize=None)
def r16_12(i, j, k):
    """(r^(0,3)_16)^i_{jk}, symmetric in jk."""
    def f(p):
        j_, k_ = p
        t = []
        for b in R3:
            t.append((Fr(2), tr(_ps(j_), _ps(k_), _ps(b), _ph(b), _ph(i))))
            t.append((Fr(-4), tr(_ps(j_), _ps(k_), _ps(b), _ph(i), _ph(b))))
            t.append((Fr(-1), tr(_ps(j_), _ps(b), _ps(k_), _ph(b), _ph(i))))
            t.append((Fr(-1), tr(_ps(j_), _ps(b), _ps(k_), _ph(i), _ph(b))))
            t.append((Fr(-2), tr(_ps(j_), _ps(k_), _ph(b), _ps(b), _ph(i))))
            t.append((Fr(-2), tr(_ps(j_), _ps(k_), _ph(i), _ps(b), _ph(b))))
            t.append((Fr(7), tr(_ps(j_), _ps(b), _ph(b), _ps(k_), _ph(i))))
            t.append((Fr(7), tr(_ps(j_), _ph(b), _ps(b), _ps(k_), _ph(i))))
            t.append((Fr(-11), tr(_ps(j_), _ps(b), _ph(i), _ps(k_), _ph(b))))
            t.append((Fr(-11), tr(_ps(j_), _ph(i), _ps(b), _ps(k_), _ph(b))))
            t.append((Fr(-4), tr(_ps(j_), _ps(k_), _ph(b), _ph(i), _ps(b))))
            t.append((Fr(2), tr(_ps(j_), _ps(k_), _ph(i), _ph(b), _ps(b))))
            t.append((Fr(3), mul(tr(_ps(j_), _ps(b)), tr(_ps(k_), _ph(b), _ph(i)))))
            t.append((Fr(-3), mul(tr(_ps(j_), _ps(b)), tr(_ps(k_), _ph(i), _ph(b)))))
            for (x, y, s) in ((b, i, Fr(1, 2)), (i, b, Fr(-1, 2))):
                t.append((6 * s, mul(tr(_ps(j_), _ph(x)), tr(_ps(k_), _ps(b), _ph(y)))))
                t.append((6 * s, mul(tr(_ps(j_), _ph(x)), tr(_ps(b), _ps(k_), _ph(y)))))
        return add(*t)
    return dict(_sym(f, (j, k)))


@lru_cache(maxsize=None)
def r16_23(i, j, k, l, m):
    """(r^(0,3)_16)^{ij}_{klm} = -tr(phi^(i phi^j) psi_(k psi_l psi_m)) - 2 tr(phi^(i psi_(k phi^j) psi_l psi_m))."""
    t = []
    for (x, y) in ((i, j), (j, i)):
        for a, b, c in permutations((k, l, m)):
            t.append((Fr(-1, 12), tr(_ph(x), _ph(y), _ps(a), _ps(b), _ps(c))))
            t.append((Fr(-2, 12), tr(_ph(x), _ps(a), _ph(y), _ps(b), _ps(c))))
    return dict(add(*t))


# ---------------------------------------------------------------- fundamental relations R (A.2), as printed
@lru_cache(maxsize=None)
def R12_03(i, j, k):
    """(R^(0,1)_12)_{ijk} = eps_{(i|a1a2} eps_{|j|b1b2} u^{a1b1} v^{a2b2}_{|k)}."""
    def f(p):
        i_, j_, k_ = p
        return add(*[(eps(i_, a1, a2) * eps(j_, b1, b2), mul(u2(a1, b1), v3(a2, b2, k_)))
                     for a1 in R3 for a2 in R3 for b1 in R3 for b2 in R3 if eps(i_, a1, a2) * eps(j_, b1, b2)])
    return dict(_sym(f, (i, j, k)))


@lru_cache(maxsize=None)
def R12_11(i, j):
    """(R^(0,1)_12)^i_j = eps_{j a1 a2} (4 u^{a1 b} v^{i a2}_b + 3 u^{i a1 b} v^{a2}_b)."""
    t = []
    for a1 in R3:
        for a2 in R3:
            e = eps(j, a1, a2)
            if not e:
                continue
            for b in R3:
                t.append((4 * e, mul(u2(a1, b), v3(i, a2, b))))
                t.append((3 * e, mul(u3(i, a1, b), v2(a2, b))))
    return dict(add(*t))


@lru_cache(maxsize=None)
def R12_22(i, j, k, l):
    """(R^(0,1)_12)^{ij}_{kl} = eps_{a1 a2 (k} ( u^{a1 (i} v^{j) a2}_{l)} + u^{ij a1} v^{a2}_{l)} )."""
    def f(p):
        k_, l_ = p
        t = []
        for a1 in R3:
            for a2 in R3:
                e = eps(a1, a2, k_)
                if not e:
                    continue
                t.append((Fr(e, 2), mul(u2(a1, i), v3(j, a2, l_))))
                t.append((Fr(e, 2), mul(u2(a1, j), v3(i, a2, l_))))
                t.append((e, mul(u3(i, j, a1), v2(a2, l_))))
        return add(*t)
    return dict(_sym(f, (k, l)))


@lru_cache(maxsize=None)
def R14_13(i, j, k, l):
    """(R^(0,2)_14)^i_{jkl} = eps_{a1 a2 (j} v^{a1}_k v^{i a2}_{l)}."""
    def f(p):
        j_, k_, l_ = p
        return add(*[(eps(a1, a2, j_), mul(v2(a1, k_), v3(i, a2, l_))) for a1 in R3 for a2 in R3 if eps(a1, a2, j_)])
    return dict(_sym(f, (j, k, l)))


@lru_cache(maxsize=None)
def R16_01(i):
    """(R^(0,2)_16)_i = eps_{i a1 a2} (12 u^{bc} v^{a1}_b v^{a2}_c + 13 u^{a1 b} v^{a2}_c v^c_b + 12 v^{a1 b}_c v^{a2 c}_b)."""
    t = []
    for a1 in R3:
        for a2 in R3:
            e = eps(i, a1, a2)
            if not e:
                continue
            for b in R3:
                for c in R3:
                    t.append((12 * e, mul(u2(b, c), v2(a1, b), v2(a2, c))))
                    t.append((13 * e, mul(u2(a1, b), v2(a2, c), v2(c, b))))
                    t.append((12 * e, mul(v3(a1, b, c), v3(a2, c, b))))
    return dict(add(*t))


@lru_cache(maxsize=None)
def R16_12(i, j, k):
    """(R^(0,2)_16)^i_{jk} = eps_{a1 a2 (j} (3 u^{ib} v^{a1}_{k)} v^{a2}_b - 7 u^{i a1} v^b_{k)} v^{a2}_b
    + 6 u^{a1 b} v^i_{k)} v^{a2}_b + 24 v^{a1 b}_{k)} v^{i a2}_b)."""
    def f(p):
        j_, k_ = p
        t = []
        for a1 in R3:
            for a2 in R3:
                e = eps(a1, a2, j_)
                if not e:
                    continue
                for b in R3:
                    t.append((3 * e, mul(u2(i, b), v2(a1, k_), v2(a2, b))))
                    t.append((-7 * e, mul(u2(i, a1), v2(b, k_), v2(a2, b))))
                    t.append((6 * e, mul(u2(a1, b), v2(i, k_), v2(a2, b))))
                    t.append((24 * e, mul(v3(a1, b, k_), v3(i, a2, b))))
        return add(*t)
    return dict(_sym(f, (j, k)))


@lru_cache(maxsize=None)
def R16_23(i, j, k, l, m):
    """(R^(0,2)_16)^{ij}_{klm} = eps_{a1 a2 (k} (u^{a1 (i} v^{j)}_l v^{a2}_{m)} + 3 v^{a1 (i}_l v^{j) a2}_{m)})."""
    def f(p):
        k_, l_, m_ = p
        t = []
        for a1 in R3:
            for a2 in R3:
                e = eps(a1, a2, k_)
                if not e:
                    continue
                for (x, y) in ((i, j), (j, i)):
                    t.append((Fr(e, 2), mul(u2(a1, x), v2(y, l_), v2(a2, m_))))
                    t.append((Fr(3 * e, 2), mul(v3(a1, x, l_), v3(y, a2, m_))))
        return add(*t)
    return dict(_sym(f, (k, l, m)))


# ---------------------------------------------------------------- O^2_24 (2.6), the SU(2) threshold operator
def o2_24():
    """O^2_24 = eps^{pqr} v^m_q v^n_r tr(psi_(m psi_n psi_p)) (an SU(2) operator; letters as above)."""
    t = []
    for p_ in R3:
        for q in R3:
            for r in R3:
                e = eps(p_, q, r)
                if not e:
                    continue
                for m in R3:
                    for n in R3:
                        s3 = add(*[(Fr(1, 6), tr(_ps(a), _ps(b), _ps(c))) for a, b, c in permutations((m, n, p_))])
                        t.append((e, mul(v2(m, q), v2(n, r), s3)))
    return dict(add(*t))


# ---------------------------------------------------------------- traceless projection (reading 1)
@lru_cache(maxsize=None)
def projector(p, q):
    """Traceless projector on SU(3) tensors T^{i1..ip}_{j1..jq} symmetric in the uppers and in the lowers:
    {full index tuple I: [(coefficient, I'), ...]} with (P T)_I = sum c T_I'."""
    import sympy as sp
    V = list(product(R3, repeat=p + q))
    Vp = list(product(R3, repeat=p + q - 2))
    pos = {I: n for n, I in enumerate(V)}
    posp = {I: n for n, I in enumerate(Vp)}

    def symm(I):
        pu = list(set(permutations(I[:p])))
        pl = list(set(permutations(I[p:])))
        return [(Fr(1, len(pu) * len(pl)), tuple(a) + tuple(b)) for a in pu for b in pl]
    C = sp.zeros(len(Vp), len(V))               # trace map
    for I in V:
        if I[0] == I[p]:
            C[posp[I[1:p] + I[p + 1:]], pos[I]] += 1
    E = sp.zeros(len(V), len(Vp))               # symmetrised delta-embedding of lower-rank tensors
    for Ip in Vp:
        up, lo = Ip[:p - 1], Ip[p - 1:]
        for a in R3:
            for c, J in symm((a,) + up + (a,) + lo):
                E[pos[J], posp[Ip]] += sp.Rational(c.numerator, c.denominator)
    S = sp.zeros(len(V), len(V))                # symmetriser
    for I in V:
        for c, J in symm(I):
            S[pos[I], pos[J]] += sp.Rational(c.numerator, c.denominator)
    Pm = S - E * (C * E).pinv() * C * S
    out = {}
    for I in V:
        row = Pm.row(pos[I])
        out[I] = [(Fr(int(sp.fraction(x)[0]), int(sp.fraction(x)[1])), V[n]) for n, x in enumerate(row) if x != 0]
    return out


_RAW = {'u2': u2, 'u3': u3, 'v2': v2, 'v3': v3,
        'r12_03': r12_03, 'r12_11': r12_11, 'r12_22': r12_22, 'r14_13': r14_13,
        'r16_01': r16_01, 'r16_12': r16_12, 'r16_23': r16_23,
        'R12_03': R12_03, 'R12_11': R12_11, 'R12_22': R12_22, 'R14_13': R14_13,
        'R16_01': R16_01, 'R16_12': R16_12, 'R16_23': R16_23}
MIXED = {'12_11': (1, 1), '12_22': (2, 2), '14_13': (1, 3), '16_12': (1, 2), '16_23': (2, 3)}   # (uppers, lowers)
R_OF_r = {'r12_03': 'R12_03', 'r12_11': 'R12_11', 'r12_22': 'R12_22', 'r14_13': 'R14_13',
          'r16_01': 'R16_01', 'r16_12': 'R16_12', 'r16_23': 'R16_23'}


@lru_cache(maxsize=None)
def tensor(name, project=True):
    """Component function of a named tensor ('u2', 'v3', 'r14_13', 'R16_23', ...). With project=True the mixed r and R
    tensors are replaced by their traceless parts (reading 1); pure-upper or pure-lower ones are already traceless."""
    f = _RAW[name]
    key = name[1:]
    if not project or name[0] not in 'rR' or key not in MIXED:
        return f
    Pm = projector(*MIXED[key])

    @lru_cache(maxsize=None)
    def g(*I):
        return dict(add(*[(c, f(*J)) for c, J in Pm[tuple(I)]]))
    return g


# ---------------------------------------------------------------- (2.24) / (2.20)
# (coefficient c_a of (2.20), label s_a of (A.4), factors).  'E' = epsilon; i, j, k free, all other indices summed.
TERMS = [
    (Fr(24), 's1', [('E', 'a1 a2 i'), ('E', 'j b1 b2'), ('v2', 'c d'), ('v2', 'k a1'), ('v2', 'd b1'), ('r12_03', 'a2 b2 c')]),
    (Fr(-16, 15), 's3', [('E', 'a1 a2 i'), ('v2', 'j b'), ('v2', 'b c'), ('v2', 'c a1'), ('r12_11', 'k a2')]),
    (Fr(4, 15), 's4', [('E', 'a1 a2 a3'), ('v2', 'b a1'), ('v2', 'i a2'), ('v2', 'j a3'), ('r12_11', 'k b')]),
    (Fr(-32, 15), 's5', [('E', 'a1 a2 i'), ('v2', 'j b'), ('v2', 'k a1'), ('v2', 'b c'), ('r12_11', 'c a2')]),
    (Fr(16), 's9', [('E', 'a1 a2 i'), ('v2', 'j d'), ('v2', 'b a1'), ('v2', 'd c'), ('r12_22', 'k c a2 b')]),
    (Fr(85), 's10', [('E', 'a1 a2 i'), ('u2', 'b c'), ('v2', 'd a1'), ('v2', 'j b'), ('r14_13', 'k a2 c d')]),
    (Fr(5), 's11', [('E', 'a1 a2 i'), ('u2', 'b c'), ('v2', 'd b'), ('v2', 'j a1'), ('r14_13', 'k a2 c d')]),
    (Fr(-40), 's12', [('E', 'a1 a2 i'), ('u2', 'j b'), ('v2', 'c a1'), ('v2', 'd b'), ('r14_13', 'k a2 c d')]),
    (Fr(-5, 3), 's13', [('E', 'a1 a2 i'), ('u2', 'c d'), ('v2', 'b a1'), ('v2', 'j b'), ('r14_13', 'k a2 c d')]),
    (Fr(5, 3), 's14', [('E', 'a1 a2 i'), ('u2', 'j c'), ('v2', 'b a1'), ('v2', 'd b'), ('r14_13', 'k a2 c d')]),
    (Fr(-40), 's15', [('E', 'a1 a2 i'), ('u2', 'j b'), ('v2', 'c a1'), ('v2', 'k d'), ('r14_13', 'd a2 b c')]),
    (Fr(125, 7), 's17', [('E', 'a1 a2 i'), ('v3', 'j b a1'), ('v3', 'c d b'), ('r14_13', 'k a2 c d')]),
    (Fr(31), 's18', [('E', 'a1 a2 i'), ('v3', 'b c a1'), ('v3', 'd j b'), ('r14_13', 'k a2 c d')]),
    (Fr(24, 7), 's19', [('E', 'a1 a2 i'), ('v3', 'j k b'), ('v3', 'c d a1'), ('r14_13', 'b a2 c d')]),
    (Fr(543, 7), 's20', [('E', 'a1 a2 a3'), ('v3', 'b i a1'), ('v3', 'j c a2'), ('r14_13', 'k a3 b c')]),
    (Fr(8, 5), 's22', [('E', 'a1 a2 i'), ('v2', 'b a1'), ('v3', 'j k b'), ('r16_01', 'a2')]),     # reading 3
    (Fr(16, 15), 's23', [('E', 'a1 a2 i'), ('v2', 'j a1'), ('v3', 'k b a2'), ('r16_01', 'b')]),
    (Fr(251, 7), 's30', [('E', 'a1 a2 a3'), ('v2', 'b1 a1'), ('v3', 'b2 i a2'), ('r16_23', 'j k a3 b1 b2')]),
    (Fr(-1), 's31', [('E', 'a1 a2 a3'), ('v2', 'i a1'), ('v3', 'b1 b2 a2'), ('r16_23', 'j k a3 b1 b2')]),
    (Fr(-9), 's32', [('E', 'a1 a2 i'), ('v2', 'j a1'), ('v3', 'b c d'), ('r16_23', 'd k a2 b c')]),
    (Fr(174, 7), 's33', [('E', 'a1 a2 i'), ('v2', 'b a1'), ('v3', 'c j d'), ('r16_23', 'k d a2 b c')]),
    (Fr(-17, 2), 's34', [('E', 'a1 a2 i'), ('v2', 'b a1'), ('v3', 'c d a2'), ('r16_23', 'j k b c d')]),
]


def prefactor_parity(factors):
    """Fermion parity of the graviton prefactor f_a (each v carries one psi)."""
    return sum(1 for f, _ in factors if f in ('v2', 'v3')) % 2


def term_expr(factors, free, use_R=False, project=True):
    """f_a r_a (or f_a R_a with use_R) for one term, at the free index values `free` ({'i': .., 'j': .., 'k': ..})."""
    names = []
    for _, idx in factors:
        for x in idx.split():
            if x not in free and x not in names:
                names.append(x)
    out = defaultdict(int)
    for vals in product(R3, repeat=len(names)):
        env = dict(free)
        env.update(zip(names, vals))
        c, exprs = 1, []
        for f, idx in factors:
            ii = tuple(env[x] for x in idx.split())
            if f == 'E':
                c *= eps(*ii)
                if c == 0:
                    break
            else:
                exprs.append(tensor(R_OF_r[f] if (use_R and f in R_OF_r) else f, project)(*ii))
        if c == 0:
            continue
        for m, v in mul(*exprs).items():
            out[m] += c * v
            if out[m] == 0:
                del out[m]
    return dict(out)


def o330(ijk=(0, 0, 0), project=True, graded_sign=True, terms=None):
    """(O^3_30)^{ijk} of (2.24) with the readings of the module docstring (both flags True: the Q-closed operator).
    project=False / graded_sign=False give the operator as printed. Returns {multitrace: Fraction}."""
    free = dict(zip('ijk', ijk))
    tot = defaultdict(int)
    for coef, lab, factors in TERMS:
        if terms is not None and lab not in terms:
            continue
        s = -1 if (graded_sign and prefactor_parity(factors)) else 1
        for m, v in term_expr(factors, free, project=project).items():
            tot[m] += s * coef * v
            if tot[m] == 0:
                del tot[m]
    return dict(tot)


def relation_2_20(ijk=(0, 0, 0), project=True):
    """sum_a c_a f_a R_a of (2.20); vanishes identically (formally) with project=True."""
    free = dict(zip('ijk', ijk))
    tot = defaultdict(int)
    for coef, lab, factors in TERMS:
        for m, v in term_expr(factors, free, use_R=True, project=project).items():
            tot[m] += coef * v
            if tot[m] == 0:
                del tot[m]
    return dict(tot)


# ---------------------------------------------------------------- exact I/O
def to_modp(expr, p=None):
    from .modp import P
    p = p or P
    out = {}
    for m, v in expr.items():
        v = Fr(v)
        x = (v.numerator * pow(v.denominator, p - 2, p)) % p
        if x:
            out[m] = x
    return out


def save_rational_json(expr, path, **meta):
    import json
    with open(path, 'w') as f:
        json.dump(dict(meta, terms=[[str(Fr(c)), [list(w) for w in m]] for m, c in sorted(expr.items())]), f)


def load_rational_json(path):
    import json
    d = json.load(open(path))
    return {tuple(tuple(w) for w in m): Fr(c) for c, m in d['terms']}
