"""Tests for fortuity.full (full 1/16-BPS letters, hw complex, exact dims, one-loop Q_1, block restriction).
Promoted from scripts/full-letter/su-checks (T-016)."""
import random
from collections import defaultdict
from itertools import product

import numpy as np

from fortuity.bmn.modp import P, matmul_mod
from fortuity.full.letters import Q_letter, is_ferm, letters_up_to
from fortuity.full.words import (Q_multi, Q_expr, add_term, canon_word, sectors_at_level, contents_of_sector,
                                 words_of_sector as words_slow)
from fortuity.full.wordenum import words_of_sector
from fortuity.full.evalpoints import PointsFull
from fortuity.full.evalg import Frame, nferm
from fortuity.full.cohom import FullDirect
from fortuity.full.dims import content_dim, sector_dim, _sym_ext
from fortuity.full.hw import E_multi
from fortuity.full.sector import Sector
from fortuity.full.q1 import pair_rule, Q1_multi, Q1_expr
from fortuity.full.restrict import restrict_monomial, PHI

# Chang-Lin (arXiv:2209.06728) Table 3: SU(2) Q-cohomology dimension at level n, all letters
CHANG_LIN_SU2 = {4: 6, 5: 6, 6: 9, 7: 18, 8: 30}


def _bmn(w):
    return all(L[0] == 0 and L[1] == 0 for x in w for L in x)


def _vec(expr, N, seed=1, npts=6):
    letters = {L for m in expr for w in m for L in w}
    degs = sorted({sum(1 for w in m for L in w if is_ferm(L)) for m in expr})
    pts = PointsFull(N, npts, max(max(degs), 1), degs, letters, seed=seed)
    return pts.expr_vector(expr)


def test_q0_on_letters():
    assert Q_letter((0, 0, (0,))) == ()                 # Q phi = 0
    assert Q_letter((1, 0, ())) == ()                   # Q lambda_alpha = 0
    assert len(Q_letter((0, 0, (0, 1)))) == 2           # Q psi = commutator of two scalars
    assert len(Q_letter((0, 0, (0, 1, 2)))) == 6        # Q f = sum_m [phi_m, psi_m]
    assert len(Q_letter((1, 0, (0,)))) == 2             # [Q, D] = [lambda, .}


def test_q0_nilpotent():
    rng = np.random.default_rng(0)
    alphabet = letters_up_to(2, 2)
    tested = 0
    for N in (2, 3):
        for _ in range(40):
            w = tuple(alphabet[i] for i in rng.integers(0, len(alphabet), int(rng.integers(2, 5))))
            w2 = tuple(alphabet[i] for i in rng.integers(0, len(alphabet), int(rng.integers(2, 4))))
            s, cw = canon_word(w)
            s2, cw2 = canon_word(w2)
            if not (s and s2):
                continue
            e = defaultdict(int)
            add_term(e, (cw, cw2), 1)
            if not e:
                continue
            qq = Q_expr(Q_expr(e))
            tested += 1
            assert not (qq and np.any(_vec(qq, N)))
    assert tested > 30


def test_fast_enumeration_equals_slow():
    for n in (6, 8):
        for s in sectors_at_level(n):
            assert sorted(words_slow(s, 4)) == sorted(words_of_sector(s, 4))


def test_su2_cohomology_matches_chang_lin():
    D = FullDirect(N=2, maxlen=3)
    for n in (4, 5, 6, 7):
        assert sum(D.dimH(s) for s in sectors_at_level(n)) == CHANG_LIN_SU2[n]


def test_exact_dims():
    # invariants in Sym^n(adj) and Lambda^n(adj) of SU(2); the grid must not alias (regression: M too small)
    for M in (None,):
        w, h, e = _sym_ext(2, 2 * 6 + 5, 6)
        assert [round((w * h[n]).sum().real) for n in range(7)] == [1, 0, 1, 0, 1, 0, 1]
        assert [round((w * e[n]).sum().real) for n in range(4)] == [1, 0, 0, 1]
    assert content_dim(2, ((2, 0, (0,)),)) == 0                       # a single traceless letter
    assert content_dim(3, ((0, 0, (0,)),) * 3) == 1                   # tr phi^3 for SU(3)
    D = FullDirect(N=2, maxlen=3)
    for n in (4, 6, 7):
        for s in sectors_at_level(n):
            assert sector_dim(2, s) == D.dimV(s), s


def test_raising_operators_commute_with_q0():
    random.seed(1)
    nt = 0
    for n in (8, 9):
        for s in sectors_at_level(n):
            ws = words_of_sector(s, 4)
            for w in random.sample(ws, min(3, len(ws))):
                for i in (1, 2):
                    a = defaultdict(int)
                    for m, c in Q_multi(w).items():
                        for m2, c2 in E_multi(m, i).items():
                            a[m2] += c * c2
                    a = {m: c for m, c in a.items() if c}
                    assert a == dict(Q_expr(E_multi(w, i)))
                    nt += 1
    assert nt > 200


def test_hw_complex_su2_matches_chang_lin():
    def dim_irrep(lam2):
        p, q = (lam2[0] - lam2[1]) // 2, (lam2[1] - lam2[2]) // 2
        return (p + 1) * (q + 1) * (p + q + 2) // 2
    for n in (4, 5, 6, 7):
        r_up, m = {}, {}
        for s in [s for s in sectors_at_level(n) if s[2] >= s[3] >= s[4]]:
            for Y in sorted({len(c) for c in contents_of_sector(s)}):
                res = Sector(2, s, Y, 3, log=lambda *a: None).run()
                assert res['check'], (s, Y)
                r_up[(s, Y)], m[(s, Y)] = res['r_up'], res['m']
        total = 0
        for (s, Y), mm in m.items():
            low = (s[0] - 1, s[1] - 1) + s[2:]
            h = mm - r_up.get((low, Y + 1), 0) - r_up[(s, Y)]
            assert h >= 0
            total += h * dim_irrep(s[2:])
        assert total == CHANG_LIN_SU2[n]


def test_q1_pair_rules():
    f, lam_free = (0, 1, 2), ()
    # Q_1 needs all three thetas between the two letters
    for S1 in [(), (0,), (1, 2), (0, 1, 2)]:
        for S2 in [(), (0,), (1,), (0, 2), (0, 1, 2)]:
            assert bool(pair_rule(S1, S2)) == (set(S1) | set(S2) == {0, 1, 2})
    # Q_1(psi_I phi^J) = delta_I^J d c d c: psi_1 = theta_1 theta_2 component
    assert {(a, b) for _, a, b in pair_rule((1, 2), (0,))} == {((1, 0, ()), (0, 1, ())), ((0, 1, ()), (1, 0, ()))}
    assert pair_rule((1, 2), (1,)) == ()
    assert len(pair_rule(f, f)) == 16


def test_q1_anticommutes_with_q0():
    random.seed(3)
    for N in (2, 3):
        nt = 0
        for n in (8, 10):
            for s in sectors_at_level(n):
                ws = [w for w in words_of_sector(s, 4) if _bmn(w)]
                for w in random.sample(ws, min(2, len(ws))):
                    tot = defaultdict(int)
                    for m, c in Q_expr(Q1_multi(w, N)).items():
                        tot[m] += c
                    for m, c in Q1_expr(Q_multi(w), N).items():
                        tot[m] += c
                    assert not {m: c for m, c in tot.items() if c}, w
                    nt += 1
        assert nt > 60


def test_exact_words_vanish_on_cartan():
    random.seed(5)
    ws = [w for s in sectors_at_level(10) for w in words_of_sector(s, 4)]
    sample = random.sample(ws, 30)
    qw = sorted({m for w in sample for m in Q_multi(w)})
    letters = sorted({L for w in qw for n in w for L in n})
    for d in sorted({nferm(w) for w in qw}):
        wd = [w for w in qw if nferm(w) == d]
        fr = Frame(3, 4, max(d + 2, 4) if d else 0, [d], letters, seed=9 + d, cartan=True)
        X = fr.eval_words(wd).astype(np.int64)
        idx = {w: i for i, w in enumerate(wd)}
        for w in sample:
            q = {m: c for m, c in Q_multi(w).items() if nferm(m) == d}
            if q:
                rows = np.array([idx[m] for m in q])
                cs = np.array(list(q.values()), dtype=np.int64) % P
                assert not np.any(matmul_mod(cs[None, :], X[rows]))


def test_block_restriction_is_chain_map():
    random.seed(7)
    nt = 0
    for s in sectors_at_level(8):
        ws = [w for w in words_of_sector(s, 4) if _bmn(w)]
        for w in random.sample(ws, min(2, len(ws))):
            for t in ((1, 0, 0), (2, 0, 0), (1, 1, 0)):
                a = defaultdict(int)
                for m, c in Q_multi(w).items():
                    for m2, c2 in restrict_monomial(m, 3, t).items():
                        a[m2] += c * c2
                assert {m: c for m, c in a.items() if c} == dict(Q_expr(restrict_monomial(w, 3, t)))
                nt += 1
    assert nt > 50


def test_block_restriction_matches_block_diagonal_evaluation():
    random.seed(11)
    rng = np.random.default_rng(1)
    npts, done = 3, 0

    def powm(a, k):
        r = np.ones_like(a)
        for _ in range(k):
            r = r * a % P
        return r

    for s in sectors_at_level(8) + sectors_at_level(10):
        ws = [w for w in words_of_sector(s, 4) if _bmn(w) and nferm(w) == 0]
        if not ws:
            continue
        w = random.choice(ws)
        letters = sorted({L for x in w for L in x})
        A = {L: rng.integers(0, P, (npts, 2, 2)) for L in letters}
        for L in letters:
            A[L][:, 1, 1] = (-A[L][:, 0, 0]) % P
        x = {L: (rng.integers(0, P, npts) if L in PHI else np.zeros(npts, dtype=np.int64)) for L in letters}

        def M3(L):
            M = np.zeros((npts, 3, 3), dtype=np.int64)
            M[:, :2, :2] = A[L]
            for i in range(2):
                M[:, i, i] = (M[:, i, i] + x[L]) % P
            M[:, 2, 2] = (-2 * x[L]) % P
            return M

        def trw(word, mats):
            val = np.ones(npts, dtype=np.int64)
            for neck in word:
                Mx = None
                for L in neck:
                    Mx = mats(L) if Mx is None else np.einsum('pij,pjk->pik', Mx, mats(L)) % P
                val = val * np.trace(Mx, axis1=1, axis2=2) % P
            return val

        full = trw(w, M3)
        tot = np.zeros(npts, dtype=np.int64)
        nphi = [sum(1 for neck in w for L in neck if L == (0, 0, (m,))) for m in range(3)]
        for t in product(*[range(k + 1) for k in nphi]):
            mono = np.ones(npts, dtype=np.int64)
            for m in range(3):
                if t[m]:
                    mono = mono * powm(x[(0, 0, (m,))], t[m]) % P
            for m2, c in restrict_monomial(w, 3, t).items():
                tot = (tot + c * mono * trw(m2, lambda L: A[L])) % P
        assert np.all(tot == full), w
        done += 1
    assert done >= 5
