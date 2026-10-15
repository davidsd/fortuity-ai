"""Coulomb-branch graviton counts for SO(2N+1), USp(2N), G2 (fortuity.bmn.gravindex group=...) and G2 Cartan letters
(fortuity.hwcoh.evalg), added for T-034."""
import numpy as np
from fortuity.bmn.gravindex import GravitonCounter
from fortuity.bmn.modp import P


def _dims(group, N, j):
    return {k: v for k, v in GravitonCounter(N, j, group=group, verbose=False).run().items() if v}


def test_so3_equals_su2():
    assert _dims('SO_odd', 1, 16) == _dims('SU', 2, 16)


def _direct_coulomb_dim(group, N, sec, nmax, seed=3):
    """Rank of all graviton monomials of BMN sector sec = (2J, q1, q2, q3) evaluated on Cartan-valued letters, built
    with the explicit matrices of fortuity.hwcoh.evalg (independent of gravindex's eigenvalue-slot model)."""
    import math
    from fortuity.bmn.words import graviton_generators, mul_expr, PHI, PSI, F
    from fortuity.bmn.modp import RREFBasis
    from fortuity.hwcoh.evalg import Frame
    ch = {PHI[0]: (0, 1, 0, 0), PHI[1]: (0, 0, 1, 0), PHI[2]: (0, 0, 0, 1),
          PSI[0]: (1, 0, 1, 1), PSI[1]: (1, 1, 0, 1), PSI[2]: (1, 1, 1, 0), F: (2, 1, 1, 1)}

    def secof(e):
        m = next(iter(e))
        return tuple(sum(ch[l][i] for w in m for l in w) for i in range(4))
    if not any(sec):
        return 1                                          # the identity operator
    gens = [e for _, e in graviton_generators(nmax, 1, qmax=sec) if e]
    gs = [secof(e) for e in gens]
    monos = []

    def rec(start, rem, chosen):
        if not any(rem):
            monos.append(chosen); return
        for i in range(start, len(gens)):
            if all(a <= b for a, b in zip(gs[i], rem)):
                rec(i + 1 if gs[i][0] % 2 else i, tuple(b - a for a, b in zip(gs[i], rem)), chosen + [i])
    rec(0, tuple(sec), [])
    exprs = []
    for mo in monos:
        e = gens[mo[0]]
        for gi in mo[1:]:
            e = mul_expr(e, gens[gi])
        e = {m: c for m, c in e.items() if c}
        if e:
            exprs.append(e)
    if not exprs:
        return 0
    words = sorted({m for e in exprs for m in e})
    index = {m: i for i, m in enumerate(words)}
    degs = list(range(sec[0] % 2, sec[0] + 1, 2))
    ranks = []
    for factor in (1, 2):                                 # point doubling: the rank must be stable
        k = max(degs) + 2
        cpp = sum(math.comb(k, d) for d in degs)
        fr = Frame(N, math.ceil(factor * (2 * len(exprs) + 8) / cpp), k, degs, seed=seed + factor, group=group, cartan=True)
        M = fr.eval_words(words, dtype=np.int64) % P
        rows = np.zeros((len(exprs), M.shape[1]), dtype=np.int64)
        for r, e in enumerate(exprs):
            for m, c in e.items():
                rows[r] = (rows[r] + (c % P) * M[index[m]]) % P
        b = RREFBasis(M.shape[1]); b.add_batched(rows)
        ranks.append(b.rank)
    assert ranks[0] == ranks[1]
    return ranks[1]


def test_coulomb_counts_against_direct_cartan_matrices():
    """gravindex's Coulomb counts for SO(5) and USp(4) equal the rank of the graviton monomials on explicit Cartan-valued
    so(5) / usp(4) matrices, sector by sector (two different matrix realisations, both independent of gravindex)."""
    for group in ('SO_odd', 'Sp'):
        d = _dims(group, 2, 12)
        for sec in sorted(s for s in d if 2 * sum(s[1:]) <= 12):
            assert _direct_coulomb_dim(group, 2, sec, 4) == d[sec], (group, sec)


def test_g2_cartan_basis():
    from fortuity.hwcoh.evalg import _g2_cartan_basis
    from fortuity.bmn.evaluate import g2_phi
    B = _g2_cartan_basis(P)
    assert B.shape == (2, 7, 7)
    assert not ((B + np.transpose(B, (0, 2, 1))) % P).any()               # in so(7)
    assert not ((B[0] @ B[1] - B[1] @ B[0]) % P).any()                    # abelian
    phi = g2_phi()
    for X in B:                                                           # annihilates the 3-form: in g2
        d = (np.einsum('da,dbc->abc', X, phi) + np.einsum('db,adc->abc', X, phi) + np.einsum('dc,abd->abc', X, phi)) % P
        assert not d.any()
