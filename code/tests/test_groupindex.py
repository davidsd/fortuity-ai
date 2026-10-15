"""Tests for the general-gauge-group BMN index (fortuity.bmn.groupindex); a few seconds."""
import numpy as np
from fortuity.bmn.groupindex import bmn_index, unrefined_series, group_dim
from fortuity.bmn.fullindex import full_index, su3_multiplicities


def test_dimensions():
    assert [group_dim('SU', n) for n in (2, 3)] == [3, 8]
    assert [group_dim('SO_odd', n) for n in (1, 2, 3)] == [3, 10, 21]
    assert [group_dim('Sp', n) for n in (1, 2, 3)] == [3, 10, 21]


def test_su_matches_fullindex():
    for N, J in ((2, 20), (3, 14)):
        Z1, A1 = bmn_index('SU', N, J)
        Z2, A2 = full_index(N, J, verbose=False)
        assert A1 == A2 and np.array_equal(Z1, Z2)


def test_isomorphic_algebras():
    # so(3) = sp(1) = su(2), so(5) = sp(2): identical BMN indices
    assert unrefined_series('SO_odd', 1, 24) == unrefined_series('Sp', 1, 24) == unrefined_series('SU', 2, 24)
    assert unrefined_series('SO_odd', 2, 20) == unrefined_series('Sp', 2, 20)


def test_so7_sp3_difference():
    # arXiv:2506.13887 eq. (4.12): I_SO(7) - I_Sp(3) = -t^18 + 3 t^20 - 3 t^22 + 11 t^24 - ...
    so7, sp3 = unrefined_series('SO_odd', 3, 24), unrefined_series('Sp', 3, 24)
    assert [a - b for a, b in zip(so7, sp3)][18:] == [-1, 0, 3, 0, -3, 0, 11]
    # common graviton index below j = 18 (their eq. (4.13)); Sp(3) still equals it at j = 18
    grav = [1, 0, 0, 0, 6, 0, -8, 0, 39, 0, -72, 0, 230, 0, -504, 0, 1203, 0, -2653]
    assert sp3[:19] == grav and so7[:18] == grav[:18] and so7[18] == -2654


def test_so7_excess_is_singlet():
    Zs, A = bmn_index('SO_odd', 3, 18)
    Zp, _ = bmn_index('Sp', 3, 18)
    D = Zs - Zp
    assert not D[:18].any()
    assert su3_multiplicities(D[18], A) == {(0, 0): -1}


def test_points_lie_algebra_conditions():
    # evaluate.Points letters lie in so(2N+1) / sp(2N) (over F_p)
    from fortuity.bmn.evaluate import Points
    from fortuity.bmn.modp import P
    for g, N, n in (('SO_odd', 3, 7), ('Sp', 3, 6)):
        pts = Points(N, 2, 3, [0, 1, 2], seed=1, group=g)
        Om = np.zeros((n, n), dtype=np.int64)
        if g == 'Sp':
            Om[:N, N:] = np.eye(N, dtype=np.int64); Om[N:, :N] = -np.eye(N, dtype=np.int64)
        for l, d in pts.letter.items():
            for M in d.values():
                M = M.reshape(-1, n, n)
                assert M.shape[-1] == n
                if g == 'SO_odd':
                    assert not ((M + np.swapaxes(M, -1, -2)) % P).any()
                else:
                    lhs = (np.einsum('ij,cjk->cik', Om, M) + np.einsum('cji,jk->cik', M, Om)) % P
                    assert not lhs.any()


def test_odd_traces_vanish_so_sp():
    # Tr of an odd product of so/sp matrices, symmetrised over orderings, vanishes: u_3 = 0
    from fortuity.bmn.evaluate import Points
    from fortuity.bmn.words import graviton_generators
    u3 = [e for name, e in graviton_generators(3, 1) if name[0] == 'u' and name[1] == 3]
    for g in ('SO_odd', 'Sp'):
        pts = Points(3, 3, 3, [0], seed=2, group=g)
        for e in u3:
            assert not pts.expr_vector(e).any()


import os
import pytest


@pytest.mark.skipif(not os.environ.get('FORTUITY_SLOW'), reason='~3 min; set FORTUITY_SLOW=1 to run')
def test_hwcoh_so7_usp6_singlet_column():
    # hw-complex port (group option): SO(7) has one non-graviton singlet at (1;3,3,3), USp(6) none (R-005)
    from fortuity.hwcoh.dims import partition_dims_nf
    from fortuity.hwcoh.sector import Sector
    from fortuity.hwcoh.gravhw import GravitonAlgebra
    GA = GravitonAlgebra(8)
    h = {}
    for g in ('SO_odd', 'Sp'):
        dims = partition_dims_nf(3, 18, group=g)
        res = {n: Sector(3, (3, 3, 3), n, dims, log=lambda s: None, graviton_algebra=GA, group=g).run()
               for n in (0, 1)}
        assert all(r['check'] for r in res.values())
        h[g] = res[1]['m'] - res[0]['r_up'] - res[1]['r_up'] - res[1]['grav']
    assert h == {'SO_odd': 1, 'Sp': 0}


def test_g2_index_low_orders():
    # G2 (Casimirs of degree 2 and 6): same as SO(7) through j=6; at j=8, 21 (u_2 u_2) + 3 (w_2) = 24, while SO(7)
    # has 15 more from the independent quartic single trace u_4 ([4,0])
    g2, so7 = unrefined_series('G2', 0, 8), unrefined_series('SO_odd', 3, 8)
    assert group_dim('G2', 0) == 14
    assert g2[:7] == so7[:7] == [1, 0, 0, 0, 6, 0, -8]
    assert (g2[8], so7[8]) == (24, 39)


def test_g2_letters():
    # G2 letters: antisymmetric 7x7 matrices annihilating the octonion 3-form, spanning a 14-dimensional space
    from fortuity.bmn.evaluate import g2_phi, project_g2
    from fortuity.bmn.modp import P, rank_mod
    phi = g2_phi()
    assert np.array_equal(np.einsum('acd,bcd->ab', phi, phi), 6 * np.eye(7, dtype=np.int64))
    X = project_g2(np.random.default_rng(1).integers(0, P, (40, 7, 7)), P)
    act = (np.einsum('nad,dbc->nabc', X, phi) + np.einsum('nbd,adc->nabc', X, phi)
           + np.einsum('ncd,abd->nabc', X, phi)) % P
    assert not ((X + np.swapaxes(X, 1, 2)) % P).any() and not act.any()
    assert rank_mod(X.reshape(40, 49)) == 14


def test_hw_cache_roundtrip(tmp_path, monkeypatch):
    # hwcoh.sector._hw_cached: cached hw vectors are identical to freshly computed ones, and a cache with a different
    # word list is ignored
    from fortuity.hwcoh import wordenum as enum
    from fortuity.hwcoh.sector import _hw_cached
    from fortuity.hwcoh.hw import hw_vectors
    monkeypatch.setenv('HW_CACHE', str(tmp_path))
    ws = enum.words_of_sector((1, 3, 3, 3), None)
    a = _hw_cached((1, 3, 3, 3), None, ws, 1)
    b = _hw_cached((1, 3, 3, 3), None, ws, 1)
    ref = hw_vectors(ws)
    for x, y, z in zip(a, b, ref):
        assert np.array_equal(x[0], z[0]) and np.array_equal(x[1], z[1]) and np.array_equal(y[1], z[1])
    c = _hw_cached((1, 3, 3, 3), None, ws[:-1], 1)       # different words: must not reuse the cache
    assert sum(H.shape[0] for _, H in c) == sum(H.shape[0] for _, H in hw_vectors(ws[:-1]))


def test_restrict_so7_to_g2_singlet():
    # hwcoh.restrict: the SO(7) non-graviton class at (1;3,3,3) maps to zero in G2's non-graviton classes (R-010),
    # and restricting SO(7) to itself keeps it (positive control)
    from fortuity.hwcoh.dims import partition_dims_nf
    from fortuity.hwcoh.gravhw import GravitonAlgebra
    from fortuity.hwcoh.restrict import restrict_sector
    dso = partition_dims_nf(3, 18, group='SO_odd', twoJ_max=3)
    dg = partition_dims_nf(0, 18, group='G2', twoJ_max=3)
    GA = GravitonAlgebra(8)
    r = restrict_sector((3, 3, 3), 1, (3, 'SO_odd', dso), (0, 'G2', dg), GA)
    assert (r['bh_G'], r['bh_G_to_H'], r['bh_H'], r['closed_in_H']) == (1, 0, 0, True)
    r = restrict_sector((3, 3, 3), 1, (3, 'SO_odd', dso), (3, 'SO_odd', dso), GA)
    assert (r['bh_G'], r['bh_G_to_H']) == (1, 1)
