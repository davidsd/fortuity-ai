"""Full-letter cohomology for SO(2N+1) and USp(2N) (T-048, from the full-letter SO/USp development code): fast content
enumeration, exact group dimensions, the group-aware hw sector, the SU(3) Weyl action, and the large-N graviton
generators. The level-18 Table 1 reproduction (ChangLin2025SDuality) is a ~2-minute run, not a unit test:
python -m fortuity.full.levels scan SO_odd 3 18 OUT pairs_18.json, then `weight OUT 1,1 6,6,6 8` and `0,0 6,6,6 8`."""
import random
from fortuity.full.words import contents_of_sector, contents_of_sector_direct, level_contents, Q_multi
from fortuity.full import gdims, dims as sudims
from fortuity.full.sector import Sector
from fortuity.full.weyl import PERMS, weyl_expr, weyl_multi
from fortuity.full.wordenum import multitraces_with_content
from fortuity.full import gravitons


def test_fast_contents_equal_direct():
    n = 0
    for j in range(1, 9):
        for sec in level_contents(j):
            assert contents_of_sector(sec) == contents_of_sector_direct(sec), sec
            n += 1
    assert n > 100
    assert contents_of_sector((1, 1, 6, 6, 6), 7) == contents_of_sector_direct((1, 1, 6, 6, 6), 7)


def test_group_dims():
    for sec in [(2, 2, 4, 4, 4), (0, 1, 3, 3, 2)]:
        for c in contents_of_sector(sec)[:6]:
            assert gdims.content_dim('SU', 3, c) == sudims.content_dim(3, c)
    bmn = lambda g, n: sum(gdims.content_dim(g, n, c) for c in contents_of_sector((1, 1, 6, 6, 6))
                           if all(L[0] + L[1] == 0 for L in c))
    assert (bmn('SO_odd', 3), bmn('Sp', 3)) == (903, 903)          # R-005, BMN part of (1;3,3,3)


def test_group_sector_matches_scan():
    # two rows of the stored level-18 scans (scripts/full-letter/so7-usp6/runs/{SO_odd,Sp}_j18.jsonl)
    r = Sector(3, (-5, 7, 6, 6, 6), 5, 10, log=lambda s: None, group='SO_odd').run()
    assert (r['m'], r['r_up'], r['check']) == (1, 1, True)
    for g in ('SO_odd', 'Sp'):
        r = Sector(3, (0, 0, 4, 4, 4), 6, 8, log=lambda s: None, group=g).run()
        assert r['check'] and r['group'] == g


def test_weyl_commutes_with_Q():
    random.seed(3)
    for sec in [(1, 1, 6, 6, 6), (2, 2, 4, 4, 4), (1, 1, 6, 4, 2)]:
        ws = [w for c in contents_of_sector(sec) if len(c) <= 5 for w in multitraces_with_content(c)]
        for w in random.sample(ws, min(6, len(ws))):
            for perm in PERMS:
                s, wm = weyl_multi(w, perm)
                rhs = {m: s * c for m, c in Q_multi(wm).items() if c} if s else {}
                assert weyl_expr(Q_multi(w), perm) == {m: c for m, c in rhs.items() if c}


def test_graviton_generators_and_counts():
    # large-N single-trace cohomology: u_2, v_2 (two components), w_2 absent at (2,2;2,2,2) Y=1, u_6 at (0,0;4,4,4),
    # tr(lam+ lam-) at (0,0;2,2,2) Y=2
    secs = [((0, 0, 4, 0, 0), 2), ((1, 1, 2, 2, 2), 2), ((2, 2, 2, 2, 2), 1), ((0, 0, 4, 4, 4), 6), ((0, 0, 2, 2, 2), 2)]
    assert [gravitons.single_trace_cohomology(s, Y)[1].shape[0] for s, Y in secs] == [1, 2, 0, 1, 1]
    gens = gravitons.generators_up_to_level(12, 6, workers=1)
    assert len(gens) == 680
    for g in ('SO_odd', 'Sp'):
        r = gravitons.graviton_count(g, 3, (0, 0, 4, 4, 4), 6, gens, log=None, maxlen=8, cartan=True)
        assert (r['dim'], r['exact'], r['h'], r['graviton'], r['coulomb_graviton']) == (29, 17, 12, 12, 12)
