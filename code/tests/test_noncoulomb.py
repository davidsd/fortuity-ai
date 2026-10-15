"""fortuity.hwcoh.noncoulomb (R-015): known sectors, and agreement of the direct Cartan rank with the independent
eigenvalue-meson count of fortuity.bmn.gravindex."""
from fortuity.hwcoh.gravhw import GravitonAlgebra
from fortuity.hwcoh.noncoulomb import cartan_rank, coulomb_hw_counts, sector_counts


def test_known_su3_sectors():
    GA = GravitonAlgebra(3)
    # (3;4,4,4), j = 24: one graviton class, the non-Coulomb R_24 of R-009/R-015
    r = sector_counts(GA, 'SU', 3, (3, 4, 4, 4), grav=1)
    assert (r['coulomb'], r['noncoulomb']) == (0, 1)
    # (0;4,2,0), j = 12: one graviton class, Coulomb-type
    r = sector_counts(GA, 'SU', 3, (0, 4, 2, 0), grav=1)
    assert (r['coulomb'], r['noncoulomb']) == (1, 0)


def test_two_routes_agree_su3():
    jmax = 14
    GA = GravitonAlgebra(3)
    counts = coulomb_hw_counts('SU', 3, jmax)
    assert counts, 'no Coulomb graviton states found'
    for (n, lam), c in sorted(counts.items()):
        sec = (n,) + lam
        blocks = GA.hw_blocks(sec)
        nrows = sum(H.shape[0] for _, H in blocks)
        assert cartan_rank(GA, blocks, 'SU', 3, sec, nrows) == c, (sec, c)
