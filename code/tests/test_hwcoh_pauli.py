"""Regression: fermion degree above 3 dim g (SU(2): d > 9). Words with more fermions than Grassmann components vanish
identically; both engines must skip those degrees instead of dividing by C(k, d) = 0 (T-027, 2026-09-29)."""
import os
from fortuity.hwcoh.dims import partition_dims_nf
from fortuity.hwcoh.sector import Sector
from fortuity.hwcoh.sectorpar import SectorPar
from fortuity.hwcoh.gravhw import GravitonAlgebra


def test_su2_degree_beyond_grassmann_bound():
    dims = partition_dims_nf(2, 40)
    GA = GravitonAlgebra(2)
    out = []
    for Eng in (Sector, SectorPar):
        S = Eng(2, (8, 8, 4), 10, dims, log=lambda s: None, graviton_algebra=GA, maxlen=3, maxlen_up=3)
        r = S.run()
        assert r['check'] and r['m'] == 4 and r['m_d'].get('10', 0) == 0, r
        out.append((r['m'], r['r_up'], r['grav']))
    assert out[0] == out[1], out
