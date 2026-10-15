"""Guards against vacuous certificates and wrong defaults (review B.3, B.4; added 2026-10-02)."""
import pytest
from fortuity.hwcoh.dims import partition_dims_nf
from fortuity.hwcoh.sector import Sector
from fortuity.hwcoh.gravhw import GravitonAlgebra


def test_dims_table_must_reach_the_sector_level():
    d14 = partition_dims_nf(3, 14)
    with pytest.raises(ValueError):
        Sector(3, (4, 4, 2), 3, d14, log=lambda s: None, graviton_algebra=GravitonAlgebra(3))


def test_maxlen_dropping_a_whole_fermion_degree_fails():
    # SU(3) (5,5,0) 2J=5 with MAXLEN=4: no words of fermion degree 5 although its exact multiplicity is 1; this used to
    # give m=0 and check=True
    d20 = partition_dims_nf(3, 20)
    with pytest.raises(RuntimeError):
        Sector(3, (5, 5, 0), 5, d20, log=lambda s: None, graviton_algebra=GravitonAlgebra(3), maxlen=4).run()


def test_default_generator_degree_is_the_top_casimir_degree():
    from fortuity.bmn.symdirect import SymDirect
    assert SymDirect(3, verbose=False).gen_nmax == 3
    assert SymDirect(3, verbose=False, group='SO_odd').gen_nmax == 6
    assert SymDirect(2, verbose=False, group='Sp').gen_nmax == 4
    assert SymDirect(0, verbose=False, group='G2').gen_nmax == 6
