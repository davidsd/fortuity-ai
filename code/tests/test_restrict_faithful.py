"""Regression test for the faithfulness of hwcoh.restrict._lower_q_rows (review, Part B.1; fixed 2026-10-01).
SU(3) (4,4,2) at 2J=3, j=20: the certified rank of Q from 2J=3 to 2J=2 is 89 (engine r_up of sector 2J=2); the old
_lower_q_rows gave 88 for every seed, so classreps.hw_nongraviton_reps reported a fake non-graviton class (h=4,
nongrav=1, a non-closed 482-term representative). The engine says h=3, all gravitons."""
from fortuity.hwcoh.dims import partition_dims_nf
from fortuity.hwcoh.sector import Sector
from fortuity.hwcoh.restrict import _lower_q_rows, certified_lower_rank
from fortuity.hwcoh.classreps import hw_nongraviton_reps
from fortuity.hwcoh.gravhw import GravitonAlgebra
from fortuity.hwcoh import ffla


def test_lower_q_rows_faithful_su3_442():
    dims = partition_dims_nf(3, 20)
    GA = GravitonAlgebra(3)
    r = certified_lower_rank(3, (4, 4, 2), 3, dims, GA, 'SU')
    assert r == 89
    S = Sector(3, (4, 4, 2), 3, dims, log=lambda s: None)
    S.prepare()
    assert ffla.rank(_lower_q_rows(S, dims)) == 89                  # stabilized, without the certificate
    assert ffla.rank(_lower_q_rows(S, dims, r_cert=r)) == 89
    _, info = hw_nongraviton_reps(3, 'SU', (4, 4, 2), 3, dims, GA, log=lambda s: None)
    assert (info['h'], info['grav'], info['nongrav']) == (3, 3, 0)
