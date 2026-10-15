"""Tests for fortuity.hwcoh.classreps_par: non-graviton representatives with the parallel engine, and the block-restriction
channel test (fused vs core)."""
import numpy as np

from fortuity.bmn import dmk
from fortuity.bmn.classify import TargetSector, ratio
from fortuity.bmn.fullindex import partition_dims
from fortuity.bmn.modp import P
from fortuity.hwcoh import classreps_par as cp
from fortuity.hwcoh.classreps import hw_nongraviton_reps
from fortuity.hwcoh.dims import partition_dims_nf
from fortuity.hwcoh.gravhw import GravitonAlgebra
from fortuity.hwcoh.restrict import certified_lower_rank


def test_su2_threshold_class(tmp_path):
    # SU(2) (2J; q) = (5; 4,4,4): one non-graviton class, O_0 = O^2_24 (R-001); m = 11, r_up = 4, grav = 0
    dims = partition_dims_nf(2, 26)
    GA = GravitonAlgebra(2)
    lam, n = (4, 4, 4), 5
    r_down = certified_lower_rank(2, lam, n, dims, GA, 'SU')
    reps, info = cp.nongraviton_reps(2, lam, n, dims, GA, r_down, expect=dict(m=11, r_up=4, grav=0, h=1, bh=1),
                                     log=lambda *a: None, workdir=str(tmp_path))
    assert len(reps) == 1 and info['h'] == 1
    ser, sinfo = hw_nongraviton_reps(2, 'SU', lam, n, dims, GA, log=lambda *a: None)
    assert sinfo['nongrav'] == 1
    T = TargetSector(2, 'SU', (5, 4, 4, 4), gen_nmax=3, seed=17, log=lambda *a: None, maxlen=3, exact_dim=153)
    ref = T.residue(dmk.to_modp(dmk.o2_24()))
    for r in (reps[0], ser[0]):
        res = T.residue(r)
        assert res.any() and ratio(res, ref) is not None          # a nonzero multiple of O_0 modulo exact + graviton
    assert not list(tmp_path.iterdir())                           # memmaps removed


def test_channel_residues_on_dmk_operator():
    # dMK's O^3_30 (R-016 (a')): its x1^3 channel is -648/5 O^2_24, so it is fused; O - O has no channel at all
    O = dmk.to_modp(dmk.o330())
    dims2 = {k: v for k, v in partition_dims(2, 26, chunk=4).items()}
    chan = cp.channel_residues([O, O], 3, (7, 4, 4), 5, [(3, 0, 0)], dims2, log=lambda *a: None)
    c = chan[0]
    assert c['sector'] == [5, 4, 4, 4] and all(x['non_graviton_component'] for x in c['classify'])
    T = TargetSector(2, 'SU', (5, 4, 4, 4), gen_nmax=3, seed=17, log=lambda *a: None, maxlen=3, exact_dim=153)
    ref = T.residue(dmk.to_modp(dmk.o2_24()))
    lam_ = ratio(c['residues'][0], ref)
    assert lam_ is not None and (lam_ * 5 + 648) % P == 0         # -648/5 mod p
    rank, K = cp.cores_from_residues(chan)
    assert rank == 1 and K.shape == (1, 2) and (K[0, 0] + K[0, 1]) % P == 0
