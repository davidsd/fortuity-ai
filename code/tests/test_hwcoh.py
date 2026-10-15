"""Tests for fortuity.hwcoh (SU(3)_R highest-weight cohomology); the column test compares with symdirect (~1 min)."""
import os
import numpy as np
from fortuity.bmn import words as W
from fortuity.bmn.fullindex import partition_dims
from fortuity.bmn.symdirect import SymDirect
from fortuity.bmn.words import contents_of_sector
from fortuity.hwcoh import wordenum as enum
from fortuity.hwcoh.dims import partition_dims_nf, collapse
from fortuity.hwcoh.column import Column, SHIFTS
from fortuity.hwcoh.gravhw import GravitonAlgebra
from fortuity.hwcoh.hw import nullspace_left
from fortuity.bmn.modp import matmul_mod, P


def test_enum_matches_old():
    for sec in [(3, 3, 3, 2), (2, 3, 2, 2), (5, 3, 3, 3), (3, 4, 3, 3)]:
        assert sorted(W.words_of_sector(sec, 100)) == sorted(enum.words_of_sector(sec))


def test_dims_nf_collapse():
    assert collapse(partition_dims_nf(3, 14)) == partition_dims(3, 14)


def test_nullspace():
    rng = np.random.default_rng(0)
    A = matmul_mod(rng.integers(0, P, (60, 40)), rng.integers(0, P, (40, 90)))
    K = nullspace_left(A)
    assert K.shape[0] == 20 and not matmul_mod(K, A).any()


def test_hw_column_vs_symdirect():
    N, j = 3, 14
    dims = partition_dims(N, j)
    GA = GravitonAlgebra(N)
    D = SymDirect(N, symmetrize=False, maxlen=100, verbose=False); D.guard.log = False
    cache = {}
    for lam in [(3, 2, 2), (4, 2, 1)]:
        rows = Column(N, lam, dims, verbose=False).run(grav_rows=lambda n, fr: GA.rows((n,) + lam, fr))
        for r in rows:
            n = r['n']
            ref = refg = 0
            for sh, s in SHIFTS:
                q = tuple(a + b for a, b in zip(lam, sh))
                sec = (n,) + tuple(sorted(q, reverse=True))
                if min(q) < 0 or not contents_of_sector(sec):
                    continue
                if sec not in cache:
                    cache[sec] = D.analyse(sec)   # symdirect.analyse is not re-entrant per sector
                x = cache[sec]
                ref += s * x['coh']; refg += s * x['grav']
            assert r['check'] and r['h'] == ref and r['grav'] == refg, (lam, r, ref, refg)




def test_sectorpar_matches_sector():
    """The parallel, memmap-based engine reproduces the reference engine sector by sector."""
    import tempfile
    from fortuity.hwcoh.sector import Sector
    from fortuity.hwcoh.sectorpar import SectorPar
    N, lam = 3, (3, 2, 2)
    dims = partition_dims_nf(N, 14)
    GA = GravitonAlgebra(N)
    with tempfile.TemporaryDirectory() as tmp:
        for n in range(0, 6):
            a = Sector(N, lam, n, dims, log=lambda s: None, graviton_algebra=GA).run()
            b = SectorPar(N, lam, n, dims, log=lambda s: None, graviton_algebra=GA, workers=2, workdir=tmp).run()
            for key in ('m', 'check', 'r_up', 'grav'):
                assert a[key] == b[key], (n, key, a[key], b[key])


def test_hw_vectors_pool_matches_serial():
    """The forked-pool path of hw.hw_vectors (Linux only; largest shape in the parent, the rest in single-threaded
    workers) must give exactly the serial blocks."""
    import sys, pytest
    if sys.platform == 'darwin':
        pytest.skip("forked hw pool is disabled on macOS")
    from fortuity.hwcoh.hw import hw_vectors
    ws = enum.words_of_sector((4, 4, 3, 3))
    a = hw_vectors(ws, workers=1)
    b = hw_vectors(ws, workers=4)
    assert len(a) == len(b)
    for (ia, Ha), (ib, Hb) in zip(a, b):
        assert np.array_equal(ia, ib) and np.array_equal(Ha, Hb)
